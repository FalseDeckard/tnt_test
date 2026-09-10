# Search Engine TNT

Поисковый сервис для русскоязычных новостных документов. Приложение предоставляет
векторный, полнотекстовый и гибридный поиск через FastAPI API и простой веб-интерфейс.

## Возможности

- семантический поиск на базе `deepvk/USER-bge-m3` и FAISS;
- полнотекстовый поиск BM25 с лемматизацией через `pymorphy3`;
- гибридное ранжирование с настраиваемыми весами;
- до 20 запросов за одно API-обращение;
- MRR, Recall@k и nDCG@k для оценки по ручной разметке;
- liveness/readiness endpoints для эксплуатации в контейнере.

## Как устроен поиск

Векторный поиск кодирует запрос моделью Sentence Transformers и ищет ближайшие
нормализованные эмбеддинги через FAISS `IndexFlatIP`.

Текстовый поиск применяет общий для документов и запросов препроцессор: приводит
текст к нижнему регистру, выделяет Unicode-слова, удаляет русские стоп-слова и
лемматизирует токены. NLTK и загрузка внешних NLP-корпусов не используются.

Гибридный поиск получает кандидатов из обоих источников и вычисляет:

```text
hybrid_score = bm25_score × bm25_weight + vector_score × vector_weight
```

По умолчанию используются веса `0.5/0.5`. Поле `score` — внутренний показатель
ранжирования выбранного метода, а не вероятность релевантности.

## Структура проекта

```text
app/
├── api/          # HTTP endpoints и жизненный цикл поисковых движков
├── core/         # BM25, vector search, hybrid search и препроцессинг
├── data/         # чтение артефактов и подготовка датасета
├── evaluation/   # qrels и стандартные ranking-метрики
├── models/       # Pydantic-схемы API
└── templates/    # веб-интерфейс
data/
├── raw/          # исходный JSONL
├── processed/    # документы, embeddings.npy и id_mapping.json
└── evaluation/   # разметка и отчёты оценки
tests/            # unit-тесты
```

## Быстрый запуск через Docker

Понадобятся Docker, доступ в интернет и место для модели и поисковых артефактов.

```bash
docker build -t search-engine-tnt .
docker run --rm -p 8000:8000 search-engine-tnt
```

Во время `docker build`:

1. устанавливаются Python-зависимости;
2. скачиваются документы и эмбеддинги;
3. проверяются SHA-256 всех загруженных артефактов;
4. заранее скачивается модель эмбеддингов.

Приложение внутри контейнера работает от непривилегированного пользователя.

После запуска доступны:

- веб-интерфейс: <http://localhost:8000/>;
- Swagger UI: <http://localhost:8000/api/docs>;
- ReDoc: <http://localhost:8000/api/redoc>;
- liveness: <http://localhost:8000/api/health/live>;
- readiness: <http://localhost:8000/api/health/ready>.

Docker-образ использует Python 3.14. Исходный код также проверяется CI на нескольких
поддерживаемых версиях Python.

## Локальный запуск

Минимальная поддерживаемая версия — Python 3.10. Для загрузки артефактов нужны
`wget` и доступная в системе утилита SHA-256.

```bash
python -m venv .venv
source .venv/bin/activate
python -m pip install --upgrade pip
python -m pip install -r requirements.txt
./download_data.sh
python -m uvicorn app.main:app --host 0.0.0.0 --port 8000
```

При запуске сервис загружает документы и эмбеддинги в память, строит BM25/FAISS
индексы и инициализирует модель. Readiness начинает возвращать `200` только после
успешной инициализации.

## API

Основной endpoint — `POST /api/search/`.

```bash
curl --request POST http://localhost:8000/api/search/ \
  --header 'Content-Type: application/json' \
  --data '{
    "queries": ["экономика России", "искусственный интеллект"],
    "method": "hybrid",
    "top_k": 5,
    "batch_size": 10,
    "weights": {"bm25": 0.5, "vector": 0.5}
  }'
```

Параметры:

- `queries` — от 1 до 20 непустых строк;
- `method` — `vector`, `text` или `hybrid`;
- `top_k` — от 1 до 20 результатов;
- `batch_size` — от 1 до 20 запросов;
- `weights` — веса `bm25` и `vector`, сумма должна быть равна `1.0`.

## Поисковые артефакты

`download_data.sh` получает четыре закреплённых файла:

- `data/raw/gazeta_test.jsonl`;
- `data/processed/processed_documents.jsonl`;
- `data/processed/embeddings.npy`;
- `data/processed/id_mapping.json`.

Загрузка выполняется во временный `.part`-файл. Целевой файл заменяется только после
проверки SHA-256, поэтому оборванная или изменённая загрузка не принимается.

Обработанный JSONL содержит, в частности, `title`, `summary`, `text`, `url`, `date`
и `text_processed`. Количество строк в документном артефакте должно совпадать с
первым измерением `embeddings.npy`.

## Оценка качества

Оценка требует ручной разметки релевантных URL. Формат показан в
`data/evaluation/qrels.example.json`:

```json
{
  "экономика": [
    "https://example.com/relevant-document-1",
    "https://example.com/relevant-document-2"
  ]
}
```

Скопируйте пример, замените URL реальными значениями из датасета и запустите:

```bash
python -m app.evaluation.evaluate \
  --qrels data/evaluation/qrels.json \
  --data-dir data/processed \
  --output-dir data/evaluation
```

Evaluator сравнит `vector`, `text` и `hybrid` по MRR, Recall@k и nDCG@k, а также
запишет среднюю и медианную задержку. Результат сохраняется в JSON-файл с UTC
timestamp в имени.

## Проверки и CI

Локальный быстрый набор проверок:

```bash
python -m pip install -r requirements-test.txt ruff==0.16.6
ruff check .
python -m compileall -q app tests
python -m unittest discover -s tests -v
git diff --check
```

Те же базовые проверки автоматически запускаются GitHub Actions для каждого PR и
push в `main`. Dependabot еженедельно проверяет Python-зависимости, Docker base image
и используемые GitHub Actions.

## Ограничения

- документы, эмбеддинги и индексы целиком находятся в памяти одного процесса;
- запуск требует скачанной модели и подготовленных артефактов;
- API пока не содержит аутентификации, rate limiting и квот;
- качество нельзя корректно сравнивать без размеченного qrels-набора;
- гибридный поиск использует линейное смешивание, без reranker или RRF.
