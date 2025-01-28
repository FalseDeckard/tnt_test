import asyncio
import json
import numpy as np
from pathlib import Path
from datetime import datetime
import logging
from app.models.schemas import SearchQuery
from app.api.endpoints import search_manager

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class DatasetEvaluator:
    def __init__(self, data_dir="data/processed"):
        self.data_dir = Path(data_dir)
        self.methods = ["vector", "text", "hybrid"]
        self.k_values = [1, 3, 5, 10]
        self.documents = self._load_documents()
        self.test_queries = self._get_test_queries()

    def _load_documents(self):
        """Загружает основной датасет"""
        docs_path = self.data_dir / "processed_documents.jsonl"
        documents = []
        with open(docs_path, 'r', encoding='utf-8') as f:
            for line in f:
                documents.append(json.loads(line.strip()))
        return documents

    def _get_test_queries(self):
        """
        Предопределенные тестовые запросы разной сложности и характера
        """
        return [
            {
                "query": "экономика",
                "description": "Однословный базовый запрос"
            },
            {
                "query": "искусственный интеллект",
                "description": "Устойчивое словосочетание"
            },
            {
                "query": "новости про газпром",
                "description": "Запрос с упоминанием компании"
            },
            {
                "query": "отношения россии и китая",
                "description": "Тематический запрос с упоминанием стран"
            },
            {
                "query": "влияние санкций на экономику россии 2020",
                "description": "Сложный аналитический запрос с временным контекстом"
            },
            {
                "query": "цены на недвижимость в москве рост",
                "description": "Запрос с географической привязкой и уточнением"
            },
            {
                "query": "новые технологии в медицине 2020",
                "description": "Общий тематический запрос с временным контекстом"
            },
            {
                "query": "как изменился курс рубля",
                "description": "Запрос в форме вопроса"
            },
            {
                "query": "российская военная операция",
                "description": "Новостной тематический запрос"
            },
            {
                "query": "развитие зеленой энергетики",
                "description": "Запрос по развивающейся тематике"
            }
        ]

    def _format_time(self, seconds):
        """Форматирует время в читаемый вид"""
        if seconds < 1:
            return f"{seconds * 1000:.0f} мс"
        return f"{seconds:.2f} сек"

    async def evaluate_method(self, method):
        """Оценивает метод поиска"""
        results = {k: {
            'avg_score': [],
            'times': [],
            'top_scores': []
        } for k in self.k_values}

        for test_case in self.test_queries:
            query = test_case["query"]
            
            # Выполняем поиск
            search_query = SearchQuery(
                queries=[query],
                method=method,
                top_k=max(self.k_values)
            )
            response = await search_manager.search(search_query)
            found_docs = response.results[0].results

            # Рассчитываем метрики для каждого k
            for k in self.k_values:
                top_k_docs = found_docs[:k]
                
                # Средний score для top-k документов
                scores = [doc.score for doc in top_k_docs]
                results[k]['avg_score'].append(np.mean(scores) if scores else 0)
                results[k]['top_scores'].extend(scores)
                results[k]['times'].append(response.total_time)

        # Агрегируем результаты
        final_results = {}
        for k in self.k_values:
            final_results[k] = {
                'avg_score': np.mean(results[k]['avg_score']),
                'max_score': np.max(results[k]['top_scores']) if results[k]['top_scores'] else 0,
                'min_score': np.min(results[k]['top_scores']) if results[k]['top_scores'] else 0,
                'avg_time': np.mean(results[k]['times']),
                'std_time': np.std(results[k]['times'])
            }

        return final_results

    def _generate_report(self, results, example_results):
        """Генерирует читаемый отчет с детальной разбивкой по типам запросов"""
        report = [
            "=" * 80,
            "ОТЧЕТ ПО ОЦЕНКЕ МЕТОДОВ ПОИСКА",
            "=" * 80,
            f"\nДата оценки: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}",
            f"Размер датасета: {len(self.documents)} документов",
            f"Количество тестовых запросов: {len(self.test_queries)}"
        ]
        
        # Список запросов с описаниями
        report.extend([
            "\nТЕСТОВЫЕ ЗАПРОСЫ:",
            "-" * 80
        ])
        for i, query in enumerate(self.test_queries, 1):
            report.extend([
                f"{i}. {query['query']}",
                f"   {query['description']}"
            ])

        # Общая статистика времени выполнения
        report.extend([
            "\nОБЩЕЕ ВРЕМЯ ВЫПОЛНЕНИЯ:",
            "-" * 80,
            "Метод          Среднее         Стд. откл.      На запрос"
        ])

        for method in self.methods:
            avg_time = np.mean([results[method][k]['avg_time'] for k in self.k_values])
            std_time = np.mean([results[method][k]['std_time'] for k in self.k_values])
            per_query = avg_time / len(self.test_queries)
            report.append(
                f"{method:<14}{self._format_time(avg_time):<16}"
                f"{self._format_time(std_time):<16}{self._format_time(per_query)}"
            )

        # Детальный анализ по типам запросов
        report.extend([
            "\nДЕТАЛЬНЫЙ АНАЛИЗ ПО ТИПАМ ЗАПРОСОВ:",
            "=" * 80
        ])

        for i, query in enumerate(self.test_queries):
            report.extend([
                f"\nЗапрос {i+1}: {query['query']}",
                f"Тип запроса: {query['description']}",
                "-" * 80,
                "Метод          Score       Время       Топ документ"
            ])

            for method in self.methods:
                results_for_query = example_results[method][i]
                top_doc = results_for_query['docs'][0] if results_for_query['docs'] else None
                
                score_str = f"{top_doc.score:.3f}" if top_doc else "N/A"
                time_str = self._format_time(results_for_query['time'])
                title_str = (top_doc.title[:50] + "...") if top_doc else "Нет результатов"
                
                report.append(
                    f"{method:<14}{score_str:<12}{time_str:<12}{title_str}"
                )

        # Агрегированные метрики по k
        for k in self.k_values:
            report.extend([
                f"\nОБЩИЕ МЕТРИКИ ПРИ k={k}:",
                "-" * 80,
                "Метод          Средний score    Мин. score      Макс. score"
            ])
            
            for method in self.methods:
                metrics = results[method][k]
                report.append(
                    f"{method:<14}{metrics['avg_score']:.3f}{'':>8}"
                    f"{metrics['min_score']:.3f}{'':>8}"
                    f"{metrics['max_score']:.3f}"
                )

        # Примеры полных результатов для каждого метода
        report.extend([
            "\nПРИМЕРЫ ПОЛНЫХ РЕЗУЛЬТАТОВ (top-3):",
            "=" * 80
        ])

        for method in self.methods:
            report.extend([
                f"\nМетод: {method.upper()}",
                "-" * 40
            ])

            first_results = example_results[method][0]  # Берем первый запрос для примера
            report.append(f"Запрос: {first_results['query']}")
            report.append(f"Время: {self._format_time(first_results['time'])}")

            for i, doc in enumerate(first_results['docs'][:3], 1):
                report.extend([
                    f"\n{i}. {doc.title}",
                    f"   Релевантность: {doc.score:.3f}",
                    f"   {doc.summary[:100]}..."
                ])

        return "\n".join(report)

    async def run_evaluation(self):
        """Запускает оценку и генерирует отчет"""
        print("Начинаем оценку методов поиска...")
        
        # Оцениваем каждый метод
        all_results = {}
        example_results = {method: [] for method in self.methods}
        
        for method in self.methods:
            print(f"Оценка метода: {method}")
            all_results[method] = await self.evaluate_method(method)
            
            # Получаем результаты для каждого запроса
            for query in self.test_queries:
                search_query = SearchQuery(
                    queries=[query['query']],
                    method=method,
                    top_k=3
                )
                response = await search_manager.search(search_query)
                example_results[method].append({
                    'query': query['query'],
                    'time': response.total_time,
                    'docs': response.results[0].results
                })

        # Генерируем отчет
        report = self._generate_report(all_results, example_results)

        # Сохраняем отчет
        output_dir = Path("data/evaluation")
        output_dir.mkdir(parents=True, exist_ok=True)

        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        report_path = output_dir / f"search_evaluation_report_{timestamp}.txt"

        with open(report_path, 'w', encoding='utf-8') as f:
            f.write(report)

        print(f"\nОтчет сохранен в: {report_path}")
        print("\nКраткие результаты:")
        print(report.split('\n')[0:20])

if __name__ == "__main__":
    async def main():
        await search_manager.initialize()
        evaluator = DatasetEvaluator()
        await evaluator.run_evaluation()
        search_manager.cleanup()

    asyncio.run(main())