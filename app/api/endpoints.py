import logging
import time
from collections.abc import Sequence
from typing import Any

from fastapi import APIRouter, Depends, HTTPException, Request
from starlette.concurrency import run_in_threadpool

from app.models.schemas import SearchQuery, SearchResponse, TimedSearchResult

router = APIRouter()
logger = logging.getLogger(__name__)


class SearchManager:
    """
    Менеджер жизненного цикла поисковых систем.
    Обеспечивает инициализацию, выполнение поиска и очистку ресурсов.

    Атрибуты:
        data_dir (str): Путь к директории с обработанными данными
        searchers (dict): Словарь доступных поисковых систем
        _vector_search (VectorSearch): Экземпляр векторного поиска
        _text_search (TextSearch): Экземпляр текстового поиска
    """

    def __init__(self, data_dir: str = "data/processed"):
        """Инициализация менеджера с указанием директории данных."""
        self.data_dir = data_dir
        self.searchers: dict[str, Any] = {}
        self._vector_search: Any | None = None
        self._text_search: Any | None = None
        self.logger = logging.getLogger(__name__)

    def initialize(self):
        """
        Синхронная инициализация поисковых систем.

        Инициализирует:
        - Векторный поиск (на основе эмбеддингов)
        - Текстовый поиск (BM25)
        - Гибридный поиск (комбинация двух методов)

        Вызывает:
            Exception: Если произошла ошибка при инициализации
        """
        if not self.searchers:
            self.logger.info("Initializing search engines...")
            try:
                from app.core.fulltext_search import TextSearch
                from app.core.hybrid_search import HybridSearch
                from app.core.vector_search import VectorSearch

                # Инициализация базовых поисковых систем
                self._vector_search = VectorSearch(data_dir=self.data_dir)
                self.logger.info("Vector search initialized")

                self._text_search = TextSearch(data_dir=self.data_dir)
                self.logger.info("Text search initialized")

                # Создание гибридного поиска
                hybrid_search = HybridSearch(
                    text_search=self._text_search, vector_search=self._vector_search
                )

                # Формирование словаря доступных методов
                self.searchers = {
                    "vector": self._vector_search,
                    "text": self._text_search,
                    "hybrid": hybrid_search,
                }

                self.logger.info("All search engines initialized successfully")
            except Exception as e:
                self.logger.error(f"Failed to initialize search engines: {e}")
                raise

    def cleanup(self):
        """Освобождение ресурсов и сброс состояния поисковых систем."""
        if self.searchers:
            self.logger.info("Cleaning up search engines...")
            if self._vector_search:
                self._vector_search.cleanup()
            if self._text_search:
                self._text_search.cleanup()
            self.searchers = {}
            self._vector_search = None
            self._text_search = None
            self.logger.info("Cleanup completed")

    @staticmethod
    def _search_kwargs(query: SearchQuery) -> dict[str, Any]:
        kwargs: dict[str, Any] = {"top_k": query.top_k}
        if query.method == "hybrid":
            kwargs["weights"] = query.weights
        return kwargs

    @staticmethod
    def _batched(items: Sequence[str], batch_size: int):
        for start in range(0, len(items), batch_size):
            yield items[start : start + batch_size]

    async def search(self, query: SearchQuery) -> SearchResponse:
        """Run CPU-bound retrieval outside the server event loop."""
        return await run_in_threadpool(self._search, query)

    def _search(self, query: SearchQuery) -> SearchResponse:
        """
        Основной метод выполнения поиска.

        Аргументы:
            query (SearchQuery): Параметры поискового запроса

        Возвращает:
            SearchResponse: Результаты поиска с метриками производительности

        Вызывает:
            HTTPException: При ошибках инициализации или неверных параметрах
        """
        if not self.searchers:
            raise HTTPException(
                status_code=503, detail="Search engines not initialized"
            )

        if query.method not in self.searchers:
            raise HTTPException(
                status_code=400, detail=f"Invalid search method: {query.method}"
            )

        try:
            start_total_time = time.perf_counter()
            results: list[TimedSearchResult] = []
            searcher = self.searchers[query.method]
            search_kwargs = self._search_kwargs(query)

            # Обработка батчей запросов
            if len(query.queries) > 1:
                self.logger.info(
                    f"Processing batch of {len(query.queries)} queries with batch_size {query.batch_size}"
                )

                for batch in self._batched(query.queries, query.batch_size):
                    self.logger.debug(f"Processing batch of {len(batch)} queries")
                    batch_start_time = time.perf_counter()

                    # Пакетный поиск
                    search_results = searcher.batch_search(
                        queries=batch,
                        **search_kwargs,
                    )

                    batch_time = time.perf_counter() - batch_start_time
                    avg_query_time = batch_time / len(batch)

                    # Формирование результатов для батча
                    for query_text, batch_results in zip(batch, search_results):
                        results.append(
                            TimedSearchResult(
                                results=batch_results,
                                query=query_text,
                                time_taken=round(avg_query_time, 3),
                            )
                        )
            else:
                # Обработка одиночного запроса
                single_query = query.queries[0]
                self.logger.info(f"Processing single query: {single_query}")

                start_query_time = time.perf_counter()
                search_results = searcher.search(
                    query=single_query,
                    **search_kwargs,
                )

                query_time = time.perf_counter() - start_query_time
                results.append(
                    TimedSearchResult(
                        results=search_results,
                        query=single_query,
                        time_taken=round(query_time, 3),
                    )
                )

            total_time = time.perf_counter() - start_total_time

            response = SearchResponse(
                results=results,
                total_time=round(total_time, 3),
                method=query.method,
                total_queries=len(query.queries),
            )

            self.logger.info(
                f"Search completed: method={query.method}, "
                f"queries={len(query.queries)}, "
                f"time={response.total_time}s"
            )

            return response

        except Exception:
            self.logger.exception("Search operation failed")
            raise HTTPException(
                status_code=500,
                detail="Search operation failed",
            )


def get_search_manager(request: Request) -> SearchManager:
    return request.app.state.search_manager


@router.get("/health/live", summary="Проверить состояние процесса")
def liveness():
    return {"status": "ok"}


@router.get("/health/ready", summary="Проверить готовность поиска")
def readiness(manager: SearchManager = Depends(get_search_manager)):
    if not manager.searchers:
        raise HTTPException(status_code=503, detail="Search engines not initialized")
    return {"status": "ready"}


@router.post(
    "/search/",
    response_model=SearchResponse,
    summary="Выполнить поиск",
    description="""
    Выполняет поиск по заданным запросам с использованием выбранного метода.
    
    Доступные методы поиска:
    - **hybrid**: Комбинирует векторный и текстовый поиск для лучших результатов
    - **vector**: Использует векторные эмбеддинги для семантического поиска
    - **text**: Использует BM25 для текстового поиска
    
    Каждый запрос возвращает top_k наиболее релевантных документов.
    """,
    response_description="Результаты поиска с временем выполнения и оценками релевантности",
)
async def search(
    query: SearchQuery,
    manager: SearchManager = Depends(get_search_manager),
):
    """
    Основной endpoint для выполнения поиска документов.

    Параметры:
        - query: Объект запроса с параметрами поиска

    Возвращает:
        SearchResponse: Объект с результатами поиска, содержащий:
            - Список результатов для каждого запроса
            - Общее время выполнения
            - Использованный метод поиска
            - Количество обработанных запросов

    Возможные ошибки:
        - 400: Неверный метод поиска
        - 503: Поисковые системы не инициализированы
        - 500: Внутренняя ошибка при выполнении поиска
    """
    return await manager.search(query)
