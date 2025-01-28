from fastapi import APIRouter, HTTPException
from app.models.schemas import SearchQuery, SearchResponse, TimedSearchResult
from app.core.vector_search import VectorSearch
from app.core.fulltext_search import TextSearch
from app.core.hybrid_search import HybridSearch
import logging
import time
from typing import List, Dict, Optional

router = APIRouter()
logger = logging.getLogger(__name__)

class SearchManager:
    """
    Класс-синглтон для управления поисковыми системами.
    Обеспечивает инициализацию, выполнение поиска и очистку ресурсов.
    
    Атрибуты:
        data_dir (str): Путь к директории с обработанными данными
        searchers (dict): Словарь доступных поисковых систем
        _vector_search (VectorSearch): Экземпляр векторного поиска
        _text_search (TextSearch): Экземпляр текстового поиска
    """
    _instance: Optional['SearchManager'] = None
    _initialized = False
    
    def __new__(cls, *args, **kwargs):
        """Реализация паттерна Singleton. Возвращает единственный экземпляр класса."""
        if not cls._instance:
            cls._instance = super(SearchManager, cls).__new__(cls)
        return cls._instance

    def __init__(self, data_dir: str = "data/processed"):
        """Инициализация менеджера с указанием директории данных (выполняется один раз)."""
        if not SearchManager._initialized:
            self.data_dir = data_dir
            self.searchers = {}
            self._vector_search = None
            self._text_search = None
            self.logger = logging.getLogger(__name__)
            SearchManager._initialized = True

    async def initialize(self):
        """
        Асинхронная инициализация поисковых систем.
        
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
                # Инициализация базовых поисковых систем
                self._vector_search = VectorSearch(data_dir=self.data_dir)
                self.logger.info("Vector search initialized")
                
                self._text_search = TextSearch(data_dir=self.data_dir)
                self.logger.info("Text search initialized")
                
                # Создание гибридного поиска
                hybrid_search = HybridSearch(
                    text_search=self._text_search,
                    vector_search=self._vector_search
                )
                
                # Формирование словаря доступных методов
                self.searchers = {
                    "vector": self._vector_search,
                    "text": self._text_search,
                    "hybrid": hybrid_search
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
            SearchManager._initialized = False
            self.logger.info("Cleanup completed")

    async def search(self, query: SearchQuery) -> SearchResponse:
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
            raise HTTPException(status_code=503, detail="Search engines not initialized")
            
        if query.method not in self.searchers:
            raise HTTPException(status_code=400, detail=f"Invalid search method: {query.method}")
        
        try:
            start_total_time = time.time()
            results: List[TimedSearchResult] = []
            
            # Обновление весов для гибридного поиска
            if query.method == "hybrid":
                self.logger.info(f"Updating hybrid search weights: {query.weights}")
                self.searchers["hybrid"].set_weights(query.weights)
            
            # Обработка батчей запросов
            if len(query.queries) > 1:
                self.logger.info(f"Processing batch of {len(query.queries)} queries with batch_size {query.batch_size}")
                
                # Разделение запросов на батчи
                batches = [query.queries[i:i + query.batch_size] 
                        for i in range(0, len(query.queries), query.batch_size)]
                
                for batch in batches:
                    self.logger.debug(f"Processing batch of {len(batch)} queries")
                    batch_start_time = time.time()
                    
                    # Пакетный поиск
                    search_results = self.searchers[query.method].batch_search(
                        queries=batch,
                        top_k=query.top_k
                    )
                    
                    batch_time = time.time() - batch_start_time
                    avg_query_time = batch_time / len(batch)
                    
                    # Формирование результатов для батча
                    for query_text, batch_results in zip(batch, search_results):
                        results.append(TimedSearchResult(
                            results=batch_results,
                            query=query_text,
                            time_taken=round(avg_query_time, 3)
                        ))
            else:
                # Обработка одиночного запроса
                single_query = query.queries[0]
                self.logger.info(f"Processing single query: {single_query}")
                
                start_query_time = time.time()
                search_results = self.searchers[query.method].search(
                    query=single_query,
                    top_k=query.top_k
                )
                
                query_time = time.time() - start_query_time
                results.append(TimedSearchResult(
                    results=search_results,
                    query=single_query,
                    time_taken=round(query_time, 3)
                ))
            
            total_time = time.time() - start_total_time
            
            response = SearchResponse(
                results=results,
                total_time=round(total_time, 3),
                method=query.method,
                total_queries=len(query.queries)
            )
            
            self.logger.info(
                f"Search completed: method={query.method}, "
                f"queries={len(query.queries)}, "
                f"time={response.total_time}s"
            )
            
            return response
                
        except Exception as e:
            self.logger.error(f"Search error: {e}")
            raise HTTPException(
                status_code=500, 
                detail=f"Search operation failed: {str(e)}"
            )

# Синглтон менеджера поиска
search_manager = SearchManager()

@router.post("/search/",
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
    response_description="Результаты поиска с временем выполнения и оценками релевантности"
)
async def search(query: SearchQuery):
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
    return await search_manager.search(query)
