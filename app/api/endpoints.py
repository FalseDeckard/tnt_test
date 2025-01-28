# app/api/endpoints.py
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
    _instance: Optional['SearchManager'] = None
    _initialized = False
    
    def __new__(cls, *args, **kwargs):
        if not cls._instance:
            cls._instance = super(SearchManager, cls).__new__(cls)
        return cls._instance

    def __init__(self, data_dir: str = "data/processed"):
        if not SearchManager._initialized:
            self.data_dir = data_dir
            self.searchers = {}
            self._vector_search = None
            self._text_search = None
            self.logger = logging.getLogger(__name__)
            SearchManager._initialized = True

    async def initialize(self):
        """Initialize search engines if not already initialized"""
        if not self.searchers:
            self.logger.info("Initializing search engines...")
            try:
                # Initialize base searchers first
                self._vector_search = VectorSearch(data_dir=self.data_dir)
                self.logger.info("Vector search initialized")
                
                self._text_search = TextSearch(data_dir=self.data_dir)
                self.logger.info("Text search initialized")
                
                # Create hybrid search using initialized components
                hybrid_search = HybridSearch(
                    text_search=self._text_search,
                    vector_search=self._vector_search
                )
                
                # Create searchers dictionary
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
        """Cleanup resources"""
        if self.searchers:
            self.logger.info("Cleaning up search engines...")
            if self._vector_search:
                self._vector_search.cleanup()
            if self._text_search:
                self._text_search.cleanup()
            self.searchers = {}
            SearchManager._initialized = False
            self.logger.info("Cleanup completed")

# В app/api/endpoints.py обновите метод search в SearchManager

    async def search(self, query: SearchQuery) -> SearchResponse:
        """Perform search using specified method"""
        if not self.searchers:
            raise HTTPException(status_code=503, detail="Search engines not initialized")
            
        if query.method not in self.searchers:
            raise HTTPException(status_code=400, detail=f"Invalid search method: {query.method}")
        
        try:
            start_total_time = time.time()
            results: List[TimedSearchResult] = []
            
            # Обновляем веса для гибридного поиска если нужно
            if query.method == "hybrid":
                self.logger.info(f"Updating hybrid search weights: {query.weights}")
                self.searchers["hybrid"].set_weights(query.weights)
            
            # Используем batch_search если есть несколько запросов
            if len(query.queries) > 1:
                self.logger.info(f"Processing batch of {len(query.queries)} queries with batch_size {query.batch_size}")
                
                # Разбиваем запросы на батчи
                batches = [query.queries[i:i + query.batch_size] 
                        for i in range(0, len(query.queries), query.batch_size)]
                
                for batch in batches:
                    self.logger.debug(f"Processing batch of {len(batch)} queries")
                    batch_start_time = time.time()
                    
                    search_results = self.searchers[query.method].batch_search(
                        queries=batch,
                        top_k=query.top_k
                    )
                    
                    batch_time = time.time() - batch_start_time
                    avg_query_time = batch_time / len(batch)
                    
                    # Обрабатываем результаты батча
                    for query_text, batch_results in zip(batch, search_results):
                        results.append(TimedSearchResult(
                            results=batch_results,
                            query=query_text,
                            time_taken=round(avg_query_time, 3)
                        ))
            else:
                # Для одного запроса используем обычный search
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

# Create singleton instance
search_manager = SearchManager()

@router.post("/search/")
async def search(query: SearchQuery):
    """
    Perform search using specified method
    """
    return await search_manager.search(query)