from typing import List, Dict
import numpy as np
import logging
from .fulltext_search import TextSearch
from .vector_search import VectorSearch

class HybridSearch:
    def __init__(
        self,
        text_search: TextSearch,
        vector_search: VectorSearch,
        weights: Dict[str, float] = None
    ):
        """
        Initialize hybrid search with configured search engines
        """
        logging.basicConfig(
            level=logging.INFO,
            format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
        )
        self.logger = logging.getLogger(__name__)
        
        self.text_search = text_search
        self.vector_search = vector_search
        
        # Set default weights if none provided
        self.weights = weights or {"bm25": 0.5, "vector": 0.5}
        self._validate_weights()
        
        self.logger.info(f"Hybrid search initialized with weights: {self.weights}")

    def _validate_weights(self):
        """Validate that weights sum to 1.0"""
        total = sum(self.weights.values())
        if not np.isclose(total, 1.0, rtol=1e-5):
            self.logger.warning(f"Weights sum to {total}, normalizing...")
            for key in self.weights:
                self.weights[key] /= total
            self.logger.info(f"Normalized weights: {self.weights}")

    def set_weights(self, weights: Dict[str, float]):
        """Update search weights"""
        self.logger.info(f"Updating weights from {self.weights} to {weights}")
        self.weights = weights
        self._validate_weights()

    def batch_search(self, queries: List[str], top_k: int = 5) -> List[List[Dict]]:
        """
        Perform hybrid search for multiple queries
        """
        try:
            self.logger.debug(f"Starting batch search for {len(queries)} queries")
            results = []
            
            # Получаем результаты для каждого запроса отдельно
            for query in queries:
                results.append(self.search(query, top_k))
            
            return results
            
        except Exception as e:
            self.logger.error(f"Error during batch search: {str(e)}")
            return [[] for _ in queries]

    def search(self, query: str, top_k: int = 5) -> List[Dict]:
        """
        Perform hybrid search combining text and vector search results
        """
        try:
            self.logger.debug(f"Starting hybrid search for query: {query}")
            
            # Get results from both methods
            text_results = self.text_search.search(query, top_k=top_k*2)
            vector_results = self.vector_search.search(query, top_k=top_k*2)
            
            # Combine and normalize scores
            doc_scores = {}
            
            # Add text search scores
            for result in text_results:
                doc_scores[result['url']] = {
                    'score': result['score'] * self.weights['bm25'],
                    'data': result
                }
            
            # Add vector search scores
            for result in vector_results:
                if result['url'] in doc_scores:
                    doc_scores[result['url']]['score'] += (
                        result['score'] * self.weights['vector']
                    )
                else:
                    doc_scores[result['url']] = {
                        'score': result['score'] * self.weights['vector'],
                        'data': result
                    }
            
            # Sort by combined score
            sorted_results = sorted(
                doc_scores.items(),
                key=lambda x: x[1]['score'],
                reverse=True
            )
            
            # Take top_k results
            results = [item[1]['data'] for item in sorted_results[:top_k]]
            
            self.logger.debug(f"Hybrid search completed, found {len(results)} results")
            return results
            
        except Exception as e:
            self.logger.error(f"Error in hybrid search: {str(e)}")
            return []

    def _combine_results(
        self,
        bm25_results: List[Dict],
        vector_results: List[Dict],
        top_k: int
    ) -> List[Dict]:
        """Объединяет результаты разных методов поиска с учетом весов.
        
        Args:
            bm25_results (List[Dict]): Результаты текстового поиска
            vector_results (List[Dict]): Результаты векторного поиска
            top_k (int): Максимальное количество результатов
            
        Returns:
            List[Dict]: Отсортированный список результатов с комбинированными оценками
        """
        try:
            self.logger.debug("Combining BM25 and vector results")
            doc_scores = {}
            
            # Взвешивание оценок BM25
            for result in bm25_results:
                doc_scores[result['url']] = {
                    'score': result['score'] * self.weights['bm25'],
                    'data': result
                }
            
            # Добавление векторных оценок
            for result in vector_results:
                url = result['url']
                vector_score = result['score'] * self.weights['vector']
                
                if url in doc_scores:
                    doc_scores[url]['score'] += vector_score
                else:
                    doc_scores[url] = {'score': vector_score, 'data': result}
            
            # Сортировка по убыванию комбинированной оценки
            sorted_results = sorted(
                doc_scores.items(),
                key=lambda x: x[1]['score'],
                reverse=True
            )
            
            self.logger.debug(f"Combined and sorted {len(sorted_results)} results")
            return [item[1]['data'] for item in sorted_results]
            
        except Exception as e:
            self.logger.error(f"Error combining results: {str(e)}")
            return []
