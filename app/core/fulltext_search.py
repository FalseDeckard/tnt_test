import logging
from pathlib import Path

import numpy as np
from rank_bm25 import BM25Okapi

from app.core.ranking import normalize_positive_scores
from app.core.text_processing import TextPreprocessor
from app.data.documents import read_documents


class TextSearch:
    """Класс для полнотекстового поиска с использованием BM25 и кэшированием запросов.
    
    Args:
        data_dir (str): Путь к директории с обработанными данными
        batch_size (int): Размер батча для пакетной обработки запросов

    Attributes:
        documents (List[Dict]): Загруженные документы
        bm25 (BM25Okapi): Поисковый индекс BM25
        query_cache (Dict): Кэш обработанных запросов
        morph (pymorphy3.MorphAnalyzer): Морфологический анализатор
        stop_words (Set[str]): Стоп-слова русского языка
    """
    
    def __init__(self, data_dir = "data/processed", batch_size = 32):
        """Инициализация поисковой системы и загрузка данных."""
        logging.basicConfig(
            level=logging.INFO,
            format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
        )
        self.logger = logging.getLogger(__name__)
        self.data_dir = Path(data_dir)
        self.batch_size = batch_size
        self.query_cache = {}
        self.max_cache_size = 1000
        
        self.text_processor = TextPreprocessor()
        self.load_data()
        self.init_bm25()
        self.logger.info("Text search initialized successfully")

    def load_data(self):
        """Загрузка предобработанных документов из JSONL-файла.
        
        Raises:
            FileNotFoundError: Если файл с данными не найден
            JSONDecodeError: При ошибках парсинга JSON
            Exception: При других неожиданных ошибках
        """
        try:
            docs_path = self.data_dir / "processed_documents.jsonl"
            self.logger.info(f"Loading documents from {docs_path}")
            
            self.documents = read_documents(docs_path)
            
            self.logger.info(f"Loaded {len(self.documents)} documents")
        except FileNotFoundError as e:
            self.logger.error(f"Data file not found: {e}")
            raise
        except (TypeError, ValueError) as e:
            self.logger.error(f"Invalid document artifact: {e}")
            raise
        except Exception as e:
            self.logger.error(f"Unexpected error loading data: {e}")
            raise

    def init_bm25(self):
        """Инициализация поискового индекса BM25.
        
        Использует предобработанные тексты документов (text_processed).
        
        Raises:
            Exception: При ошибках создания индекса
        """
        try:
            corpus = [doc['text_processed'].split() for doc in self.documents]
            self.bm25 = BM25Okapi(corpus)
            self.logger.info("BM25 index initialized")
        except Exception as e:
            self.logger.error(f"Error initializing BM25: {e}")
            raise

    def _cache_query(self, query, processed):
        """Кэширование обработанных запросов с LRU-логикой.
        
        Args:
            query (str): Оригинальный текст запроса
            processed (str): Обработанная версия запроса
        """
        if len(self.query_cache) >= self.max_cache_size:
            self.query_cache.pop(next(iter(self.query_cache)))
        self.query_cache[query] = processed

    def process_text(self, text):
        """Обработка текста запроса с кэшированием результатов.
        
        Args:
            text (str): Входной текст запроса
            
        Returns:
            str: Обработанный текст (лемматизация + очистка)
        """
        if not isinstance(text, str):
            return ""
            
        if text in self.query_cache:
            return self.query_cache[text]
        
        processed = self.text_processor.full_clean(text)
        self._cache_query(text, processed)
        return processed

    def batch_search(self, queries, top_k = 5):
        """Пакетный поиск по нескольким запросам.
        
        Args:
            queries (List[str]): Список поисковых запросов
            top_k (int): Количество возвращаемых результатов на запрос
            
        Returns:
            List[List[Dict]]: Результаты поиска для каждого запроса
            
        Raises:
            Exception: При ошибках во время поиска
        """
        all_results = []
        limit = min(top_k, len(self.documents))
        for i in range(0, len(queries), self.batch_size):
            batch = queries[i:i + self.batch_size]
            for query in batch:
                query_tokens = self.process_text(query).split()
                if not query_tokens:
                    all_results.append([])
                    continue

                scores = self.bm25.get_scores(query_tokens)
                top_indices = np.argsort(scores)[::-1][:limit]
                ranked_scores = normalize_positive_scores(scores, top_indices)
                results = []
                for idx, score in ranked_scores:
                    doc = self.documents[idx]
                    results.append({
                        'title': doc['title'],
                        'summary': doc['summary'],
                        'url': doc['url'],
                        'date': doc['date'],
                        'score': score,
                    })
                all_results.append(results)

        return all_results

    def search(self, query, top_k= 5):
        """Поиск по одному запросу.
        
        Args:
            query (str): Поисковый запрос
            top_k (int): Количество возвращаемых результатов
            
        Returns:
            List[Dict]: Список результатов поиска
        """
        return self.batch_search([query], top_k)[0]

    def cleanup(self):
        """Очистка ресурсов и кэшей."""
        try:
            self.query_cache.clear()
            self.logger.info("Cleanup completed successfully")
        except Exception as e:
            self.logger.error(f"Error during cleanup: {e}")

    def __del__(self):
        """Деструктор для автоматической очистки ресурсов."""
        self.cleanup()
