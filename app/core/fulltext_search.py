from rank_bm25 import BM25Okapi
import json
from pathlib import Path
import numpy as np
from nltk.tokenize import word_tokenize
import nltk
import pymorphy3
from nltk.corpus import stopwords
import logging

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
    
    def __init__(self, data_dir = "processed_data", batch_size = 32):
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
        
        self.setup_text_processing()
        self.load_data()
        self.init_bm25()
        self.logger.info("Text search initialized successfully")

    def setup_text_processing(self):
        """Инициализация инструментов обработки текста.
        
        Raises:
            Exception: При ошибках загрузки NLP-ресурсов
        """
        try:
            self.morph = pymorphy3.MorphAnalyzer()
            nltk.download('punkt_tab', quiet=True)
            nltk.download('stopwords', quiet=True)
            self.stop_words = set(stopwords.words('russian'))
            self.logger.info("Text processing tools initialized")
        except Exception as e:
            self.logger.error(f"Error setting up text processing: {e}")
            raise

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
            
            self.documents = []
            with open(docs_path, 'r', encoding='utf-8') as f:
                for line in f:
                    doc = json.loads(line.strip())
                    self.documents.append(doc)
            
            self.logger.info(f"Loaded {len(self.documents)} documents")
        except FileNotFoundError as e:
            self.logger.error(f"Data file not found: {e}")
            raise
        except json.JSONDecodeError as e:
            self.logger.error(f"Error parsing JSON: {e}")
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
        
        try:
            tokens = word_tokenize(text.lower(), language='russian')
            processed_tokens = []
            for token in tokens:
                if token in self.stop_words or not any(c.isalpha() for c in token):
                    continue
                lemma = self.morph.parse(token)[0].normal_form
                processed_tokens.append(lemma)
            
            processed = ' '.join(processed_tokens)
            self._cache_query(text, processed)
            return processed
        except Exception as e:
            self.logger.error(f"Error processing text: {e}")
            return text

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
        try:
            all_results = []
            for i in range(0, len(queries), self.batch_size):
                batch = queries[i:i + self.batch_size]
                batch_results = []
                
                for query in batch:
                    processed_query = self.process_text(query)
                    query_tokens = processed_query.split()
                    
                    scores = self.bm25.get_scores(query_tokens)
                    top_indices = np.argsort(scores)[-top_k:][::-1]
                    top_scores = scores[top_indices]
                    
                    max_score = np.max(top_scores) if len(top_scores) > 0 else 1
                    normalized_scores = top_scores / max_score if max_score != 0 else top_scores
                    
                    results = []
                    for idx, score in zip(top_indices, normalized_scores):
                        doc = self.documents[idx]
                        results.append({
                            'title': doc['title'],
                            'summary': doc['summary'],
                            'url': doc['url'],
                            'date': doc['date'],
                            'score': float(score)
                        })
                    batch_results.append(results)
                
                all_results.extend(batch_results)
            
            return all_results
        except Exception as e:
            self.logger.error(f"Error during batch search: {e}")
            return [[] for _ in queries]

    def search(self, query, top_k= 5):
        """Поиск по одному запросу.
        
        Args:
            query (str): Поисковый запрос
            top_k (int): Количество возвращаемых результатов
            
        Returns:
            List[Dict]: Список результатов поиска
        """
        try:
            return self.batch_search([query], top_k)[0]
        except Exception as e:
            self.logger.error(f"Error during search: {e}")
            return []

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
