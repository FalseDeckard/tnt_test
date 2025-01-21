from elasticsearch import Elasticsearch
from elasticsearch.helpers import bulk
import json
from pathlib import Path
import logging
from typing import List, Dict, Optional
import time

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class FullTextSearch:
    """Класс для полнотекстового поиска с использованием Elasticsearch"""

    def __init__(
            self,
            index_name: str = "news_articles",
            host: str = "localhost",
            port: int = 9200
    ):
        """
        Инициализация поискового движка

        Args:
            index_name: имя индекса в Elasticsearch
            host: хост Elasticsearch
            port: порт Elasticsearch
        """
        self.index_name = index_name
        self.es = Elasticsearch([{'host': host, 'port': port}])

        # Настройки индекса для русского языка
        self.index_settings = {
            "settings": {
                "analysis": {
                    "analyzer": {
                        "russian_analyzer": {
                            "type": "custom",
                            "tokenizer": "standard",
                            "filter": [
                                "lowercase",
                                "russian_stop",
                                "russian_stemmer"
                            ]
                        }
                    }
                }
            },
            "mappings": {
                "properties": {
                    "title": {
                        "type": "text",
                        "analyzer": "russian_analyzer"
                    },
                    "text": {
                        "type": "text",
                        "analyzer": "russian_analyzer"
                    },
                    "processed_text": {
                        "type": "text",
                        "analyzer": "russian_analyzer"
                    },
                    "date": {
                        "type": "date",
                        "format": "yyyy-MM-dd||yyyy-MM-dd'T'HH:mm:ss||epoch_millis"
                    },
                    "url": {
                        "type": "keyword"
                    }
                }
            }
        }

    def create_index(self) -> None:
        """Создание индекса с настройками для русского языка"""
        try:
            if not self.es.indices.exists(index=self.index_name):
                self.es.indices.create(
                    index=self.index_name,
                    body=self.index_settings
                )
                logger.info(f"Индекс {self.index_name} успешно создан")
            else:
                logger.info(f"Индекс {self.index_name} уже существует")
        except Exception as e:
            logger.error(f"Ошибка при создании индекса: {e}")
            raise

    def index_documents(self, documents_path: str) -> None:
        """
        Индексация документов из JSON файла

        Args:
            documents_path: путь к JSON файлу с документами
        """
        try:
            with open(documents_path, 'r', encoding='utf-8') as f:
                documents = json.load(f)

            # Подготовка документов для bulk индексации
            actions = [
                {
                    "_index": self.index_name,
                    "_id": doc["id"],
                    "_source": doc
                }
                for doc in documents
            ]

            # Bulk индексация
            start_time = time.time()
            success, failed = bulk(
                self.es,
                actions,
                chunk_size=1000,
                request_timeout=300
            )
            end_time = time.time()

            logger.info(f"Индексация завершена за {end_time - start_time:.2f} секунд")
            logger.info(f"Успешно: {success}, Ошибок: {len(failed)}")

        except Exception as e:
            logger.error(f"Ошибка при индексации документов: {e}")
            raise

    def search(
            self,
            query: str,
            top_k: int = 10,
            min_score: float = 0.1
    ) -> List[Dict]:
        """
        Поиск документов по запросу

        Args:
            query: поисковый запрос
            top_k: количество возвращаемых документов
            min_score: минимальный score для включения в результаты

        Returns:
            Список найденных документов с их score
        """
        try:
            start_time = time.time()

            # Составляем поисковый запрос
            search_query = {
                "query": {
                    "multi_match": {
                        "query": query,
                        "fields": ["title^2", "text", "processed_text^1.5"],
                        "type": "best_fields",
                        "operator": "and",
                        "minimum_should_match": "70%"
                    }
                },
                "size": top_k,
                "_source": ["id", "title", "text", "date", "url"]
            }

            # Выполняем поиск
            response = self.es.search(
                index=self.index_name,
                body=search_query
            )

            end_time = time.time()

            # Формируем результаты
            results = []
            for hit in response['hits']['hits']:
                if hit['_score'] >= min_score:
                    doc = hit['_source']
                    doc['score'] = hit['_score']
                    results.append(doc)

            return {
                'results': results,
                'total_found': response['hits']['total']['value'],
                'execution_time': end_time - start_time
            }

        except Exception as e:
            logger.error(f"Ошибка при поиске: {e}")
            raise
