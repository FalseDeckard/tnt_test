from typing import List, Dict
import numpy as np
import logging
from .fulltext_search import TextSearch
from .vector_search import VectorSearch

class HybridSearch:
    """
    Реализация гибридного поиска, объединяющего текстовый (BM25) и векторный методы.
    
    Особенности:
    - Комбинирует результаты двух методов поиска с настраиваемыми весами
    - Автоматическая нормализация весов
    - Поддержка пакетной обработки запросов
    - Интеграция с существующими системами текстового и векторного поиска
    
    Аргументы инициализации:
        text_search (TextSearch): Экземпляр текстового поиска
        vector_search (VectorSearch): Экземпляр векторного поиска
        weights (Dict[str, float], optional): Веса для методов (bm25 и vector)
    """
    
    def __init__(
        self,
        text_search: TextSearch,
        vector_search: VectorSearch,
        weights: Dict[str, float] = None
    ):
        # Настройка системы логирования
        logging.basicConfig(
            level=logging.INFO,
            format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
        )
        self.logger = logging.getLogger(__name__)
        
        self.text_search = text_search
        self.vector_search = vector_search
        
        # Инициализация весов с проверкой нормализации
        self.weights = weights or {"bm25": 0.5, "vector": 0.5}
        self._validate_weights()  # Проверка и корректировка весов
        
        self.logger.info(f"Гибридный поиск инициализирован с весами: {self.weights}")

    def _validate_weights(self):
        """
        Проверяет и нормализует веса для методов поиска.
        
        Логика:
        - Если сумма весов не равна 1, выполняется нормализация
        - Выводит предупреждение в лог при изменении весов
        """
        total = sum(self.weights.values())
        if not np.isclose(total, 1.0, rtol=1e-5):
            self.logger.warning(f"Сумма весов {total}, выполняется нормализация...")
            for key in self.weights:
                self.weights[key] /= total
            self.logger.info(f"Нормализованные веса: {self.weights}")

    def set_weights(self, weights: Dict[str, float]):
        """
        Обновление весов для методов поиска с автоматической нормализацией.
        
        Аргументы:
            weights (Dict[str, float]): Новые веса в формате {'bm25': x, 'vector': y}
        """
        self.logger.info(f"Обновление весов с {self.weights} на {weights}")
        self.weights = weights
        self._validate_weights()  # Повторная проверка после обновления

    def batch_search(self, queries: List[str], top_k: int = 5) -> List[List[Dict]]:
        """
        Пакетная обработка нескольких поисковых запросов.
        
        Аргументы:
            queries (List[str]): Список поисковых запросов
            top_k (int): Количество возвращаемых результатов для каждого запроса
            
        Возвращает:
            List[List[Dict]]: Список результатов для каждого запроса
            
        Логирование:
            - Записывает ошибки в лог без прерывания работы
        """
        try:
            self.logger.debug(f"Начало пакетной обработки {len(queries)} запросов")
            return [self.search(query, top_k) for query in queries]
        except Exception as e:
            self.logger.error(f"Ошибка пакетной обработки: {str(e)}")
            return [[] for _ in queries]

    def search(self, query: str, top_k: int = 5) -> List[Dict]:
        """
        Основной метод гибридного поиска для одного запроса.
        
        Этапы работы:
        1. Получение результатов от обоих методов поиска
        2. Объединение и взвешивание оценок
        3. Сортировка по комбинированной оценке
        4. Возврат топ-k результатов
        
        Аргументы:
            query (str): Поисковый запрос
            top_k (int): Количество возвращаемых результатов
            
        Возвращает:
            List[Dict]: Список результатов с полями:
                - url: Ссылка на документ
                - score: Общая оценка релевантности
                - text: Текст документа
                - metadata: Дополнительные данные
        """
        try:
            self.logger.debug(f"Старт гибридного поиска: '{query}'")
            
            # Получение расширенного набора результатов от каждого метода
            text_results = self.text_search.search(query, top_k=top_k*2)
            vector_results = self.vector_search.search(query, top_k=top_k*2)
            
            # Объединение и ранжирование результатов
            combined = self._combine_results(text_results, vector_results, top_k)
            
            self.logger.debug(f"Найдено результатов: {len(combined)}")
            return combined
            
        except Exception as e:
            self.logger.error(f"Ошибка поиска: {str(e)}")
            return []

    def _combine_results(
        self,
        bm25_results: List[Dict],
        vector_results: List[Dict],
        top_k: int
    ) -> List[Dict]:
        """
        Внутренний метод для объединения и ранжирования результатов.
        
        Алгоритм:
        1. Взвешивание оценок по каждому методу
        2. Суммирование оценок для общих документов
        3. Сортировка по убыванию общей оценки
        4. Выбор топ-k результатов
        
        Аргументы:
            bm25_results (List[Dict]): Результаты текстового поиска
            vector_results (List[Dict]): Результаты векторного поиска
            top_k (int): Требуемое количество результатов
            
        Возвращает:
            List[Dict]: Отсортированный список документов с комбинированными оценками
        """
        try:
            self.logger.debug("Объединение BM25 и векторных результатов")
            doc_scores = {}
            
            # Обработка текстовых результатов
            for result in bm25_results:
                doc_key = result['url']
                doc_scores[doc_key] = {
                    'score': result['score'] * self.weights['bm25'],
                    'data': {**result, 'score_type': 'bm25'}
                }
            
            # Обработка векторных результатов
            for result in vector_results:
                doc_key = result['url']
                weighted_score = result['score'] * self.weights['vector']
                
                if doc_key in doc_scores:
                    # Суммирование оценок для существующих документов
                    doc_scores[doc_key]['score'] += weighted_score
                    doc_scores[doc_key]['data']['score_type'] = 'hybrid'
                else:
                    # Добавление новых записей
                    doc_scores[doc_key] = {
                        'score': weighted_score,
                        'data': {**result, 'score_type': 'vector'}
                    }
            
            # Сортировка и выборка топ результатов
            sorted_results = sorted(
                doc_scores.values(),
                key=lambda x: x['score'],
                reverse=True
            )[:top_k]
            
            # Формирование финального представления
            return [
                {**item['data'], 'score': round(item['score'], 4)} 
                for item in sorted_results
            ]
            
        except Exception as e:
            self.logger.error(f"Ошибка объединения: {str(e)}")
            return []
