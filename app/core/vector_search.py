from sentence_transformers import SentenceTransformer
import numpy as np
import faiss
import torch
import json
from pathlib import Path
import logging

class VectorSearch:
    """Класс для векторного поиска с использованием FAISS и Sentence Transformers.
    
    Args:
        data_dir (str): Путь к директории с обработанными данными
        model_name (str): Название предобученной модели Sentence Transformers
        device (str, optional): Устройство для вычислений (cuda/mps/cpu)
        batch_size (int): Размер батча для обработки запросов

    Attributes:
        documents (List[Dict]): Загруженные документы с метаданными
        embeddings (np.ndarray): Массив векторных представлений документов
        index (faiss.Index): Поисковый индекс FAISS
        model (SentenceTransformer): Модель для кодирования текстов
    """
    
    _model_instance = None  # Кэш модели на уровне класса

    def __init__(self, 
                 data_dir: str = "processed_data",
                 model_name = "deepvk/USER-bge-m3",
                 device = None,
                 batch_size = 32):
        """Инициализация векторного поиска и загрузка данных."""
        logging.basicConfig(level=logging.INFO,
                          format='%(asctime)s - %(levelname)s - %(message)s')
        self.logger = logging.getLogger(__name__)
        self.data_dir = Path(data_dir)
        self.batch_size = batch_size
        
        # Автоматическое определение устройства
        self.device = device or ("cuda" if torch.cuda.is_available() else 
                               "mps" if torch.backends.mps.is_available() else 
                               "cpu")
        
        self.load_data()
        self.init_faiss()
        self._initialize_model(model_name)
        self.logger.info(f"VectorSearch initialized on {self.device}")

    def load_data(self):
        """Загружает предобработанные документы и векторные представления.
        
        Raises:
            FileNotFoundError: Если файлы данных не найдены
            Exception: При других ошибках загрузки
        """
        try:
            # Загрузка документов из JSONL
            docs_path = self.data_dir / "processed_documents.jsonl"
            self.documents = []
            with open(docs_path, 'r', encoding='utf-8') as f:
                for line in f:
                    self.documents.append(json.loads(line.strip()))
            
            # Загрузка эмбеддингов из .npy файла
            self.embeddings = np.load(self.data_dir / "embeddings.npy")
            
            self.logger.info(f"Loaded {len(self.documents)} documents and embeddings")
        except Exception as e:
            self.logger.error(f"Data loading error: {e}")
            raise

    def init_faiss(self):
        """Инициализирует FAISS индекс для быстрого поиска.
        
        Использует косинусное сходство через inner product (IndexFlatIP)
        с автоматической конвертацией на GPU при наличии.
        """
        try:
            dimension = self.embeddings.shape[1]
            self.index = faiss.IndexFlatIP(dimension)
            self.index.add(self.embeddings.astype(np.float32))
            
            # Перенос индекса на GPU при наличии
            if self.device == "cuda" and faiss.get_num_gpus() > 0:
                self.index = faiss.index_cpu_to_gpu(
                    faiss.StandardGpuResources(), 0, self.index)
            
            self.logger.info(f"FAISS index initialized (dim={dimension})")
        except Exception as e:
            self.logger.error(f"FAISS init error: {e}")
            raise

    def _initialize_model(self, model_name):
        """Инициализирует или получает кэшированную модель эмбеддингов."""
        if VectorSearch._model_instance is None:
            self.logger.info(f"Loading model: {model_name}")
            VectorSearch._model_instance = SentenceTransformer(
                model_name, device=self.device)
        self.model = VectorSearch._model_instance

    def batch_encode(self, queries):
        """Кодирует список запросов в векторные представления.
        
        Args:
            queries (List[str]): Список текстовых запросов
            
        Returns:
            np.ndarray: Массив эмбеддингов формы (N, D)
        """
        embeddings = []
        for i in range(0, len(queries), self.batch_size):
            batch = queries[i:i + self.batch_size]
            batch_emb = self.model.encode(
                batch,
                normalize_embeddings=True,
                convert_to_tensor=True,
                show_progress_bar=False
            )
            embeddings.append(batch_emb.cpu().numpy())
        return np.vstack(embeddings)

    def normalize_scores(self, distances):
        return (distances)
    
    def batch_search(self, queries, top_k = 5):
        try:
            # Получаем эмбеддинги для всех запросов
            query_embeddings = self.batch_encode(queries)
            
            # Поиск в FAISS
            distances, indices = self.index.search(
                query_embeddings.astype(np.float32), 
                top_k
            )
            
            # Нормализуем scores
            normalized_scores = [self.normalize_scores(d) for d in distances]
            
            # Форматируем результаты
            all_results = []
            for query_indices, query_scores in zip(indices, normalized_scores):
                results = []
                for idx, score in zip(query_indices, query_scores):
                    doc = self.documents[idx]
                    results.append({
                        'title': doc['title'],
                        'summary': doc['summary'],
                        'url': doc['url'],
                        'date': doc['date'],
                        'score': float(score),
                    })
                all_results.append(results)
            
            return all_results
        except Exception as e:
            self.logger.error(f"Error during batch search: {e}")
            return [[] for _ in queries]

    def search(self, query, top_k = 5):
        """Поиск по одному запросу.
        
        Args:
            query (str): Текст запроса
            top_k (int): Количество возвращаемых результатов
            
        Returns:
            List[Dict]: Отсортированные результаты поиска
        """
        try:
            return self.batch_search([query], top_k)[0]
        except Exception as e:
            self.logger.error(f"Search error: {e}")
            return []

    def cleanup(self):
        """Освобождает ресурсы и очищает GPU кеш."""
        try:
            if hasattr(self, 'index') and self.device == "cuda":
                self.index.reset()
            if self.device == "cuda":
                torch.cuda.empty_cache()
            self.logger.info("Resources cleaned up")
        except Exception as e:
            self.logger.error(f"Cleanup error: {e}")

    def __del__(self):
        """Гарантирует очистку ресурсов при удалении объекта."""
        self.cleanup()
