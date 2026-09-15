import logging
from pathlib import Path
from typing import ClassVar

import faiss
import numpy as np
import torch
from sentence_transformers import SentenceTransformer

from app.core.model_config import MODEL_NAME, MODEL_REVISION
from app.core.vector_results import bounded_top_k, format_results, validate_artifacts
from app.data.documents import read_documents


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
    
    _models: ClassVar[dict[tuple[str, str, str], SentenceTransformer]] = {}

    def __init__(self, 
                 data_dir: str = "data/processed",
                 model_name = MODEL_NAME,
                 model_revision = MODEL_REVISION,
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
        self._initialize_model(model_name, model_revision)
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
            self.documents = read_documents(docs_path)
            
            # Загрузка эмбеддингов из .npy файла
            self.embeddings = np.load(
                self.data_dir / "embeddings.npy",
                allow_pickle=False,
            )
            validate_artifacts(len(self.documents), self.embeddings.shape)
            if not np.isfinite(self.embeddings).all():
                raise ValueError("Embeddings contain non-finite values")
            
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
                self._gpu_resources = faiss.StandardGpuResources()
                self.index = faiss.index_cpu_to_gpu(
                    self._gpu_resources, 0, self.index)
            
            self.logger.info(f"FAISS index initialized (dim={dimension})")
        except Exception as e:
            self.logger.error(f"FAISS init error: {e}")
            raise

    def _initialize_model(self, model_name, model_revision):
        """Инициализирует или получает кэшированную модель эмбеддингов."""
        cache_key = (model_name, model_revision, self.device)
        if cache_key not in self._models:
            self.logger.info(f"Loading model: {model_name}@{model_revision}")
            self._models[cache_key] = SentenceTransformer(
                model_name,
                revision=model_revision,
                device=self.device,
            )
        self.model = self._models[cache_key]

    def batch_encode(self, queries):
        """Кодирует список запросов в векторные представления.
        
        Args:
            queries (List[str]): Список текстовых запросов
            
        Returns:
            np.ndarray: Массив эмбеддингов формы (N, D)
        """
        if not queries:
            return np.empty((0, self.index.d), dtype=np.float32)

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

    def batch_search(self, queries, top_k = 5):
        if not queries:
            return []

        query_embeddings = self.batch_encode(queries)
        if query_embeddings.shape[1] != self.index.d:
            raise ValueError(
                "Query and document embedding dimensions differ: "
                f"{query_embeddings.shape[1]} != {self.index.d}"
            )

        limit = bounded_top_k(top_k, len(self.documents))
        distances, indices = self.index.search(
            query_embeddings.astype(np.float32),
            limit,
        )
        return [
            format_results(self.documents, query_indices, query_scores)
            for query_indices, query_scores in zip(indices, distances)
        ]

    def search(self, query, top_k = 5):
        """Поиск по одному запросу.
        
        Args:
            query (str): Текст запроса
            top_k (int): Количество возвращаемых результатов
            
        Returns:
            List[Dict]: Отсортированные результаты поиска
        """
        return self.batch_search([query], top_k)[0]

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
