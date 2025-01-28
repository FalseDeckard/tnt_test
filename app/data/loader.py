from sentence_transformers import SentenceTransformer
import numpy as np
import torch
import json
import re
from tqdm import tqdm
from pathlib import Path
import gc
import logging
import pymorphy3
from nltk.tokenize import word_tokenize
from nltk.corpus import stopwords
import nltk

nltk.download('punkt')
nltk.download('stopwords')

class TextPreprocessor:
    """Класс для предварительной обработки текстовых данных.
    
    Выполняет:
    - Базовую очистку текста (удаление спецсимволов, нормализация пробелов)
    - Полную обработку для BM25 (токенизация, удаление стоп-слов, лемматизация)
    
    Attributes:
        stop_words (set): Множество стоп-слов русского языка
        morph (pymorphy3.MorphAnalyzer): Морфологический анализатор
    """
    
    def __init__(self):
        """Инициализирует стоп-слова и морфологический анализатор."""
        self.stop_words = set(stopwords.words('russian'))
        self.morph = pymorphy3.MorphAnalyzer()

    def basic_clean(self, text):
        """Выполняет базовую очистку текста для генерации эмбеддингов.
        
        Args:
            text (str): Исходный текст для обработки
            
        Returns:
            str: Очищенный текст с удаленными спецсимволами и нормализованными пробелами
        """
        if not isinstance(text, str):
            return ""
        text = re.sub(r'[^\w\s\.\,\!\?\-\'«»]+', ' ', text)
        text = re.sub(r'\s+', ' ', text)
        return text.strip()

    def full_clean(self, text: str):
        """Выполняет полную обработку текста для использования в BM25.
        
        Args:
            text (str): Исходный текст для обработки
            
        Returns:
            str: Обработанный текст с лемматизацией и удаленными стоп-словами
        """
        if not isinstance(text, str):
            return ""
        
        text = self.basic_clean(text)
        tokens = word_tokenize(text.lower(), language='russian')
        processed_tokens = []
        
        for token in tokens:
            if token not in self.stop_words:
                lemma = self.morph.parse(token)[0].normal_form
                processed_tokens.append(lemma)
        
        return ' '.join(processed_tokens)


class DatasetProcessor:
    """Основной класс для обработки датасета и генерации эмбеддингов.
    
    Args:
        model_name (str): Название модели Sentence Transformers
        batch_size (int): Размер батча для обработки
        output_dir (str): Директория для сохранения результатов
        input_file (str): Путь к входному файлу .jsonl
        device (str, optional): Устройство для вычислений (cuda/mps/cpu)
    
    Attributes:
        text_processor (TextPreprocessor): Экземпляр текстового препроцессора
        logger (logging.Logger): Логгер для записи процесса обработки
    """
    
    def __init__(self, 
                 model_name: str = "deepvk/USER-bge-m3",
                 batch_size: int = 16,
                 output_dir: str = "processed_data",
                 input_file: str = "dataset.jsonl",
                 device: str = None):
        
        self.model_name = model_name
        self.batch_size = batch_size
        self.output_dir = Path(output_dir)
        self.input_file = Path(input_file)
        self.output_dir.mkdir(exist_ok=True)
        self.text_processor = TextPreprocessor()
        
        if device is None:
            device = "cuda" if torch.cuda.is_available() else "mps" if torch.backends.mps.is_available() else "cpu"
        self.device = device
        
        logging.basicConfig(level=logging.INFO,
                          format='%(asctime)s - %(levelname)s - %(message)s')
        self.logger = logging.getLogger(__name__)
        self.logger.info(f"Using device: {self.device}")

    def load_jsonl(self):
        """Загружает документы из JSONL-файла.
        
        Returns:
            list: Список загруженных документов
            
        Raises:
            JSONDecodeError: При ошибке парсинга строки JSON
        """
        documents = []
        with open(self.input_file, 'r', encoding='utf-8') as f:
            for line in f:
                try:
                    doc = json.loads(line.strip())
                    documents.append(doc)
                except json.JSONDecodeError as e:
                    self.logger.error(f"Error parsing JSON line: {e}")
                    continue
        return documents

    def prepare_document(self, doc):
        """Подготавливает документ для обработки.
        
        Args:
            doc (dict): Исходный документ из JSONL
            
        Returns:
            dict: Обработанный документ с полями:
                - id: URL документа
                - date: Дата документа
                - url: URL документа
                - title/summary/text: Базово очищенные тексты
                - *_processed: Полностью обработанные тексты
        """
        try:
            return {
                'id': doc['url'],
                'date': doc['date'],
                'url': doc['url'],
                'title': self.text_processor.basic_clean(doc['title']),
                'summary': self.text_processor.basic_clean(doc['summary']),
                'text': self.text_processor.basic_clean(doc['text']),
                'title_processed': self.text_processor.full_clean(doc['title']),
                'summary_processed': self.text_processor.full_clean(doc['summary']),
                'text_processed': self.text_processor.full_clean(doc['text']),
            }
        except Exception as e:
            self.logger.error(f"Error preparing document: {str(e)}")
            return None

    @torch.no_grad()
    def generate_embeddings(self, texts, model: SentenceTransformer):
        """Генерирует эмбеддинги для списка текстов.
        
        Args:
            texts (list): Список текстов для обработки
            model (SentenceTransformer): Модель для генерации эмбеддингов
            
        Returns:
            numpy.ndarray: Массив эмбеддингов формы (N, D)
        """
        try:
            if self.device == "mps":
                model = model.to(self.device)
                embeddings = model.encode(
                    texts,
                    normalize_embeddings=True,
                    show_progress_bar=False,
                    batch_size=self.batch_size,
                    convert_to_tensor=True
                )
                embeddings = embeddings.cpu().numpy()
            else:
                embeddings = model.encode(
                    texts,
                    normalize_embeddings=True,
                    show_progress_bar=False,
                    batch_size=self.batch_size,
                    device=self.device
                )
            return embeddings
        except Exception as e:
            self.logger.error(f"Error generating embeddings: {e}")
            return None

    def process_dataset(self):
        """Основной пайплайн обработки датасета.
        
        Returns:
            tuple: Кортеж с результатами обработки:
                - docs (list): Обработанные документы
                - embeddings (numpy.ndarray): Массив эмбеддингов
                - id_mapping (dict): Маппинг URL -> индекс документа
                
        Raises:
            Exception: При критических ошибках обработки
        """
        try:
            self.logger.info("Loading dataset and model...")
            documents = self.load_jsonl()
            model = SentenceTransformer(self.model_name)
            
            all_docs = []
            all_embeddings = []
            
            total_batches = len(documents) // self.batch_size + (1 if len(documents) % self.batch_size != 0 else 0)
            self.logger.info(f"Processing {len(documents)} documents in {total_batches} batches...")
            
            for i in tqdm(range(0, len(documents), self.batch_size)):
                batch = documents[i:i + self.batch_size]
                processed_docs = [self.prepare_document(doc) for doc in batch]
                processed_docs = [doc for doc in processed_docs if doc is not None]
                
                if processed_docs:
                    texts = [doc['text'] for doc in processed_docs]
                    batch_embeddings = self.generate_embeddings(texts, model)
                    
                    if batch_embeddings is not None:
                        all_docs.extend(processed_docs)
                        all_embeddings.append(batch_embeddings)
                
                if i % (self.batch_size * 10) == 0:
                    gc.collect()
                    if self.device == "cuda":
                        torch.cuda.empty_cache()
                    elif self.device == "mps":
                        torch.mps.empty_cache()
            
            final_embeddings = np.vstack(all_embeddings)
            self.logger.info(f"Final embeddings shape: {final_embeddings.shape}")
            
            np.save(self.output_dir / "embeddings.npy", final_embeddings)
            
            with (self.output_dir / "documents.json").open('w', encoding='utf-8') as f:
                json.dump(all_docs, f, ensure_ascii=False, indent=2)
            
            id_mapping = {doc['id']: idx for idx, doc in enumerate(all_docs)}
            with (self.output_dir / "id_mapping.json").open('w', encoding='utf-8') as f:
                json.dump(id_mapping, f, ensure_ascii=False)
            
            self.logger.info(f"Processing complete. Total documents: {len(all_docs)}")
            return all_docs, final_embeddings, id_mapping
            
        except Exception as e:
            self.logger.error(f"Error during processing: {e}")
            raise

if __name__ == "__main__":
    processor = DatasetProcessor(
        batch_size=16,
        output_dir="data/processed",
        input_file="data/raw/gazeta_test.jsonl",
        device="cuda"
    )
    
    docs, embeddings, id_mapping = processor.process_dataset()