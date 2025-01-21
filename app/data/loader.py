from datasets import load_dataset
import logging
from pathlib import Path
from typing import Optional
import pandas as pd
from tqdm import tqdm
from pymorphy3 import MorphAnalyzer
import json

# Настройка логирования
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

class DataLoader:
    """Класс для загрузки и обработки новостного датасета Lenta.ru"""

    def __init__(
        self,
        sample_size: Optional[int] = None,
        data_dir: str = "data/processed"
    ):
        """
        Инициализация загрузчика данных

        Args:
            sample_size (int, optional): Размер выборки. None для загрузки всего датасета
            data_dir (str): Путь к директории для сохранения обработанных данных
        """
        self.sample_size = sample_size
        self.morph = MorphAnalyzer()
        self.data_dir = Path(data_dir)
        self.data_dir.mkdir(parents=True, exist_ok=True)

    def load_dataset(self) -> pd.DataFrame:
        """Загрузка Lenta.ru"""
        logger.info("Загрузка датасета Lenta.ru...")

        try:
            dataset = load_dataset("yhavinga/lenta.ru-news")
            df = pd.DataFrame(dataset['train'])

            if self.sample_size:
                df = df.sample(n=min(self.sample_size, len(df)), random_state=42)

            logger.info(f"Загружено {len(df)} документов")
            return df

        except Exception as e:
            logger.error(f"Ошибка при загрузке датасета: {e}")
            raise

    def clean_text(self, text: str) -> str:
        """Очистка текста"""
        text = text.lower().strip()
        text = ' '.join(text.split())  # Убираем множественные пробелы
        return text

    def lemmatize_text(self, text: str) -> str:
        """Лемматизация текста с использованием pymorphy3"""
        words = text.split()
        lemmatized = []

        for word in words:
            parsed = self.morph.parse(word)[0]
            lemmatized.append(parsed.normal_form)

        return ' '.join(lemmatized)

    def preprocess_text(self, text: str) -> str:
        """Полная предобработка текста"""
        text = self.clean_text(text)
        text = self.lemmatize_text(text)
        return text

    def process_and_save(self):
        """Загрузка, обработка и сохранение данных"""
        try:
            # Загрузка данных
            df = self.load_dataset()
            logger.info("Обработка текстов...")

            # Лемматизация и очистка с progress bar
            tqdm.pandas(desc="Preprocessing texts")
            df['processed_text'] = df['text'].progress_apply(self.preprocess_text)

            # Сохраняем обработанные данные
            processed_path = self.data_dir / "processed_data.json"
            df[['title', 'text', 'processed_text']].to_json(
                processed_path, orient='records', lines=True, force_ascii=False
            )

            logger.info(f"Обработанные данные сохранены в {processed_path}")

        except Exception as e:
            logger.error(f"Ошибка обработки данных: {e}")
            raise


if __name__ == "__main__":
    # Пример использования
    loader = DataLoader(sample_size=1000)  # Загружаем 1000 документов
    loader.process_and_save()
