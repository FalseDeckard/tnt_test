from app.data.loader import DataLoader
from pathlib import Path
import json


def test_loader():
    # Создаем экземпляр DataLoader
    loader = DataLoader(sample_size=10, data_dir="data/processed_test")

    # Запускаем процесс загрузки, обработки и сохранения данных
    loader.process_and_save()

    # Проверяем, что файлы созданы
    processed_file = Path("data/processed_test/processed_data.json")

    assert processed_file.exists(), "Файл с обработанными данными не был создан!"

    # Загружаем файл для проверки
    with open(processed_file, "r", encoding="utf-8") as f:
        data = [json.loads(line) for line in f]

    # Проверяем формат данных
    assert len(data) == 10, f"Ожидалось 10 документов, но получено {len(data)}"
    assert "processed_text" in data[0], "Поле 'processed_text' отсутствует в обработанных данных!"

    print("Тест пройден успешно!")


if __name__ == "__main__":
    test_loader()

