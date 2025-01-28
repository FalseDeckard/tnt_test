from fastapi import FastAPI
from fastapi.responses import HTMLResponse
from pathlib import Path
from app.api.endpoints import router, search_manager
import logging
from contextlib import asynccontextmanager
import uvicorn

"""
Главный модуль приложения FastAPI для поисковой системы.
Обеспечивает:
- Инициализацию и управление жизненным циклом приложения
- REST API для поиска документов
- Веб-интерфейс для тестирования
"""

# Настройка логирования
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Базовые пути
BASE_DIR = Path(__file__).resolve().parent  # Корневая директория приложения

# Чтение HTML шаблона для веб-интерфейса
try:
    with open(BASE_DIR / "templates" / "search.html", 'r', encoding='utf-8') as f:
        HTML_CONTENT = f.read()
except FileNotFoundError:
    HTML_CONTENT = """
    <html>
        <body>
            <h1>Ошибка: файл шаблона не найден</h1>
            <p>Проверьте наличие файла templates/search.html</p>
        </body>
    </html>
    """
    logger.error("HTML template file not found!")

@asynccontextmanager
async def lifespan(app: FastAPI):
    """
    Контекстный менеджер для управления жизненным циклом приложения.
    
    Выполняет:
    - Инициализацию поисковых систем при запуске
    - Очистку ресурсов при завершении
    
    Args:
        app (FastAPI): Экземпляр приложения FastAPI
    """
    logger.info("Запуск приложения...")
    try:
        # Инициализация поисковых систем
        await search_manager.initialize()
        logger.info("Поисковые системы успешно инициализированы")
        yield
    finally:
        # Завершение работы
        logger.info("Остановка приложения...")
        search_manager.cleanup()
        logger.info("Ресурсы успешно освобождены")

# Инициализация FastAPI приложения
app = FastAPI(
    title="Search API",
    description="""
    API для выполнения поиска по документам с использованием различных методов.
    
    ## Возможности
    
    * Поддержка трех методов поиска:
        * Гибридный (hybrid)
        * Векторный (vector)
        * Текстовый (text)
    * Множественные запросы в одном обращении
    * Настраиваемое количество результатов
    * Оценка релевантности для каждого документа
    * Измерение времени выполнения
    
    ## Использование
    
    1. Отправьте POST запрос на `/api/search` с параметрами:
        * `queries`: список запросов
        * `method`: метод поиска
        * `top_k`: количество результатов
        
    2. Получите результаты с оценками релевантности и временем выполнения
    
    ## Примеры
    
    ```python
    import requests
    
    response = requests.post(
        'http://localhost:8000/api/search',
        json={
            "queries": ["экономика россии"],
            "method": "hybrid",
            "top_k": 5
        }
    )
    results = response.json()
    ```
    
    ## Веб-интерфейс
    
    Доступен по адресу `/` для интерактивного тестирования API.
    """,
    version="0.0.1",
    lifespan=lifespan,  # Подключение управления жизненным циклом
    docs_url="/api/docs",  # URL для Swagger документации
    redoc_url="/api/redoc"  # URL для Redoc документации
)

# Подключение API роутов
app.include_router(router, prefix="/api")

# Маршрут для веб-интерфейса
@app.get("/", response_class=HTMLResponse)
async def get_search_page():
    """
    Возвращает HTML страницу поискового интерфейса.

    """
    return HTML_CONTENT

if __name__ == "__main__":
    """
    Точка входа для запуска приложения в production-режиме.
    
    Параметры запуска:
    - host="0.0.0.0": Доступ с любого IP
    - port=8000: Порт по умолчанию
    - reload=False: Отключение горячей перезагрузки
    """
    uvicorn.run("app.main:app", host="0.0.0.0", port=8000, reload=False)
