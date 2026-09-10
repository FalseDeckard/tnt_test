import logging
from contextlib import asynccontextmanager
from pathlib import Path

import uvicorn
from fastapi import FastAPI
from fastapi.responses import HTMLResponse
from starlette.concurrency import run_in_threadpool

from app.api.endpoints import SearchManager, router

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

def create_app(search_manager: SearchManager | None = None) -> FastAPI:
    """Build an application with an injectable search manager."""
    manager = search_manager or SearchManager()

    @asynccontextmanager
    async def lifespan(application: FastAPI):
        application.state.search_manager = manager
        logger.info("Запуск приложения...")
        try:
            await run_in_threadpool(manager.initialize)
            logger.info("Поисковые системы успешно инициализированы")
            yield
        finally:
            logger.info("Остановка приложения...")
            await run_in_threadpool(manager.cleanup)
            logger.info("Ресурсы успешно освобождены")

    application = FastAPI(
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
        version="0.1.0",
        lifespan=lifespan,
        docs_url="/api/docs",
        redoc_url="/api/redoc",
    )

    application.include_router(router, prefix="/api")

    @application.get("/", response_class=HTMLResponse)
    async def get_search_page():
        return HTML_CONTENT

    return application


app = create_app()

if __name__ == "__main__":
    """
    Точка входа для запуска приложения в production-режиме.
    
    Параметры запуска:
    - host="0.0.0.0": Доступ с любого IP
    - port=8000: Порт по умолчанию
    - reload=False: Отключение горячей перезагрузки
    """
    uvicorn.run("app.main:app", host="0.0.0.0", port=8000, reload=False)
