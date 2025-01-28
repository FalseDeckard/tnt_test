from fastapi import FastAPI, Request
from fastapi.responses import HTMLResponse
from fastapi.staticfiles import StaticFiles
from pathlib import Path
from app.api.endpoints import router, search_manager
import logging
from contextlib import asynccontextmanager
import uvicorn

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Base directory and HTML content
BASE_DIR = Path(__file__).resolve().parent

# Read HTML content at module level
try:
    with open(BASE_DIR / "templates" / "search.html", 'r', encoding='utf-8') as f:
        HTML_CONTENT = f.read()
except FileNotFoundError:
    HTML_CONTENT = """
    <html>
        <body>
            <h1>Template file not found</h1>
        </body>
    </html>
    """

@asynccontextmanager
async def lifespan(app: FastAPI):
    """Startup and shutdown events"""
    logger.info("Starting up the application...")
    try:
        # Инициализируем поисковые движки при старте
        await search_manager.initialize()
        logger.info("Search engines initialized successfully")
        yield
    finally:
        logger.info("Shutting down the application...")
        search_manager.cleanup()
        logger.info("Cleanup completed")

# Создаем приложение
app = FastAPI(
    title="Search API",
    description="API for performing vector, text, and hybrid search",
    version="1.0.0",
    lifespan=lifespan  # Важно: добавляем lifespan
)

# Include API routes
app.include_router(router, prefix="/api")

# Add route for the web interface
@app.get("/", response_class=HTMLResponse)
async def get_search_page():
    return HTML_CONTENT

if __name__ == "__main__":
    uvicorn.run("app.main:app", host="0.0.0.0", port=8000, reload=False)