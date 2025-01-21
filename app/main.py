from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from app.api.routes import router

# Создание экземпляра FastAPI приложения
app = FastAPI(
    title="Document Search Engine",
    description="Поисковый движок для русскоязычных документов",
    version="0.1.0"
)

# Настройка CORS - позволяет делать запросы с других доменов
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],      # Разрешаем запросы с любых доменов
    allow_credentials=True,   # Разрешаем передачу credentials
    allow_methods=["*"],      # Разрешаем все HTTP методы
    allow_headers=["*"],      # Разрешаем все заголовки
)

# Подключаем роутер с префиксом /api/v1
# Теперь все маршруты из routes.py будут доступны по /api/v1/...
app.include_router(router, prefix="/api/v1")

# Корневой маршрут для проверки работы API
@app.get("/")
async def root():
    return {"message": "Document Search Engine API"}

# Запуск приложения при непосредственном выполнении файла
if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)