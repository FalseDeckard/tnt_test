from pydantic import BaseModel, Field, field_validator
from typing import List, Dict, Literal
from datetime import datetime

class SearchQuery(BaseModel):
    """
    Модель для поискового запроса.
    """
    queries: List[str] = Field(
        description="Список поисковых запросов для пакетного поиска",
        example=["нефть", "экономика россии"],
        min_items=1,
        max_items=20  # ограничиваем максимальное количество запросов
    )
    method: Literal["vector", "text", "hybrid"] = Field(
        default="hybrid",
        description="Метод поиска: векторный, текстовый или гибридный"
    )
    top_k: int = Field(
        default=5,
        ge=1,
        le=20,
        description="Количество возвращаемых результатов"
    )
    batch_size: int = Field(
        default=10,
        ge=1,
        le=20,
        description="Размер батча для обработки запросов"
    )
    weights: Dict[str, float] = Field(
        default={"bm25": 0.5, "vector": 0.5},
        description="Веса для гибридного поиска (bm25 и vector)"
    )

    @field_validator('weights')
    def validate_weights(cls, v):
        if set(v.keys()) != {"bm25", "vector"}:
            raise ValueError("Weights must contain exactly 'bm25' and 'vector' keys")
        if not all(0 <= weight <= 1 for weight in v.values()):
            raise ValueError("Weights must be between 0 and 1")
        total = sum(v.values())
        if not 0.99 <= total <= 1.01:  # допускаем небольшую погрешность
            raise ValueError("Weights must sum to 1.0")
        return v

    class Config:
        json_schema_extra = {
            "example": {
                "queries": ["экономика россии"],
                "method": "hybrid",
                "top_k": 5,
                "batch_size": 10,
                "weights": {"bm25": 0.5, "vector": 0.5}
            }
        }

class SearchResult(BaseModel):
    """
    Модель для отдельного результата поиска.
    """
    title: str = Field(description="Заголовок документа")
    summary: str = Field(description="Краткое содержание документа")
    url: str = Field(description="URL источника")
    date: str = Field(description="Дата публикации")
    score: float = Field(description="Оценка релевантности (0 до 1)")

class TimedSearchResult(BaseModel):
    """
    Модель для результатов поиска с временем выполнения.
    """
    results: List[SearchResult] = Field(description="Список найденных документов")
    query: str = Field(description="Исходный поисковый запрос")
    time_taken: float = Field(description="Время выполнения запроса в секундах")

class SearchResponse(BaseModel):
    """
    Модель ответа API.
    """
    results: List[TimedSearchResult] = Field(description="Результаты для каждого запроса")
    total_time: float = Field(description="Общее время выполнения всех запросов")
    method: str = Field(description="Использованный метод поиска")
    total_queries: int = Field(description="Общее количество обработанных запросов")
