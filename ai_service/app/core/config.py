"""
Конфигурация приложения FastAPI для ИИ-сервиса поиска номенклатуры.

Все параметры загружаются из переменных окружения (задаются в docker-compose.yml).
"""

import os
from typing import Optional


class Settings:
    """Глобальные настройки приложения."""
    
    # === FastAPI ===
    APP_NAME: str = "AI-Search Service for 1C:ERP"
    APP_VERSION: str = "1.0.0"
    
    # === Qdrant (Векторная БД) ===
    QDRANT_HOST: str = os.getenv("QDRANT_HOST", "localhost")
    QDRANT_PORT: int = int(os.getenv("QDRANT_PORT", 6333))
    QDRANT_COLLECTION_NAME: str = "nomenclature"
    QDRANT_VECTOR_SIZE: int = 768  # Размер векторов из sentence-transformers
    
    # === Embedding Model (Нейросетевая модель для преобразования текста в вектор) ===
    EMBEDDING_MODEL_NAME: str = os.getenv(
        "EMBEDDING_MODEL_NAME", 
        "intfloat/multilingual-e5-base"
    )
    HF_CACHE_FOLDER: str = os.getenv("HF_HOME", "/app/models")
    
    # === Пути к файлам ===
    LOG_LEVEL: str = os.getenv("LOG_LEVEL", "INFO")
    
    # === Поиск параметры ===
    SEARCH_DEFAULT_LIMIT: int = 10
    SEARCH_MAX_LIMIT: int = 50
    SEARCH_MIN_QUERY_LENGTH: int = 1
    
    # === Обучение модели параметры ===
    TRAIN_MIN_TEXTS: int = 10
    TRAIN_MAX_EPOCHS: int = 5
    TRAIN_BATCH_SIZE: int = 16
    TRAIN_LEARNING_RATE: float = 3e-5
    
    # === Индексация параметры ===
    INDEX_BATCH_SIZE: int = 100  # Размер батча при индексации (батчинг)
    
    @classmethod
    def get_model_display_name(cls) -> str:
        """Вернуть понятное имя модели для логирования."""
        model_short = cls.EMBEDDING_MODEL_NAME.split('/')[-1]
        return f"{model_short} ({cls.EMBEDDING_MODEL_NAME})"


# Глобальный объект конфигурации
settings = Settings()
