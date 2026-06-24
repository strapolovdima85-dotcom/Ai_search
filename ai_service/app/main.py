import os
from fastapi import FastAPI, HTTPException, status
from pydantic import BaseModel, Field
from typing import List, Optional
from contextlib import asynccontextmanager

# Импорт разработанных нами ИИ-сервисов
from app.services.embedder import NomenclatureEmbedder
from app.services.search_engine import SearchEngine

# Глобальные переменные для синглтонов ИИ-модулей
embedder = None
search_engine = None

@asynccontextmanager
async def lifespan(app: FastAPI):
    """
    Управление жизненным циклом FastAPI (Lifespan).
    Срабатывает строго при запуске и остановке Docker-контейнера.
    """
    global embedder, search_engine
    print("[Система] Запуск ИИ-сервиса. Инициализация компонентов...")
    
    # 1. Загрузка нейросети RuBERT в память (CPU/GPU)
    embedder = NomenclatureEmbedder()
    
    # 2. Подключение к векторной базе данных Qdrant (с правильной размерностью вектора от модели)
    search_engine = SearchEngine(vector_size=embedder.vector_size)
    
    # 3. Автоматическое создание коллекции 'nomenclature', если её нет
    await search_engine.initialize_collection()
    
    print("[Система] Все ИИ-компоненты успешно запущены и готовы к работе.")
    yield
    print("[Система] Остановка ИИ-сервиса. Освобождение ресурсов...")

# Инициализация приложения с привязкой жизненного цикла
app = FastAPI(
    title="AI-Search Service for 1C:ERP",
    description="Микросервис семантического поиска номенклатуры организации (Диплом Архитектор ИИ)",
    version="1.0.0",
    lifespan=lifespan
)

# --- СХЕМЫ ДАННЫХ (Валидация контрактов с 1С через Pydantic) ---

class SearchRequest(BaseModel):
    query: str = Field(..., min_length=1, description="Текстовый запрос от менеджера 1С")
    limit: int = Field(default=10, ge=1, le=50, description="Лимит выдачи аналогов")

class ItemItem(BaseModel):
    id_1c: str = Field(..., description="GUID карточки из 1С:ERP")
    name: str = Field(..., description="Рабочее наименование")
    article: Optional[str] = Field(default="", description="Артикул")
    description: Optional[str] = Field(default="", description="Склеенные НаименованиеПолное + Описание из 1С")

class SearchResultItem(BaseModel):
    id_1c: str
    score: float

class TrainRequest(BaseModel):
    texts: List[str] = Field(..., description="Массив полных описаний номенклатуры для обучения ИИ")
    epochs: Optional[int] = Field(default=1, ge=1, le=5)


# --- ЭНДПОИНТЫ API ---
@app.post("/train", status_code=status.HTTP_200_OK, tags=["Обучение"])
async def trigger_training(request: TrainRequest):
    """
    Эндпоинт для запуска дообучения модели без остановки контейнера.
    """
    try:
        if not embedder:
            raise HTTPException(status_code=503, detail="Сервис еще не инициализирован")
        
        if len(request.texts) < 10:
            raise HTTPException(status_code=400, detail="Слишком мало данных для обучения (минимум 10 строк)")
            
        # Запуск фонового обучения (синхронный вызов, т.к. это тяжелая операция)
        saved_path = embedder.train_on_nomenclature(texts=request.texts, epochs=request.epochs)
        
        return {
            "status": "success",
            "message": f"Модель успешно обучена и обновлена в памяти. Веса сохранены в: {saved_path}"
        }
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Ошибка при дообучении: {str(e)}")

@app.get("/health", status_code=status.HTTP_200_OK, tags=["Системные"])
async def health_check():
    """Проверка доступности ИИ-сервиса."""
    return {"status": "healthy"}


@app.post("/search", response_model=List[SearchResultItem], status_code=status.HTTP_200_OK, tags=["Поиск"])
async def search_nomenclature(request: SearchRequest):
    """
    Real-time семантический поиск. 
    Вызывается при изменении строки поиска в форме подбора 1С:ERP.
    """
    try:
        if not embedder or not search_engine:
            raise HTTPException(status_code=503, detail="Сервис еще не инициализирован")
        
        # 1. Переводим текст запроса менеджера в математический вектор через RuBERT
        query_vector = embedder.get_embedding(request.query)
        
        # 2. Ищем ближайшие по смыслу векторы товаров в Qdrant
        results = await search_engine.search_semantic(
            query_embedding=query_vector, 
            limit=request.limit
        )
        return results

    except Exception as e:
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Ошибка выполнения ИИ-поиска: {str(e)}"
        )


@app.post("/index", status_code=status.HTTP_201_CREATED, tags=["Индексация"])
async def index_nomenclature(items: List[ItemItem]):
    """
    Пакетная индексация (батчинг).
    Вызывается при нажатии кнопки синхронизации в 1С для наполнения базы векторов.
    """
    try:
        if not embedder or not search_engine:
            raise HTTPException(status_code=503, detail="Сервис еще не инициализирован")
        
        if not items:
            raise HTTPException(status_code=status.HTTP_400_BAD_REQUEST, detail="Список пуст")

        # 1. Формируем тексты для векторизации (соединяем название и расширенное описание)
        texts_to_embed = []
        items_payload = []
        
        for item in items:
            # Создаем богатый текстовый контекст для нейросети
            full_context = f"{item.name} {item.article} {item.description}"
            texts_to_embed.append(full_context)
            
            # Сохраняем структуру для передачи метаданных в Qdrant
            items_payload.append(item.model_dump())
        
        # 2. Генерируем эмбеддинги пакетом (Batching) на CPU/GPU
        embeddings = embedder.get_embeddings_batch(texts_to_embed)
        
        # 3. Сохраняем векторы и GUID в базу данных Qdrant
        await search_engine.upsert_items(items=items_payload, embeddings=embeddings)
        
        return {
            "status": "success",
            "message": f"Успешно векторизовано и проиндексировано позиций: {len(items)}"
        }

    except Exception as e:
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Ошибка пакетной индексации: {str(e)}"
        )
