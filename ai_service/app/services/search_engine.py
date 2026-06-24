import os
from typing import List, Dict, Any
from qdrant_client import QdrantClient, AsyncQdrantClient
from qdrant_client.http import models
from qdrant_client.http.models import Distance, VectorParams, PointStruct

class SearchEngine:
    def __init__(self, vector_size: int = None):
        # Чтение параметров подключения из переменных окружения (задаются в docker-compose)
        self.host = os.getenv("QDRANT_HOST", "localhost")
        self.port = int(os.getenv("QDRANT_PORT", 6333))
        self.collection_name = "nomenclature"
        
        # Инициализация асинхронного клиента Qdrant для работы внутри FastAPI
        self.client = AsyncQdrantClient(host=self.host, port=self.port)
        
        # Размерность вектора: берем из параметра или используем значение по умолчанию
        # multilingual-e5-small: 384, multilingual-e5-base: 768
        self.vector_size = vector_size or 768

    async def initialize_collection(self) -> None:
        """
        Проверяет существование коллекции 'nomenclature' в Qdrant.
        Если коллекции нет, создает её с косинусным расстоянием для сравнения векторов.
        """
        collections_response = await self.client.get_collections()
        existing_collections = [c.name for c in collections_response.collections]
        
        if self.collection_name not in existing_collections:
            # Создание коллекции с HNSW-индексом (оптимально для быстрого ANN-поиска)
            await self.client.create_collection(
                collection_name=self.collection_name,
                vectors_config=VectorParams(
                    size=self.vector_size, 
                    distance=Distance.COSINE
                )
            )
            print(f"[Qdrant] Коллекция '{self.collection_name}' успешно создана.")

    async def upsert_items(self, items: List[Dict[str, Any]], embeddings: List[List[float]]) -> None:
        """
        Добавляет или обновляет (upsert) пакет номенклатуры в векторной БД.
        
        :param items: Список словарей с метаданными номенклатуры из 1С (id_1c, name, article, description)
        :param embeddings: Список соответствующих векторов, сгенерированных нейросетью
        """
        points = []
        for idx, item in enumerate(items):
            # Формируем структуру точки для Qdrant
            point = PointStruct(
                id=item["id_1c"], # В качестве ID точки используем GUID из 1С:ERP
                vector=embeddings[idx],
                payload={
                    "id_1c": item["id_1c"],
                    "name": item["name"],
                    "article": item.get("article", ""),
                    "description": item.get("description", "")
                }
            )
            points.append(point)
            
        # Асинхронная пакетная запись в БД
        await self.client.upsert(
            collection_name=self.collection_name,
            wait=True,
            points=points
        )

    async def search_semantic(self, query_embedding: List[float], limit: int = 10) -> List[Dict[str, Any]]:
        """
        Выполняет поиск ближайших соседей (Ближайших векторов товаров) по вектору поискового запроса.
        
        :param query_embedding: Вектор поискового запроса от менеджера
        :param limit: Максимальное количество возвращаемых результатов
        :return: Упорядоченный список словарей с id_1c (GUID) и score (степень схожести)
        """
        search_result = await self.client.search(
            collection_name=self.collection_name,
            query_vector=query_embedding,
            limit=limit,
            with_payload=True # Нужны payload, чтобы вернуть оригинальный id_1c из метаданных
        )
        
        # Форматируем ответ для API эндпоинта
        formatted_results = []
        for hit in search_result:
            payload_id = None
            if isinstance(hit.payload, dict):
                payload_id = hit.payload.get("id_1c")

            formatted_results.append({
                "id_1c": payload_id or str(hit.id),
                "score": float(hit.score)
            })
        
        return formatted_results
