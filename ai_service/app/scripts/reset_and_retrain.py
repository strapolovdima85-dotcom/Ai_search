import asyncio
from app.services.embedder import NomenclatureEmbedder
from app.services.search_engine import SearchEngine
from qdrant_client.http.models import Distance, VectorParams

SAMPLE_ITEMS = [
    {
        "id_1c": "123e4567-e89b-12d3-a456-426614174000",
        "name": "Саморез чёрный",
        "article": "SR-001",
        "description": "Саморез с потайной головкой 3.5x25 мм, чёрный, для крепления в дерево"
    },
    {
        "id_1c": "223e4567-e89b-12d3-a456-426614174001",
        "name": "Дюбель пластиковый",
        "article": "DB-002",
        "description": "Дюбель для бетона 6x30 мм, упаковка 50 штук"
    },
    {
        "id_1c": "323e4567-e89b-12d3-a456-426614174002",
        "name": "Болт оцинкованный",
        "article": "BLT-003",
        "description": "Болт м6 оцинкованный, полная резьба, класс прочности 8.8"
    }
]

async def main():
    print("[Reset] Инициализация embedder и search engine")
    embedder = NomenclatureEmbedder()
    search_engine = SearchEngine(vector_size=embedder.vector_size)

    try:
        await search_engine.client.delete_collection(collection_name=search_engine.collection_name)
        print(f"[Reset] Коллекция '{search_engine.collection_name}' удалена")
    except Exception as error:
        print(f"[Reset] Не удалось удалить коллекцию или коллекция не существовала: {error}")

    await search_engine.client.create_collection(
        collection_name=search_engine.collection_name,
        vectors_config=VectorParams(size=search_engine.vector_size, distance=Distance.COSINE)
    )
    print(f"[Reset] Коллекция '{search_engine.collection_name}' создана заново")

    texts = []
    for item in SAMPLE_ITEMS:
        full_context = f"{item['name']} {item['article']} {item['description']}"
        texts.append(full_context)

    embeddings = embedder.get_embeddings_batch(texts)
    await search_engine.upsert_items(items=SAMPLE_ITEMS, embeddings=embeddings)
    print(f"[Reset] Проиндексировано {len(SAMPLE_ITEMS)} тестовых позиций")

    print("[Reset] Запуск дообучения модели на тестовых данных")
    embedder.train_on_nomenclature(texts=texts, epochs=1)
    print("[Reset] Дообучение завершено")

if __name__ == '__main__':
    asyncio.run(main())
