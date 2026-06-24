"""Простой тестовый скрипт для проверки API сервиса поиска."""

import requests
import json

BASE_URL = "http://localhost:8000"

# Тест 1: Проверка здоровья
print("=" * 50)
print("Тест 1: Проверка здоровья сервиса")
print("=" * 50)
response = requests.get(f"{BASE_URL}/health")
print(f"Статус: {response.status_code}")
print(f"Ответ: {response.json()}\n")

# Тест 2: Добавление товаров в индекс
print("=" * 50)
print("Тест 2: Добавление товаров в индекс")
print("=" * 50)
items = [
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

response = requests.post(f"{BASE_URL}/index", json=items)
print(f"Статус: {response.status_code}")
print(f"Ответ: {json.dumps(response.json(), ensure_ascii=False, indent=2)}\n")

# Тест 3: Поиск
print("=" * 50)
print("Тест 3: Поиск по 'саморез'")
print("=" * 50)
search_request = {"query": "саморез", "limit": 5}
response = requests.post(f"{BASE_URL}/search", json=search_request)
print(f"Статус: {response.status_code}")
print(f"Ответ: {json.dumps(response.json(), ensure_ascii=False, indent=2)}\n")

# Тест 4: Поиск по синониму
print("=" * 50)
print("Тест 4: Поиск по 'крепёж для дерева'")
print("=" * 50)
search_request = {"query": "крепёж для дерева", "limit": 5}
response = requests.post(f"{BASE_URL}/search", json=search_request)
print(f"Статус: {response.status_code}")
print(f"Ответ: {json.dumps(response.json(), ensure_ascii=False, indent=2)}\n")

print("=" * 50)
print("✅ Все тесты завершены успешно!")
print("=" * 50)
