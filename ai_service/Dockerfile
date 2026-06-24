# Использовать официальный стабильный образ Python
FROM python:3.11-slim

# Установка системных зависимостей для сборки некоторых Python-пакетов
RUN apt-get update && apt-get install -y --no-install-recommends \
    build-essential \
    curl \
    && rm -rf /var/lib/apt/lists/*

# Установка рабочей директории внутри контейнера
WORKDIR /app

# Копирование файла зависимостей
COPY requirements.txt .

# Установка библиотек ИИ (без кэша, чтобы уменьшить размер образа)
RUN pip install --no-cache-dir --upgrade pip && \
    pip install --no-cache-dir -r requirements.txt

# Установка NLTK data напрямую, минуя downloader, чтобы избежать проблем с импортом NLTK на этапе сборки
RUN mkdir -p /usr/local/share/nltk_data/corpora /usr/local/share/nltk_data/tokenizers && \
    python - <<'PY'
import os, urllib.request
base = 'https://raw.githubusercontent.com/nltk/nltk_data/gh-pages/packages'
files = {
    'wordnet': 'corpora/wordnet.zip',
    'punkt': 'tokenizers/punkt.zip',
    'omw-1.4': 'corpora/omw-1.4.zip'
}
output = '/usr/local/share/nltk_data'
for name, path in files.items():
    dest = os.path.join(output, path)
    print('Downloading', name, dest)
    data = urllib.request.urlopen(f"{base}/{path}").read()
    with open(dest, 'wb') as f:
        f.write(data)
PY

ENV NLTK_DATA=/usr/local/share/nltk_data

# Копирование всего исходного кода приложения
COPY ./app ./app

# Создание папки под локальное хранение весов моделей
RUN mkdir -p /app/models

# Переменная окружения для загрузки моделей Hugging Face в локальную папку контейнера
ENV HF_HOME=/app/models

# Открытие порта для FastAPI
EXPOSE 8000

# Команда для запуска асинхронного сервера uvicorn
CMD ["uvicorn", "app.main:app", "--host", "0.0.0.0", "--port", "8000"]
