import os
import torch
import re
from typing import List
from sentence_transformers import SentenceTransformer, models, losses
from torch.utils.data import DataLoader
from sentence_transformers.datasets import DenoisingAutoEncoderDataset

class NomenclatureEmbedder:
    def __init__(self):
        # Поддержка динамического переключения модели через переменные окружения docker-compose
        self.base_model_name = os.getenv("EMBEDDING_MODEL_NAME", "intfloat/multilingual-e5-base")
        self.cache_folder = os.getenv("HF_HOME", "/app/models")
        
        # Защита от OOM и зависания контейнера на CPU
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        if self.device == "cpu":
            torch.set_num_threads(4) # Жестко ограничиваем ИИ, чтобы он не вешал докер
            
        # Выстраиваем имя папки для дообученной модели на основе базовой
        model_short_name = self.base_model_name.split('/')[-1]
        self.finetuned_path = os.path.join(self.cache_folder, f"{model_short_name}-finetuned")
        
        # Проверяем наличие дообученных весов
        if os.path.exists(self.finetuned_path):
            print(f"[ИИ-Инициализация] Обнаружена ДООБУЧЕННАЯ модель. Загрузка из {self.finetuned_path}...")
            self.model_path = self.finetuned_path
        else:
            print(f"[ИИ-Инициализация] Загрузка БАЗОВОЙ модели {self.base_model_name}...")
            self.model_path = self.base_model_name

        self.load_model()

    def load_model(self):
        """ Загружает или перегружает модель в память """
        self.model = SentenceTransformer(
            model_name_or_path=self.model_path,
            cache_folder=self.cache_folder,
            device=self.device
        )
        self.vector_size = self.model.get_sentence_embedding_dimension()

    def clean_text(self, text: str) -> str:
        if not text: return ""
        text = text.strip().lower()
        text = re.sub(r'(\d+)([а-яА-Яa-zA-Z])', r'\1 \2', text)
        text = re.sub(r'([а-яА-Яa-zA-Z])(\d+)', r'\1 \2', text)
        text = re.sub(r'[""\'«»®№°]', ' ', text)
        return " ".join(text.split())

    def get_embedding(self, text: str) -> List[float]:
        cleaned = self.clean_text(text)
        return self.model.encode(f"query: {cleaned}", convert_to_numpy=True).tolist()

    def get_embeddings_batch(self, texts: List[str], batch_size: int = 64) -> List[List[float]]:
        formatted_texts = [f"passage: {self.clean_text(t)}" for t in texts]
        return self.model.encode(formatted_texts, batch_size=batch_size, show_progress_bar=False, convert_to_numpy=True).tolist()

    def train_on_nomenclature(self, texts: List[str], epochs: int = 1) -> str:
        """ Метод запуска самообучения TSDAE прямо внутри работающего сервиса """
        print(f"[ИИ-Обучение] Старт дообучения на выборке из {len(texts)} строк...")
        
        # Очищаем тексты для обучения
        cleaned_texts = [self.clean_text(t) for t in texts if len(self.clean_text(t)) > 5]
        
        # ИСПРАВЛЕНО: Безопасная инициализация модулей без передачи несуществующих аргументов cache_dir
        # Используем глобальный параметр окружения HF_HOME, который sentence-transformers подхватит автоматически
        word_embedding_model = models.Transformer(self.base_model_name)
        pooling_model = models.Pooling(word_embedding_model.get_word_embedding_dimension(), pooling_mode='mean')
        
        train_model = SentenceTransformer(modules=[word_embedding_model, pooling_model], device=self.device)
        
        # Подготовка данных с автошумом
        dataset = DenoisingAutoEncoderDataset(cleaned_texts)
        dataloader = DataLoader(dataset, batch_size=16, shuffle=True)
        train_loss = losses.DenoisingAutoEncoderLoss(train_model)
        
        # Обучение
        train_model.fit(
            train_objectives=[(dataloader, train_loss)],
            epochs=epochs,
            weight_decay=0.01,
            optimizer_params={'lr': 3e-5}
        )
        
        # Сохраняем новую модель на диск
        train_model.save(self.finetuned_path)
        
        # Переключаем текущий инференс-инструмент на новые веса
        self.model_path = self.finetuned_path
        self.load_model()
        
        print(f"[ИИ-Обучение] Дообучение успешно завершено. Модель обновлена.")
        return self.finetuned_path
