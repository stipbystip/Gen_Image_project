from abc import ABC, abstractmethod

class EmbeddingsModel(ABC):
    @abstractmethod
    def generate(self, processed_text):
        # Генерация эмбеддингов на основе обработанного текста
        raise NotImplemented