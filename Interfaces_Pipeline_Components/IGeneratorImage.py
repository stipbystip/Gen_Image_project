from abc import ABC, abstractmethod

class ImageGenerator(ABC):
    @abstractmethod
    def generate(self, embedding_data):
        # Генерация изображения на основе эмбеддингов
        raise NotImplemented