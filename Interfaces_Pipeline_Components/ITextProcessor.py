from abc import ABC, abstractmethod

class TextProcessor(ABC):
    @abstractmethod
    def process(self, text):
        # Обработка входного текста
        raise NotImplemented