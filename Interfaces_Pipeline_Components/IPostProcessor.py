from abc import ABC, abstractmethod

class Postprocessor(ABC):
    @abstractmethod
    def process(self, image):
        # Постобработка сгенерированного изображения
        raise NotImplemented