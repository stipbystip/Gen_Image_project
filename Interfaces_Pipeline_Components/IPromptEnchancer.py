from abc import ABC, abstractmethod

class IPromptEnchancer(ABC):
    @abstractmethod
    def enhance(self, text, style=''):
        # Постобработка сгенерированного изображения
        raise NotImplemented