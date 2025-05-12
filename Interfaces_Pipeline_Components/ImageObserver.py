from abc import ABC, abstractmethod
from PIL import Image


class ImageObserver(ABC):
    @abstractmethod
    def handle_generated_image(self, image: Image.Image):
        pass
