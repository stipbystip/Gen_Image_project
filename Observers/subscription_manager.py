from PIL import Image
from Interfaces_Pipeline_Components.ImageObserver import ImageObserver


class SubscriptionManager:
    def __init__(self):
        self.__subscribers = []

    def subscribe(self, observer: ImageObserver):
        if observer not in self.__subscribers:
            self.__subscribers.append(observer)
            print(f"Подписан новый наблюдатель: {type(observer).__name__}")

    def unsubscribe(self, observer: ImageObserver):
        if observer in self.__subscribers:
            self.__subscribers.remove(observer)
            print(f"Отписан наблюдатель: {type(observer).__name__}")

    def notify_all(self, image: Image.Image):
        for observer in self.__subscribers:
            observer.handle_generated_image(image)
