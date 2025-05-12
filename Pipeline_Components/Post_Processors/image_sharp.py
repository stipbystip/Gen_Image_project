from Interfaces_Pipeline_Components.IPostProcessor import Postprocessor
from PIL import Image, ImageFilter

class ImageSharp(Postprocessor):
    def __init__(self, save_path='assets/images/res.png'):
        self.save_path=save_path
    def process(self, image):

        if not isinstance(image, Image.Image):
            image = Image.fromarray(image)


        sharpened_image = image.filter(ImageFilter.SHARPEN)

        if self.save_path:
            try:
                sharpened_image.save(self.save_path)
                print(f"Изображение сохранено по пути: {self.save_path}")
            except Exception as e:
                print(f"Ошибка при сохранении изображения: {e}")

        return sharpened_image