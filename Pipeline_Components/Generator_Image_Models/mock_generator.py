from PIL import Image

class MockImageGenerator:
    def generate(self, *args, **kwargs) -> Image.Image:
        image = Image.new("RGB", (512, 512), color="white")
        return image