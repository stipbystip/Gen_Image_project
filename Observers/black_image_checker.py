from numpy import bool_

from Observers.base_image_checker import BaseImageChecker
from PIL import Image
import numpy as np


class BlackImageChecker(BaseImageChecker):
    def handle_generated_image(self, image: Image.Image):
        self.last_check_result = self._is_black_image(image)
        if self.last_check_result:
            print(f"[BlackImageChecker] Предупреждение: Изображение полностью черное")

    def _is_black_image(self, image: Image.Image) -> bool_:
        arr = np.array(image)
        return np.all(arr == [0, 0, 0])
