from abc import ABC
from Interfaces_Pipeline_Components.ImageObserver import ImageObserver


class BaseImageChecker(ImageObserver, ABC):
    def __init__(self):
        self.last_check_result = None
