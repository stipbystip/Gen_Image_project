from abc import ABC, abstractmethod

class IPipelineBuilder(ABC):
    @abstractmethod
    def build_text_processor(self):
        raise NotImplemented

    @abstractmethod
    def build_embeddings(self):
        raise NotImplemented

    @abstractmethod
    def build_image_generator(self):
        raise NotImplemented

    @abstractmethod
    def build_postprocessor(self):
        raise NotImplemented

    @abstractmethod
    def get_pipeline(self):
        raise NotImplemented