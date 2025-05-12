from Builder.pipeline_builder import PipelineBuilder
from image_pipeline import ImagePipeline
from styler_pipeline import StylerPipeline
from text_pipeline import TextPipeline


class Director:
    def __init__(self, type_builder='image'):
        self.builder = PipelineBuilder(type_builder)

    def construct_pipeline(self):
        if isinstance(self.builder.pipeline, TextPipeline):
            self.builder.build_text_processor()
            self.builder.build_prompt_enchancer()
            self.builder.build_embeddings()

        elif isinstance(self.builder.pipeline, StylerPipeline):
            self.builder.build_styler()

        elif isinstance(self.builder.pipeline, ImagePipeline):
            self.builder.build_image_generator()
            self.builder.build_postprocessor()

        return self.builder.get_pipeline()