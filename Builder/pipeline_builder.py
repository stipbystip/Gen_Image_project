from Builder.Ipipeline_Builder import IPipelineBuilder
from Pipeline_Components.Post_Processors.image_sharp import ImageSharp
from Pipeline_Components.Embeddings_Models.clip_model import ClipModel
from Pipeline_Components.Text_Processors.text_normalize import TextNormalizer
from Pipeline_Components.Text_Processors.prompt_enhancing import PromptEnhancing
from Pipeline_Components.Stylers.LORA import LORASelector
from Pipeline_Components.Generator_Image_Models.stable_diffusion import StableDiffusion
from image_pipeline import ImagePipeline
from styler_pipeline import StylerPipeline
from text_pipeline import TextPipeline


class PipelineBuilder(IPipelineBuilder):
    def __init__(self, type_builder='image'):
        if type_builder == 'image':
            self.pipeline = ImagePipeline()
        elif type_builder == 'styler':
            self.pipeline = StylerPipeline()
        elif type_builder == 'text':
            self.pipeline = TextPipeline()

    def build_text_processor(self):
        self.pipeline.text_processor = TextNormalizer()

    def build_prompt_enchancer(self):
        self.pipeline.promt_enchancer = PromptEnhancing()

    def build_embeddings(self):
        self.pipeline.embeddings = ClipModel()

    def build_styler(self):
        self.pipeline.styler = LORASelector()

    def build_image_generator(self):
        self.pipeline.image_generator = StableDiffusion()

    def build_postprocessor(self):
        self.pipeline.postprocessor = ImageSharp()

    def get_pipeline(self):
        return self.pipeline
