from Builder.director import Director
from Observers.black_image_checker import BlackImageChecker
from Observers.nsfw_content_checker import NSFWContentChecker

class PipelineFacade:
    def __init__(self, pipeline_type="partial_pipeline"):
        self.pipeline_type = pipeline_type
        self.text_pipeline = Director("text").construct_pipeline()
        self.image_pipeline = Director("image").construct_pipeline()
        self.styler_pipeline = Director("styler").construct_pipeline() if pipeline_type == "full_pipeline" else None

        self.checkers = [
            BlackImageChecker(),
            NSFWContentChecker()
        ]
        for checker in self.checkers:
            self.image_pipeline.subscription_manager.subscribe(checker)

    def run(self, input_text, style='nothing'):
        text_embedding = self.text_pipeline.run(input_text, style=style)
        if self.pipeline_type == "full_pipeline":
            lora_path = self.styler_pipeline.run(input_text)
            result_image = self.image_pipeline.run(text_embedding, lora_path)
        else:
            result_image = self.image_pipeline.run(text_embedding)

        return result_image


if __name__ == '__main__':
    full_pipeline = PipelineFacade("full_pipeline")
    full_result = full_pipeline.run("Красивая девушка сидит на стуле и держит в руке телефон", style='anime')
    print(full_result)
