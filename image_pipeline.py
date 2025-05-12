from Observers.subscription_manager import SubscriptionManager

class ImagePipeline:
    def __init__(self):
        self.image_generator = None
        self.postprocessor = None
        self.subscription_manager = SubscriptionManager()

    def run(self, embedding_data, lora_path=None):
        image = self.image_generator.generate(embedding_data, lora_path)
        self.subscription_manager.notify_all(image)
        final_image = self.postprocessor.process(image)
        return final_image
