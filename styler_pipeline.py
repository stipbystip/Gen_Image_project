class StylerPipeline:
    def __init__(self):
        self.styler = None

    def run(self, text):
        lora_path = self.styler.select_lora(text)
        return lora_path