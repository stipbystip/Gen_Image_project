class TextPipeline:
    def __init__(self):
        self.text_processor = None
        self.promt_enchancer = None
        self.embeddings = None

    def run(self, text, style='anime'):
        processed_text = self.text_processor.process(text)
        enchancer_text = self.promt_enchancer.enhance(processed_text, style)
        embedding_data = self.embeddings.generate(enchancer_text)
        return embedding_data