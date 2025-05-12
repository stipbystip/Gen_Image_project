from Interfaces_Pipeline_Components.IEmbeddingsModel import EmbeddingsModel
from transformers import CLIPTextModel, CLIPTokenizer
import torch


class ClipModel(EmbeddingsModel):
    def __init__(self):
        super().__init__()
        self.model_name = 'openai/clip-vit-large-patch14'
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.tokenizer = CLIPTokenizer.from_pretrained(self.model_name)
        self.model = CLIPTextModel.from_pretrained(self.model_name).to(self.device)

    def generate(self, processed_text):
        inputs = self.tokenizer(
            processed_text,
            return_tensors="pt",
            padding=True,
            truncation=True,
            max_length=77
        ).to(self.device)

        max_length = inputs.input_ids.shape[-1]

        unconditional_input = self.tokenizer(
            [""],
            padding="max_length",
            max_length=max_length,
            return_tensors="pt"
        ).to(self.device)

        with torch.no_grad():
            unconditional_embeddings = self.model(**unconditional_input).last_hidden_state
            text_features = self.model(**inputs).last_hidden_state

        embeddings = torch.cat([unconditional_embeddings, text_features])
        return embeddings.cpu().numpy()
