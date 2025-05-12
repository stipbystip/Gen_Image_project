class LORASelector:

    LORA_MODELS = {
        "anime": "assets/models/anime-nouveau-xl.safetensors",
        "shukezouma": "assets/models/MoXinV1.safetensors",
    }

    def __init__(self):
        self.keywords = {
            "anime": ["anime", "manga", "ghibli", "japanese"],
            "shukezouma": ['chinese painting', 'shukezouma', 'MoXin']
        }

    def select_lora(self, prompt):
        prompt_lower = prompt.lower()
        for style, words in self.keywords.items():
            if any(word in prompt_lower for word in words):
                return self.LORA_MODELS.get(style)
        return None