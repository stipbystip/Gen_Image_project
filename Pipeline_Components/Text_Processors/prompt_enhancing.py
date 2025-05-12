import torch
from Interfaces_Pipeline_Components.IPromptEnchancer import IPromptEnchancer
from transformers import GenerationConfig, GPT2LMHeadModel, GPT2Tokenizer, LogitsProcessor, LogitsProcessorList



class CustomLogitsProcessor(LogitsProcessor):
    def __init__(self, bias):
        super().__init__()
        self.bias = bias

    def __call__(self, input_ids, scores):
        if len(input_ids.shape) == 2:
            last_token_id = input_ids[0, -1]
            self.bias[last_token_id] = -1e10
        return scores + self.bias



class PromptEnhancing(IPromptEnchancer):
    def __init__(self):
        self.tokenizer = GPT2Tokenizer.from_pretrained("Gustavosta/MagicPrompt-Stable-Diffusion")
        self.model = GPT2LMHeadModel.from_pretrained("Gustavosta/MagicPrompt-Stable-Diffusion",
                                           torch_dtype=torch.float16).to("cuda")
        self.model.eval()
        self.word_pairs = ["highly detailed", "high quality", "enhanced quality", "perfect composition",
                           "dynamic light"]
        self.styles = {
            "shukezouma": "chinese traditional painting of {prompt}, highly detailed, cinemascope, gorgeous",
            "anime": "anime artwork of {prompt}, anime style, key visual, vibrant, studio anime, highly detailed",
        }

        self.words = [
            "aesthetic", "astonishing", "beautiful", "breathtaking", "composition", "contrasted", "epic", "moody",
            "enhanced",
            "exceptional", "fascinating", "flawless", "glamorous", "glorious", "illumination", "impressive", "improved",
            "inspirational", "magnificent", "majestic", "hyperrealistic", "smooth", "sharp", "focus", "stunning",
            "detailed",
            "intricate", "dramatic", "high", "quality", "perfect", "light", "ultra", "highly", "radiant", "satisfying",
            "soothing", "sophisticated", "stylish", "sublime", "terrific", "touching", "timeless", "wonderful",
            "unbelievable",
            "elegant", "awesome", "amazing", "dynamic", "trendy",
        ]
        self.processor_list = self.create_processor_list()

    def create_processor_list(self):
        word_ids = [self.tokenizer.encode(word, add_prefix_space=True)[0] for word in self.words]
        bias = torch.full((self.tokenizer.vocab_size,), -float("Inf")).to("cuda")
        bias[word_ids] = 0
        processor = CustomLogitsProcessor(bias)
        return LogitsProcessorList([processor])

    def find_and_order_pairs(self, s, pairs):
        words = s.split()
        found_pairs = []
        for pair in pairs:
            pair_words = pair.split()
            if pair_words[0] in words and pair_words[1] in words:
                found_pairs.append(pair)
                words.remove(pair_words[0])
                words.remove(pair_words[1])

        for word in words[:]:
            for pair in pairs:
                if word in pair.split():
                    words.remove(word)
                    break
        ordered_pairs = ", ".join(found_pairs)
        remaining_s = ", ".join(words)
        return ordered_pairs, remaining_s

    def enhance(self, prompt, style='cinematic'):
        inputs = self.tokenizer(prompt, return_tensors='pt').to('cuda')
        token_count = inputs["input_ids"].shape[1]
        max_new_tokens = 20 - token_count

        gen_config = GenerationConfig(
            penalty_alpha=0.7,
            eos_token_id=self.model.config.eos_token_id,
            pad_token_id=self.model.config.eos_token_id,
            top_k=15,
            do_sample=True,
        )
        with torch.no_grad():
            gen_ids = self.model.generate(
                input_ids=inputs["input_ids"],
                attention_mask=inputs["attention_mask"],
                max_new_tokens=max_new_tokens,
                generation_config=gen_config,
                logits_processor=self.processor_list,
            )
        output_tokens = [self.tokenizer.decode(generated_id, skip_special_tokens=True) for generated_id in gen_ids]
        input_part, generated_part = output_tokens[0][: len(prompt)], output_tokens[0][len(prompt):]
        pairs, words = self.find_and_order_pairs(generated_part, self.word_pairs)
        formatted_generated_part = ", ".join(filter(None, [pairs, words]))

        if style in self.styles:
            style = self.styles[style].format(prompt=input_part)
            enhanced_prompt = style + (", " + formatted_generated_part if formatted_generated_part else "")
        else:
            enhanced_prompt = input_part + (
                ", " + formatted_generated_part if formatted_generated_part else "")
        return enhanced_prompt


