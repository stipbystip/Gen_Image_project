import re
import spacy
from typing import List
from natasha import Segmenter, MorphVocab, NewsEmbedding, NewsMorphTagger, Doc
from deep_translator import GoogleTranslator
from Interfaces_Pipeline_Components.ITextProcessor import TextProcessor


class TextNormalizer(TextProcessor):
    def __init__(self):
        self.segmenter = Segmenter()
        self.morph_vocab = MorphVocab()
        self.emb = NewsEmbedding()
        self.morph_tagger = NewsMorphTagger(self.emb)
        self.translator = GoogleTranslator(source="ru", target="en")
        self.russian_letters = set("абвгдеёжзийклмнопрстуфхцчшщъыьэюя")
        self.EXCEPTIONS_WORDS = {"тебя", "меня", "его", "её", "нас", "вас", "их"}
        self.spacy_nlp = spacy.load("assets\\models\\en_core_web_sm\\en_core_web_sm-3.8.0")

    def get_tokens_ru(self, text: str) -> str:
        doc = Doc(text)
        doc.segment(self.segmenter)
        doc.tag_morph(self.morph_tagger)
        for token in doc.tokens:
            token.lemmatize(self.morph_vocab)

        tokens = [
            token.text if token.text in self.EXCEPTIONS_WORDS else token.lemma
            for token in doc.tokens
        ]
        translated = self.translator.translate(" ".join(tokens))
        return translated.lower()

    def get_tokens_en(self, text: str) -> str:
        doc = self.spacy_nlp(text)
        return " ".join(
            token.text for token in doc
            if token.pos_ not in ["DET", "ADP", "AUX", "PART", "PRON"]
        )

    def process(self, text: str) -> str:
        if not text.strip():
            raise ValueError("На вход пришел пустой текст.")

        text = re.sub(r"[^\w\s+-]", "", text.lower())
        ru_letter_count = sum(1 for char in text if char in self.russian_letters)
        total_letter_count = sum(1 for char in text if char.isalpha())

        if total_letter_count == 0:
            raise ValueError("В тексте нет букв.")

        if ru_letter_count / total_letter_count >= 0.7:
            return self.get_tokens_ru(text)
        else:
            return self.get_tokens_en(text)
