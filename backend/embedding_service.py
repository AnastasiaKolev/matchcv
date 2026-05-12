from sentence_transformers import SentenceTransformer
import numpy as np


class EmbeddingService:
    def __init__(self):
        # Лёгкая модель (420 МБ), работает на CPU
        self.model = SentenceTransformer(
            "paraphrase-multilingual-MiniLM-L12-v2",
            device="cpu"
        )

    def encode(self, texts: list[str]) -> np.ndarray:
        # MiniLM не нужны префиксы query/passage
        embeddings = self.model.encode(texts, normalize_embeddings=True)
        return embeddings