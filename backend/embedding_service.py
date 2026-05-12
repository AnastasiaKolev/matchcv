from sentence_transformers import SentenceTransformer
import torch
import numpy as np


class EmbeddingService:
    def __init__(self, model_name: str = "intfloat/multilingual-e5-large"):
        self.model = SentenceTransformer(model_name, device="cuda" if torch.cuda.is_available() else "cpu")

    def encode(self, texts: list[str]) -> np.ndarray:
        texts = [f"query: {t}" if len(t) < 500 else f"passage: {t}" for t in texts]
        embeddings = self.model.encode(texts, normalize_embeddings=True)
        return embeddings