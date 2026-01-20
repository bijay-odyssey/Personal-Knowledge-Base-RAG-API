from sentence_transformers import SentenceTransformer
from sklearn.metrics.pairwise import cosine_similarity
from typing import List
import numpy as np

class Embedder:
    def __init__(self, model_name: str):
        self.model = SentenceTransformer(model_name)

    def embed(self, texts: List[str]) -> np.ndarray:
        return self.model.encode(texts)

    def compute_similarity(self, vec1: np.ndarray, vec2: np.ndarray) -> float:
        return cosine_similarity([vec1], [vec2])[0][0]
    
     