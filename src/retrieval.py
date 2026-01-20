from sentence_transformers.cross_encoder import CrossEncoder
from typing import List, Dict, Any
import numpy as np
from src.embedding import Embedder
from src.vector_db import VectorDB


class Retriever:
    def __init__(self, db: VectorDB, embedder: Embedder, reranker_model: str = None):
        self.db = db
        self.embedder = embedder
        self.reranker = CrossEncoder(reranker_model) if reranker_model else None

    def retrieve(
        self, 
        query: str, 
        top_k_initial: int = 10, 
        top_k_final: int = 3, 
        filters: Dict[str, Any] = None
    ) -> List[Dict[str, str]]:
        query_vec = self.embedder.embed([query])[0]
        candidates = self.db.search(query_vec, top_k=top_k_initial, filters=filters)
        
        valid_candidates = [c for c in candidates if isinstance(c, dict) and "text" in c]

        if self.reranker and valid_candidates:
            pairs = [[query, c["text"]] for c in valid_candidates]
            scores = self.reranker.predict(pairs)
            sorted_idx = np.argsort(scores)[::-1][:top_k_final]
            return [valid_candidates[i] for i in sorted_idx]

        return valid_candidates[:top_k_final]
