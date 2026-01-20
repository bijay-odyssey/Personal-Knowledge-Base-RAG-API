from abc import ABC, abstractmethod
import faiss
from qdrant_client import QdrantClient
from typing import List, Dict, Any
import numpy as np
import uuid

class VectorDB(ABC):
    @abstractmethod
    def add(self, vectors: np.ndarray, metadata: List[Dict[str, Any]]):
        pass

    @abstractmethod
    def search(self, query_vector: np.ndarray, top_k: int = 5, filters: Dict[str, Any] = None) -> List[Dict[str, Any]]:
        pass

class FAISSVectorDB(VectorDB):
    def __init__(self, path: str, dim: int = 384):
        self.index = faiss.IndexFlatIP(dim)  # Cosine similarity
        self.metadata: List[Dict[str, Any]] = []

    def add(self, vectors: np.ndarray, metadata: List[Dict[str, Any]]):
        faiss.normalize_L2(vectors)  # normalize for cosine similarity
        self.index.add(vectors)
        self.metadata.extend(metadata)

    def search(self, query_vector: np.ndarray, top_k: int = 5, filters: Dict[str, Any] = None) -> List[Dict[str, Any]]:
        # normalize query
        query_vector = query_vector.reshape(1, -1).astype(np.float32)
        faiss.normalize_L2(query_vector)
        distances, indices = self.index.search(query_vector, top_k)

        results = []
        for i in indices[0]:
            if i < len(self.metadata):
                item = self.metadata[i].copy()
                item.setdefault("text", "")   # ensure key exists
                item.setdefault("source", "unknown")
                results.append(item)

        # Apply filter if provided
        if filters:
            results = [r for r in results if r.get("source") == filters.get("source")]

        return results

class QdrantVectorDB(VectorDB):
    def __init__(self, path: str):
        self.client = QdrantClient(path=path)
        self.client.recreate_collection(
            collection_name="pkb",
            vectors_config={"size": 384, "distance": "Cosine"}
        )
                
    def add(self, vectors: np.ndarray, metadata: List[Dict[str, Any]]):
        points = [
            {
                "id": str(uuid.uuid4()),
                "vector": vec.tolist(),
                "payload": {
                    "text": meta.get("text", ""),
                    "source": meta.get("source", "unknown")
                }
            }
            for vec, meta in zip(vectors, metadata)
        ]
        self.client.upsert(collection_name="pkb", points=points)

    def search(self, query_vector: np.ndarray, top_k: int = 5, filters: Dict[str, Any] = None) -> List[Dict[str, Any]]:
        filter_cond = None
        if filters:
            from qdrant_client.http.models import Filter, FieldCondition, MatchValue
            filter_cond = Filter(
                must=[FieldCondition(key="source", match=MatchValue(value=filters.get("source")))]
            )

        hits = self.client.search(
            collection_name="pkb",
            query_vector=query_vector.tolist(),
            limit=top_k,
            query_filter=filter_cond
        )

        results = []
        for hit in hits:
            text = hit.payload.get("text", "")
            source = hit.payload.get("source", "unknown")
            results.append({"text": text, "source": source})

        return results
