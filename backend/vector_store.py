"""Tiny in-memory numpy vector store (one per visitor)."""
import numpy as np


class VectorStore:
    def __init__(self):
        self.vectors = None
        self.chunks: list[dict] = []

    def add(self, chunks: list[dict], vectors: np.ndarray) -> None:
        vectors = vectors.astype("float32")
        self.vectors = vectors if self.vectors is None else np.vstack([self.vectors, vectors])
        self.chunks.extend(chunks)

    def remove_source(self, source: str) -> None:
        keep = [i for i, c in enumerate(self.chunks) if c["source"] != source]
        self.chunks = [self.chunks[i] for i in keep]
        self.vectors = self.vectors[keep] if keep else None

    def search(self, query_vec: np.ndarray, k: int = 5) -> list[dict]:
        if self.vectors is None or not self.chunks:
            return []
        scores = self.vectors @ query_vec
        top = np.argsort(-scores)[:k]
        return [{**self.chunks[i], "rank": r, "score": float(scores[i])} for r, i in enumerate(top, start=1)]
