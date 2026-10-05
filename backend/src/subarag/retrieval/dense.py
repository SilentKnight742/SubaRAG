"""Exact and HNSW dense indexes with identical result contracts."""

from __future__ import annotations

from collections.abc import Sequence
from pathlib import Path

from .models import Document, RetrievalHit


def _numpy():
    try:
        import numpy as np
    except ImportError as exc:
        raise RuntimeError("Install the rag or experiments extra to use dense retrieval") from exc
    return np


def normalized_float32(vectors):
    np = _numpy()
    values = np.ascontiguousarray(vectors, dtype=np.float32)
    if values.ndim != 2 or not len(values):
        raise ValueError("Expected a non-empty two-dimensional embedding matrix")
    norms = np.linalg.norm(values, axis=1, keepdims=True)
    if np.any(norms == 0):
        raise ValueError("Zero-length embeddings cannot be indexed")
    return np.ascontiguousarray(values / norms, dtype=np.float32)


class ExactDenseIndex:
    name = "exact"

    def __init__(self, documents: Sequence[Document], embeddings) -> None:
        if len(documents) != len(embeddings):
            raise ValueError("Document and embedding counts differ")
        self.documents = tuple(documents)
        self.embeddings = normalized_float32(embeddings)

    @property
    def size_bytes(self) -> int:
        return self.embeddings.nbytes

    def search(self, query_embedding, k: int) -> list[RetrievalHit]:
        np = _numpy()
        if k < 1:
            raise ValueError("k must be positive")
        query = normalized_float32(np.asarray(query_embedding).reshape(1, -1))[0]
        scores = self.embeddings @ query
        limit = min(k, len(self.documents))
        if limit == len(self.documents):
            indices = np.argsort(-scores)
        else:
            candidates = np.argpartition(-scores, limit - 1)[:limit]
            indices = candidates[np.argsort(-scores[candidates])]
        return [RetrievalHit(self.documents[int(index)], float(scores[index]), rank, "dense")
                for rank, index in enumerate(indices[:limit], 1)]


class HnswDenseIndex:
    name = "hnsw"

    def __init__(
        self,
        documents: Sequence[Document],
        embeddings,
        *,
        m: int = 32,
        ef_construction: int = 200,
        ef_search: int = 64,
    ) -> None:
        try:
            import faiss
        except ImportError as exc:
            raise RuntimeError("Install the rag or experiments extra to use HNSW") from exc
        if len(documents) != len(embeddings):
            raise ValueError("Document and embedding counts differ")
        self.documents = tuple(documents)
        self.embeddings = normalized_float32(embeddings)
        self.index = faiss.IndexHNSWFlat(self.embeddings.shape[1], m, faiss.METRIC_INNER_PRODUCT)
        self.index.hnsw.efConstruction = ef_construction
        self.index.hnsw.efSearch = ef_search
        self.index.add(self.embeddings)

    @classmethod
    def load(
        cls,
        documents: Sequence[Document],
        path: str | Path,
        *,
        ef_search: int = 64,
    ) -> "HnswDenseIndex":
        try:
            import faiss
        except ImportError as exc:
            raise RuntimeError("Install the rag or experiments extra to use HNSW") from exc
        instance = cls.__new__(cls)
        instance.documents = tuple(documents)
        instance.embeddings = None
        instance.index = faiss.read_index(str(path))
        if instance.index.ntotal != len(instance.documents):
            raise ValueError("Index and document counts differ")
        if not hasattr(instance.index, "hnsw"):
            raise ValueError("The saved FAISS index is not HNSW")
        instance.index.hnsw.efSearch = ef_search
        return instance

    def save(self, path: str | Path) -> None:
        try:
            import faiss
        except ImportError as exc:  # pragma: no cover - construction already requires it
            raise RuntimeError("Install the rag or experiments extra to save HNSW") from exc
        faiss.write_index(self.index, str(path))

    @property
    def size_bytes(self) -> int:
        try:
            import faiss
        except ImportError:  # pragma: no cover - construction already requires it
            return 0
        return len(faiss.serialize_index(self.index))

    def search(self, query_embedding, k: int) -> list[RetrievalHit]:
        np = _numpy()
        if k < 1:
            raise ValueError("k must be positive")
        query = normalized_float32(np.asarray(query_embedding).reshape(1, -1))
        limit = min(k, len(self.documents))
        scores, indices = self.index.search(query, limit)
        return [RetrievalHit(self.documents[int(index)], float(score), rank, "dense")
                for rank, (index, score) in enumerate(zip(indices[0], scores[0], strict=True), 1)
                if index >= 0]
