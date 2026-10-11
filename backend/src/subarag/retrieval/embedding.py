"""Embedding boundary with an E5-aware Sentence Transformers implementation."""

from __future__ import annotations

from collections.abc import Sequence
from typing import Protocol


class Embedder(Protocol):
    name: str
    dimension: int

    def embed_documents(self, texts: Sequence[str]): ...

    def embed_queries(self, texts: Sequence[str]): ...


class SentenceTransformerEmbedder:
    """Load one model at a time; dependencies stay optional for the base API package."""

    def __init__(
        self, model_name: str, *, batch_size: int = 32, device: str = "cpu",
        revision: str | None = None,
    ) -> None:
        try:
            from sentence_transformers import SentenceTransformer
        except ImportError as exc:
            raise RuntimeError("Install the rag or experiments extra to load embeddings") from exc
        self.name = model_name
        self.batch_size = batch_size
        self._e5 = "e5" in model_name.lower()
        options = {"revision": revision} if revision is not None else {}
        self._model = SentenceTransformer(model_name, device=device, **options)
        dimension = (
            self._model.get_embedding_dimension()
            if hasattr(self._model, "get_embedding_dimension")
            else self._model.get_sentence_embedding_dimension()
        )
        if dimension is None:
            raise RuntimeError(f"Embedding dimension is unavailable for {model_name}")
        self.dimension = dimension

    def _encode(self, texts: Sequence[str], *, query: bool):
        prefix = "query: " if query else "passage: "
        values = [prefix + text if self._e5 else text for text in texts]
        return self._model.encode(
            values,
            batch_size=self.batch_size,
            convert_to_numpy=True,
            normalize_embeddings=True,
            show_progress_bar=len(values) >= self.batch_size * 4,
        )

    def embed_documents(self, texts: Sequence[str]):
        return self._encode(texts, query=False)

    def embed_queries(self, texts: Sequence[str]):
        return self._encode(texts, query=True)
