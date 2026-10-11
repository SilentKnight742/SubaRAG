"""Unicode-safe BM25S retrieval and rank fusion."""

from __future__ import annotations

import unicodedata
from collections.abc import Sequence

from .models import Document, RetrievalHit


def unicode_tokens(text: str) -> list[str]:
    """Keep combining marks attached so Devanagari words are not truncated."""
    tokens: list[str] = []
    current: list[str] = []
    for character in unicodedata.normalize("NFC", text).casefold():
        category = unicodedata.category(character)
        if category[0] in {"L", "N"} or category in {"Mc", "Me", "Mn"}:
            current.append(character)
        elif current:
            tokens.append("".join(current))
            current.clear()
    if current:
        tokens.append("".join(current))
    return tokens


class Bm25SparseIndex:
    name = "bm25s"

    def __init__(self, documents: Sequence[Document]) -> None:
        try:
            import bm25s
        except ImportError as exc:
            raise RuntimeError("Install the rag or experiments extra to use BM25S") from exc
        if not documents:
            raise ValueError("Cannot index an empty corpus")
        self.documents = tuple(documents)
        self._retriever = bm25s.BM25()
        self._retriever.index([unicode_tokens(document.text) for document in documents], show_progress=False)

    def search(self, query: str, k: int) -> list[RetrievalHit]:
        if k < 1:
            raise ValueError("k must be positive")
        results = self._retriever.retrieve(
            [unicode_tokens(query)], k=min(k, len(self.documents)), show_progress=False,
        )
        return [RetrievalHit(self.documents[int(index)], float(score), rank, "lexical")
                for rank, (index, score) in enumerate(
                    zip(results.documents[0], results.scores[0], strict=True), 1
                )]


def reciprocal_rank_fusion(
    dense: Sequence[RetrievalHit], lexical: Sequence[RetrievalHit], *, constant: int = 60,
) -> list[RetrievalHit]:
    documents: dict[str, Document] = {}
    scores: dict[str, float] = {}
    for result in (dense, lexical):
        for hit in result:
            documents[hit.document.id] = hit.document
            scores[hit.document.id] = scores.get(hit.document.id, 0.0) + 1.0 / (constant + hit.rank)
    ordered = sorted(scores, key=scores.__getitem__, reverse=True)
    return [RetrievalHit(documents[identifier], scores[identifier], rank, "hybrid")
            for rank, identifier in enumerate(ordered, 1)]
