import numpy as np

from subarag.experiments.reranking import _rerank
from subarag.retrieval import Document, RetrievalHit


def test_reranker_reorders_candidates_without_changing_document_identity():
    hits = [
        RetrievalHit(Document("one", "en", "First"), 0.9, 1, "dense"),
        RetrievalHit(Document("two", "en", "Second"), 0.8, 2, "dense"),
    ]

    class CrossEncoder:
        def predict(self, pairs, *, batch_size, show_progress_bar):
            assert pairs == [("query", "First"), ("query", "Second")]
            assert batch_size == 2
            assert show_progress_bar is False
            return np.asarray([0.1, 0.9])

    reranked, elapsed = _rerank(CrossEncoder(), "query", hits, batch_size=2)
    assert [hit.document.id for hit in reranked] == ["two", "one"]
    assert [hit.rank for hit in reranked] == [1, 2]
    assert all(hit.channel == "reranked" for hit in reranked)
    assert elapsed >= 0
