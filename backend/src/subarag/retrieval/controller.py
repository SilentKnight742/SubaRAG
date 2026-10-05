"""Measured production retrieval and isolated experimental routing policies."""

from __future__ import annotations

from dataclasses import replace
from time import perf_counter
from typing import Protocol

from .cache import BoundedCache, retrieval_cache_key
from .lexical import reciprocal_rank_fusion
from .models import RetrievalOutcome, RetrievalTelemetry
from .policies import route_lexical, select_adaptive_k


class Retriever(Protocol):
    def retrieve(self, query: str, language: str | None = None) -> RetrievalOutcome: ...


class FixedDenseRetriever:
    """Production default selected by the multilingual retrieval benchmark.

    The versioned cache stores retrieval outcomes, never generated answers.
    """

    def __init__(
        self,
        *,
        embedder,
        dense_index,
        corpus_version: str,
        retriever_version: str,
        k: int = 8,
        cache_entries: int = 1024,
    ) -> None:
        if k < 1:
            raise ValueError("k must be positive")
        self.embedder = embedder
        self.dense_index = dense_index
        self.corpus_version = corpus_version
        self.retriever_version = retriever_version
        self.k = k
        self.cache: BoundedCache[RetrievalOutcome] = BoundedCache(cache_entries)

    def retrieve(self, query: str, language: str | None = None) -> RetrievalOutcome:
        key = retrieval_cache_key(
            query,
            language,
            corpus_version=self.corpus_version,
            retriever_version=self.retriever_version,
        )
        cached = self.cache.get(key)
        if cached is not None:
            return RetrievalOutcome(
                hits=cached.hits,
                telemetry=replace(cached.telemetry, cache_hit=True),
            )

        started = perf_counter()
        embedding = self.embedder.embed_queries([query])[0]
        hits = tuple(self.dense_index.search(embedding, self.k))
        dense_ms = (perf_counter() - started) * 1000
        outcome = RetrievalOutcome(
            hits=hits,
            telemetry=RetrievalTelemetry(
                route="dense",
                dense_ms=dense_ms,
                lexical_ms=0.0,
                fusion_ms=0.0,
                selected_k=len(hits),
                candidate_k=self.k,
                selection_policy="fixed",
                lexical_triggered=False,
                lexical_reasons=(),
            ),
        )
        self.cache.put(key, outcome)
        return outcome


class AdaptiveRetriever:
    def __init__(
        self,
        *,
        embedder,
        dense_index,
        lexical_index=None,
        corpus_version: str,
        retriever_version: str,
        candidate_k: int = 32,
        max_k: int = 12,
        min_relative_gap: float = 0.08,
        lexical_min_top_score: float = 0.72,
        lexical_min_margin: float = 0.035,
        lexical_max_spread: float = 0.08,
        cache_entries: int = 1024,
    ) -> None:
        self.embedder = embedder
        self.dense_index = dense_index
        self.lexical_index = lexical_index
        self.corpus_version = corpus_version
        self.retriever_version = retriever_version
        self.candidate_k = candidate_k
        self.max_k = max_k
        self.min_relative_gap = min_relative_gap
        self.lexical_min_top_score = lexical_min_top_score
        self.lexical_min_margin = lexical_min_margin
        self.lexical_max_spread = lexical_max_spread
        self.cache: BoundedCache[RetrievalOutcome] = BoundedCache(cache_entries)

    def retrieve(self, query: str, language: str | None = None) -> RetrievalOutcome:
        key = retrieval_cache_key(
            query, language, corpus_version=self.corpus_version,
            retriever_version=self.retriever_version,
        )
        cached = self.cache.get(key)
        if cached is not None:
            telemetry = replace(cached.telemetry, cache_hit=True)
            return RetrievalOutcome(hits=cached.hits, telemetry=telemetry)

        started = perf_counter()
        embedding = self.embedder.embed_queries([query])[0]
        dense_hits = self.dense_index.search(embedding, self.candidate_k)
        dense_ms = (perf_counter() - started) * 1000

        decision = route_lexical(
            query, dense_hits, min_top_score=self.lexical_min_top_score,
            min_margin=self.lexical_min_margin, max_spread=self.lexical_max_spread,
        )
        lexical_ms = fusion_ms = 0.0
        combined = dense_hits
        if decision.run_lexical and self.lexical_index is not None:
            started = perf_counter()
            lexical_hits = self.lexical_index.search(query, self.candidate_k)
            lexical_ms = (perf_counter() - started) * 1000
            started = perf_counter()
            combined = reciprocal_rank_fusion(dense_hits, lexical_hits)[:self.candidate_k]
            fusion_ms = (perf_counter() - started) * 1000

        selection = select_adaptive_k(
            combined, max_k=self.max_k, min_relative_gap=self.min_relative_gap,
        )
        outcome = RetrievalOutcome(
            hits=tuple(combined[:selection.k]),
            telemetry=RetrievalTelemetry(
                route="conditional_hybrid" if decision.run_lexical and self.lexical_index else "dense",
                dense_ms=dense_ms, lexical_ms=lexical_ms, fusion_ms=fusion_ms,
                selected_k=selection.k, candidate_k=self.candidate_k,
                selection_policy="adaptive",
                lexical_triggered=decision.run_lexical and self.lexical_index is not None,
                lexical_reasons=decision.reasons,
            ),
        )
        self.cache.put(key, outcome)
        return outcome
