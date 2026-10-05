"""Deterministic retrieval metrics and latency summaries for experiment reports."""

from __future__ import annotations

from math import log2
from statistics import mean

from .models import EvaluationQuery, RetrievalHit


def ranking_metrics(
    queries: list[EvaluationQuery], rankings: dict[str, list[RetrievalHit]], *, k: int = 10,
) -> dict[str, float]:
    if not queries:
        raise ValueError("At least one evaluable query is required")
    recalls: list[float] = []
    reciprocal_ranks: list[float] = []
    ndcgs: list[float] = []
    for query in queries:
        ids = [hit.document.id for hit in rankings[query.id][:k]]
        relevant = query.relevant_passage_ids
        found = [index + 1 for index, identifier in enumerate(ids) if identifier in relevant]
        recalls.append(len(set(ids) & relevant) / len(relevant))
        reciprocal_ranks.append(1.0 / min(found) if found else 0.0)
        dcg = sum(1.0 / log2(rank + 1) for rank in found)
        ideal = sum(1.0 / log2(rank + 1) for rank in range(1, min(len(relevant), k) + 1))
        ndcgs.append(dcg / ideal if ideal else 0.0)
    return {
        f"recall@{k}": mean(recalls),
        f"mrr@{k}": mean(reciprocal_ranks),
        f"ndcg@{k}": mean(ndcgs),
    }


def latency_summary(values: list[float]) -> dict[str, float]:
    if not values:
        return {"p50": 0.0, "p95": 0.0, "p99": 0.0, "mean": 0.0}
    ordered = sorted(values)

    def percentile(percent: float) -> float:
        position = (len(ordered) - 1) * percent
        lower = int(position)
        upper = min(lower + 1, len(ordered) - 1)
        fraction = position - lower
        return ordered[lower] * (1 - fraction) + ordered[upper] * fraction

    return {
        "p50": percentile(0.50),
        "p95": percentile(0.95),
        "p99": percentile(0.99),
        "mean": mean(ordered),
    }
