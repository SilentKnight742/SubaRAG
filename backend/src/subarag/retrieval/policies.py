"""Cheap, deterministic policies used by the adaptive retrieval controller."""

from __future__ import annotations

import re
from collections.abc import Sequence

from .models import AdaptiveSelection, RetrievalHit, RouteDecision


def select_adaptive_k(
    hits: Sequence[RetrievalHit],
    *,
    min_k: int = 1,
    max_k: int = 12,
    min_relative_gap: float = 0.08,
    buffer: int = 0,
) -> AdaptiveSelection:
    """Choose the strongest score boundary, falling back to max_k when none is meaningful."""
    if min_k < 1 or max_k < min_k:
        raise ValueError("Invalid adaptive-k bounds")
    limit = min(max_k, len(hits))
    if limit <= min_k or not hits:
        return AdaptiveSelection(k=limit, boundary_gap=None, boundary_ratio=None)
    candidates: list[tuple[float, float, int]] = []
    for index in range(min_k - 1, limit - 1):
        gap = hits[index].score - hits[index + 1].score
        scale = max(abs(hits[index].score), 1e-9)
        ratio = gap / scale
        candidates.append((ratio, gap, index + 1))
    ratio, gap, boundary = max(candidates)
    if ratio < min_relative_gap:
        return AdaptiveSelection(k=limit, boundary_gap=gap, boundary_ratio=ratio)
    return AdaptiveSelection(
        k=min(limit, max(min_k, boundary + buffer)), boundary_gap=gap, boundary_ratio=ratio,
    )


_LEXICAL_MARKERS = re.compile(r"\d|[\"']|\b[A-Z]{2,}\b")


def route_lexical(
    query: str,
    dense_hits: Sequence[RetrievalHit],
    *,
    min_top_score: float = 0.72,
    min_margin: float = 0.035,
    max_spread: float = 0.08,
) -> RouteDecision:
    reasons: list[str] = []
    if _LEXICAL_MARKERS.search(query):
        reasons.append("lexical_marker")
    if not dense_hits:
        reasons.append("no_dense_results")
    else:
        if dense_hits[0].score < min_top_score:
            reasons.append("weak_top_score")
        if len(dense_hits) > 1 and dense_hits[0].score - dense_hits[1].score < min_margin:
            reasons.append("small_top_margin")
        window = dense_hits[: min(8, len(dense_hits))]
        if len(window) > 1 and window[0].score - window[-1].score < max_spread:
            reasons.append("flat_score_distribution")
    return RouteDecision(run_lexical=bool(reasons), reasons=tuple(reasons))
