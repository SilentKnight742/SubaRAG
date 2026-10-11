"""Small immutable contracts shared by retrieval implementations and experiments."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Literal


@dataclass(frozen=True, slots=True)
class Document:
    id: str
    language: str
    text: str
    source: dict[str, Any] = field(default_factory=dict)


@dataclass(frozen=True, slots=True)
class EvaluationQuery:
    id: str
    language: str
    query: str
    relevant_passage_ids: frozenset[str]
    candidate_passage_ids: tuple[str, ...] = ()


@dataclass(frozen=True, slots=True)
class RetrievalHit:
    document: Document
    score: float
    rank: int
    channel: Literal["dense", "lexical", "hybrid", "reranked"]


@dataclass(frozen=True, slots=True)
class AdaptiveSelection:
    k: int
    boundary_gap: float | None
    boundary_ratio: float | None


@dataclass(frozen=True, slots=True)
class RouteDecision:
    run_lexical: bool
    reasons: tuple[str, ...]


@dataclass(frozen=True, slots=True)
class RetrievalTelemetry:
    route: str
    dense_ms: float
    lexical_ms: float
    fusion_ms: float
    selected_k: int
    candidate_k: int
    selection_policy: Literal["fixed", "adaptive"]
    lexical_triggered: bool
    lexical_reasons: tuple[str, ...]
    cache_hit: bool = False

    @property
    def adaptive_k(self) -> int:
        """Compatibility alias for reports produced before selection policy was explicit."""
        return self.selected_k


@dataclass(frozen=True, slots=True)
class RetrievalOutcome:
    hits: tuple[RetrievalHit, ...]
    telemetry: RetrievalTelemetry
