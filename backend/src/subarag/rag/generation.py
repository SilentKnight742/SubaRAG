"""Generator contract that cannot manufacture citation text or document metadata."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Protocol


@dataclass(frozen=True, slots=True)
class Evidence:
    document_id: str
    text: str
    rank: int
    source: dict[str, object]


@dataclass(frozen=True, slots=True)
class GenerationRequest:
    question: str
    language: str | None
    evidence: tuple[Evidence, ...]


@dataclass(frozen=True, slots=True)
class GeneratedAnswer:
    text: str
    cited_document_ids: tuple[str, ...]


class AnswerGenerator(Protocol):
    async def generate(self, request: GenerationRequest) -> GeneratedAnswer | None: ...


class UngroundedAnswerError(RuntimeError):
    """Raised when a generator cites evidence it was never given."""
