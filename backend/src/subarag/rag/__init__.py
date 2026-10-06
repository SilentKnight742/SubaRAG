"""Replaceable, citation-enforcing RAG boundary."""

from .engine import GroundedRagEngine, RagEngine, UnavailableRagEngine
from .generation import (
    AnswerGenerator,
    Evidence,
    GeneratedAnswer,
    GenerationProviderError,
    GenerationRequest,
    UngroundedAnswerError,
)
from .models import Citation, RagQuery, RagResult
from .runtime import build_grounded_rag_engine

__all__ = [
    "AnswerGenerator",
    "Citation",
    "Evidence",
    "GeneratedAnswer",
    "GenerationProviderError",
    "GenerationRequest",
    "GroundedRagEngine",
    "RagEngine",
    "RagQuery",
    "RagResult",
    "UnavailableRagEngine",
    "UngroundedAnswerError",
    "build_grounded_rag_engine",
]
