"""Application-facing interface, without an embedding or generation implementation."""

from typing import Literal, Protocol

from .models import RagQuery, RagResult


class RagEngine(Protocol):
    @property
    def status(self) -> Literal["unavailable", "ready"]: ...

    async def answer(self, query: RagQuery) -> RagResult: ...


class UnavailableRagEngine:
    status: Literal["unavailable"] = "unavailable"

    async def answer(self, query: RagQuery) -> RagResult:
        return RagResult(
            status="unavailable",
            message="Your question is ready. Answers are coming soon.",
        )
