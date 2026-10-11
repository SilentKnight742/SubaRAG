"""Application-facing RAG boundary with server-owned grounding enforcement."""

from __future__ import annotations

import asyncio
from concurrent.futures import ThreadPoolExecutor
from collections.abc import Callable
from typing import Literal, Protocol

from ..retrieval import Retriever, extract_supporting_spans
from .generation import (
    AnswerGenerator,
    Evidence,
    GenerationRequest,
    UngroundedAnswerError,
)
from .models import RagQuery, RagResult
from .models import Citation


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


class GroundedRagEngine:
    """Retrieve, compress, generate, then resolve citations from trusted evidence."""

    status: Literal["ready"] = "ready"

    def __init__(
        self,
        *,
        retriever: Retriever,
        generator: AnswerGenerator,
        compressor: Callable = extract_supporting_spans,
        max_context_tokens: int = 500,
        retrieval_executor: ThreadPoolExecutor | None = None,
    ) -> None:
        self.retriever = retriever
        self.generator = generator
        self.compressor = compressor
        self.max_context_tokens = max_context_tokens
        self.retrieval_executor = retrieval_executor
        self._retrieval_future = None

    @property
    def busy(self) -> bool:
        # A timed-out await cannot stop native inference already in progress.
        return self._retrieval_future is not None and not self._retrieval_future.done()

    def close(self):
        if self.retrieval_executor is not None:
            self.retrieval_executor.shutdown(wait=True, cancel_futures=True)
        close = getattr(self.retriever, "close", None)
        if close is not None:
            close()

    async def answer(self, query: RagQuery) -> RagResult:
        if self.retrieval_executor is None:
            outcome = self.retriever.retrieve(query.question, query.language)
        else:
            # Keep HTTP/health checks responsive during CPU inference without
            # a second model process. Retriever locking covers cancelled awaits.
            if self.busy:
                raise RuntimeError("Retrieval is still running")
            self._retrieval_future = self.retrieval_executor.submit(
                self.retriever.retrieve, query.question, query.language,
            )
            outcome = await asyncio.wrap_future(self._retrieval_future)
        if not outcome.hits:
            return RagResult(
                status="no_answer",
                message="I couldn't find supporting evidence for that question.",
            )

        spans = self.compressor(
            query.question,
            outcome.hits,
            max_total_tokens=self.max_context_tokens,
        )
        if not spans:
            return RagResult(
                status="no_answer",
                message="I couldn't find supporting evidence for that question.",
            )

        evidence = tuple(
            Evidence(
                document_id=hit.document.id,
                text=text,
                rank=hit.rank,
                source=hit.document.source,
            )
            for hit, text in spans
        )
        generated = await self.generator.generate(GenerationRequest(
            question=query.question,
            language=query.language,
            evidence=evidence,
        ))
        if generated is None:
            return RagResult(
                status="no_answer",
                message="The retrieved sources do not support a reliable answer.",
            )

        evidence_by_id: dict[str, list[Evidence]] = {}
        for item in evidence:
            evidence_by_id.setdefault(item.document_id, []).append(item)
        cited_ids = tuple(dict.fromkeys(generated.cited_document_ids))
        if not cited_ids or any(identifier not in evidence_by_id for identifier in cited_ids):
            raise UngroundedAnswerError("Generator returned unsupported citations")

        citations = []
        for identifier in cited_ids:
            items = evidence_by_id[identifier]
            source_value = items[0].source.get("url") or items[0].source.get("file")
            citations.append(Citation(
                document_id=identifier,
                text=" ".join(item.text for item in items),
                source=str(source_value) if source_value is not None else None,
            ))
        return RagResult(
            status="answered",
            answer=generated.text,
            citations=citations,
        )
