from __future__ import annotations

import asyncio

import pytest

from subarag.rag import (
    GeneratedAnswer,
    GroundedRagEngine,
    RagQuery,
    UngroundedAnswerError,
)
from subarag.retrieval import Document, RetrievalHit, RetrievalOutcome
from subarag.retrieval.models import RetrievalTelemetry


def outcome(*hits: RetrievalHit) -> RetrievalOutcome:
    return RetrievalOutcome(
        hits=hits,
        telemetry=RetrievalTelemetry(
            route="dense",
            dense_ms=1,
            lexical_ms=0,
            fusion_ms=0,
            selected_k=len(hits),
            candidate_k=8,
            selection_policy="fixed",
            lexical_triggered=False,
            lexical_reasons=(),
        ),
    )


class FixtureRetriever:
    def __init__(self, result: RetrievalOutcome) -> None:
        self.result = result

    def retrieve(self, query: str, language: str | None = None) -> RetrievalOutcome:
        return self.result


def test_grounded_engine_owns_citation_text_and_metadata():
    document = Document(
        "goa",
        "en",
        "Panaji is the capital of Goa. It is beside the Mandovi.",
        {"url": "https://example.test/goa"},
    )
    seen = []

    class Generator:
        async def generate(self, request):
            seen.append(request)
            return GeneratedAnswer("Panaji is Goa's capital.", ("goa",))

    engine = GroundedRagEngine(
        retriever=FixtureRetriever(outcome(RetrievalHit(document, 0.9, 1, "dense"))),
        generator=Generator(),
    )
    result = asyncio.run(engine.answer(RagQuery(question="What is Goa's capital?", language="en")))

    assert result.status == "answered"
    assert result.citations[0].document_id == "goa"
    assert result.citations[0].text == (
        "Panaji is the capital of Goa. It is beside the Mandovi."
    )
    assert result.citations[0].source == "https://example.test/goa"
    assert seen[0].evidence[0].document_id == "goa"


def test_grounded_engine_rejects_generator_citations_outside_evidence():
    document = Document("goa", "en", "Panaji is the capital of Goa.")

    class Generator:
        async def generate(self, request):
            return GeneratedAnswer("Unsupported", ("invented",))

    engine = GroundedRagEngine(
        retriever=FixtureRetriever(outcome(RetrievalHit(document, 0.9, 1, "dense"))),
        generator=Generator(),
    )
    with pytest.raises(UngroundedAnswerError):
        asyncio.run(engine.answer(RagQuery(question="Capital?")))


def test_grounded_engine_returns_no_answer_without_evidence_or_generator_call():
    class Generator:
        called = False

        async def generate(self, request):
            self.called = True
            return GeneratedAnswer("Should not happen", ("x",))

    generator = Generator()
    engine = GroundedRagEngine(
        retriever=FixtureRetriever(outcome()),
        generator=generator,
    )
    result = asyncio.run(engine.answer(RagQuery(question="Unknown?")))
    assert result.status == "no_answer"
    assert result.answer is None
    assert result.citations == []
    assert generator.called is False
