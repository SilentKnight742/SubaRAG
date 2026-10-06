from __future__ import annotations

import asyncio

from subarag.experiments.generation import GenerationCase
from subarag.experiments.hosted_generation import evaluate_generator
from subarag.rag import GeneratedAnswer, GenerationProviderError


def cases() -> list[GenerationCase]:
    shared = {
        "language": "en",
        "question": "What is Goa's capital?",
        "reference_answers": ("Panaji",),
    }
    return [
        GenerationCase(
            id="answerable",
            evidence=(("D1", "source-1", "Panaji is Goa's capital."),),
            relevant_aliases=frozenset({"D1"}),
            expected_status="answered",
            **shared,
        ),
        GenerationCase(
            id="no-answer",
            evidence=(("D1", "source-2", "Margao is a city in Goa."),),
            relevant_aliases=frozenset(),
            expected_status="no_answer",
            **shared,
        ),
    ]


def test_hosted_gate_scores_typed_answers_and_refusals() -> None:
    class Generator:
        async def generate(self, request):
            if "capital" in request.evidence[0].text:
                return GeneratedAnswer("Panaji", ("D1",))
            return None

    report = asyncio.run(evaluate_generator(Generator(), cases()))

    assert report["summary"]["all"]["valid_json_rate"] == 1
    assert report["summary"]["all"]["grounded_citation_rate"] == 1
    assert report["summary"]["all"]["no_answer_accuracy"] == 1
    assert report["gate"]["passed"] is True


def test_hosted_gate_records_safe_provider_failure_as_failed_case() -> None:
    class Generator:
        async def generate(self, request):
            raise GenerationProviderError(
                "Safe message",
                code="rate_limited",
                retryable=True,
            )

    report = asyncio.run(evaluate_generator(Generator(), cases()[:1]))

    assert report["cases"][0]["error_code"] == "rate_limited"
    assert report["cases"][0]["valid_json"] is False
    assert report["gate"]["passed"] is False
