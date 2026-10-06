"""Run the frozen grounded-generation gate against a hosted generator."""

from __future__ import annotations

import argparse
import asyncio
import json
import platform
from datetime import datetime, timezone
from pathlib import Path
from time import perf_counter
from typing import Any

from ..api.settings import Settings
from ..data.manifest import load_manifest
from ..rag import Evidence, GeneratedAnswer, GenerationProviderError, GenerationRequest
from ..rag.providers import GeminiAnswerGenerator, create_generation_http_client
from ..retrieval.metrics import latency_summary
from .generation import GenerationCase, build_cases, score_response


FORMAT_GATE = 0.95
GROUNDING_GATE = 0.80
NO_ANSWER_GATE = 0.80
LATENCY_P95_GATE_MS = 3_000


def _mean(values: list[float]) -> float:
    return sum(values) / len(values) if values else 0.0


def _as_scored_json(answer: GeneratedAnswer | None) -> str:
    if answer is None:
        value = {"status": "no_answer", "answer": None, "citations": []}
    else:
        value = {
            "status": "answered",
            "answer": answer.text,
            "citations": list(answer.cited_document_ids),
        }
    return json.dumps(value, ensure_ascii=False)


async def evaluate_generator(generator, cases: list[GenerationCase]) -> dict[str, Any]:
    """Evaluate a generator using document aliases so grounding stays measurable."""
    results: list[dict[str, Any]] = []
    for case_number, case in enumerate(cases, 1):
        request = GenerationRequest(
            question=case.question,
            language=case.language,
            evidence=tuple(
                Evidence(
                    document_id=alias,
                    text=text,
                    rank=index,
                    source={"dataset_document_id": document_id},
                )
                for index, (alias, document_id, text) in enumerate(case.evidence, 1)
            ),
        )
        started = perf_counter()
        error_code = None
        try:
            answer = await generator.generate(request)
            raw = _as_scored_json(answer)
        except GenerationProviderError as error:
            raw = ""
            error_code = error.code
        total_ms = (perf_counter() - started) * 1_000
        score = score_response(case, raw)
        results.append({
            "case_id": case.id,
            "language": case.language,
            "expected_status": case.expected_status,
            "total_ms": total_ms,
            "error_code": error_code,
            **score,
        })
        print(
            f"gemini: {case_number}/{len(cases)} {case.id} "
            f"score={score['score']:.3f} total_ms={total_ms:.1f}",
            flush=True,
        )

    summary: dict[str, Any] = {}
    active_languages = tuple(dict.fromkeys(item["language"] for item in results))
    for language in (*active_languages, "all"):
        selected = results if language == "all" else [
            item for item in results if item["language"] == language
        ]
        answerable = [item for item in selected if item["expected_status"] == "answered"]
        no_answer = [item for item in selected if item["expected_status"] == "no_answer"]
        summary[language] = {
            "cases": len(selected),
            "mean_score": _mean([item["score"] for item in selected]),
            "valid_json_rate": _mean([float(item["valid_json"]) for item in selected]),
            "status_accuracy": _mean([float(item["status_correct"]) for item in selected]),
            "grounded_citation_rate": _mean([
                float(item["citations_valid"] and item["relevant_citation"])
                for item in answerable
            ]),
            "answer_token_f1": _mean([item["answer_token_f1"] for item in answerable]),
            "no_answer_accuracy": _mean([
                float(item["status_correct"]) for item in no_answer
            ]),
            "total_ms": latency_summary([item["total_ms"] for item in selected]),
        }

    aggregate = summary["all"]
    checks = {
        "valid_json": aggregate["valid_json_rate"] >= FORMAT_GATE,
        "grounded_citations": aggregate["grounded_citation_rate"] >= GROUNDING_GATE,
        "no_answer": aggregate["no_answer_accuracy"] >= NO_ANSWER_GATE,
        "latency": aggregate["total_ms"]["p95"] <= LATENCY_P95_GATE_MS,
    }
    return {
        "summary": summary,
        "gate": {
            "passed": all(checks.values()),
            "checks": checks,
            "thresholds": {
                "valid_json_rate": FORMAT_GATE,
                "grounded_citation_rate": GROUNDING_GATE,
                "no_answer_accuracy": NO_ANSWER_GATE,
                "total_ms_p95": LATENCY_P95_GATE_MS,
            },
        },
        "cases": results,
    }


async def run(args: argparse.Namespace) -> dict[str, Any]:
    settings = Settings()
    if settings.gemini_api_key is None:
        raise RuntimeError("GEMINI_API_KEY must be configured in backend/.env.")
    manifest = load_manifest(args.manifest, verify_files=True)
    cases = build_cases(
        args.manifest,
        languages=tuple(args.languages),
        cases_per_language=args.cases_per_language,
        max_documents=args.documents,
    )
    async with create_generation_http_client() as client:
        generator = GeminiAnswerGenerator(
            api_key=settings.gemini_api_key.get_secret_value(),
            client=client,
            model=args.model or settings.gemini_model,
        )
        evaluation = await evaluate_generator(generator, cases)
    return {
        "schema_version": 1,
        "created_at": datetime.now(timezone.utc).isoformat(),
        "environment": {
            "python": platform.python_version(),
            "platform": platform.platform(),
        },
        "artifact": {
            "manifest": str(args.manifest.resolve()),
            "source_revision": manifest.source.revision,
            "documents_per_language": args.documents,
            "languages": args.languages,
            "cases": len(cases),
        },
        "configuration": {
            "provider": "gemini",
            "model": args.model or settings.gemini_model,
            "cases_per_language": args.cases_per_language,
        },
        **evaluation,
        "limitations": [
            "This gate isolates generation using labeled evidence; it does not include retrieval errors.",
            "Token F1 is a deterministic proxy and does not replace human multilingual review.",
            "The non-streaming endpoint measures total response latency, not time to first token.",
            "Hosted latency and rate limits can vary by region, load, and account tier.",
        ],
    }


def parser() -> argparse.ArgumentParser:
    value = argparse.ArgumentParser(description="Run the hosted Gemini generation gate")
    value.add_argument("--manifest", type=Path, required=True)
    value.add_argument("--output", type=Path, required=True)
    value.add_argument("--model")
    value.add_argument("--languages", nargs="+", choices=("en", "hi", "mr"), default=["en", "hi", "mr"])
    value.add_argument("--documents", type=int, default=10_000)
    value.add_argument("--cases-per-language", type=int, default=3)
    return value


def main() -> None:
    args = parser().parse_args()
    report = asyncio.run(run(args))
    args.output.mkdir(parents=True, exist_ok=True)
    destination = args.output / "hosted-generation-bakeoff.json"
    temporary = destination.with_suffix(".json.tmp")
    temporary.write_text(json.dumps(report, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    temporary.replace(destination)
    print(destination.resolve())


if __name__ == "__main__":
    main()
