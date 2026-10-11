"""Run a small real-generation check against an already verified compact bundle.

This makes seven Gemini requests. It never transcribes audio, uploads artifacts,
or writes credentials to its report.
"""

from __future__ import annotations

import argparse
import asyncio
import json
import os
from pathlib import Path
from tempfile import mkdtemp
from time import perf_counter

import httpx


LANGUAGES = ("en", "hi", "mr")
UNSUPPORTED = {
    "id": "unsupported-fictional-star",
    "language": "en",
    "query": "What is the official spectral classification of the fictional star QZX-9471?",
    "relevant": [],
    "expected": "no_answer",
}


def read(path: Path):
    return json.loads(Path(path).read_text())


def write(path: Path, value) -> None:
    path.write_text(json.dumps(value, ensure_ascii=False, indent=2) + "\n")


def select_supported_cases(root: Path):
    fresh = read(root / "fresh-cases.json")
    rankings = read(root / "compact.json")["rankings"]
    selected = []
    for language in LANGUAGES:
        candidates = []
        for case in fresh:
            if case["language"] != language:
                continue
            ranking = rankings[language + ":" + case["id"]]
            relevant = set(case["relevant"])
            if not relevant.intersection(ranking):
                continue
            candidates.append((ranking[0] not in relevant, case["id"], case, ranking))
        candidates.sort(key=lambda item: (item[0], item[1]))
        if len(candidates) < 2:
            raise ValueError(f"Not enough fresh retrievable cases for {language}")
        for _top_miss, _identifier, case, ranking in candidates[:2]:
            selected.append({**case, "expected": "answered", "saved_ranking": ranking})
    return selected


async def run(serving_report: Path, output: Path) -> None:
    from subarag.api.app import create_app
    from subarag.api.settings import Settings

    api_key = os.environ.pop("GEMINI_API_KEY", None)
    if not api_key:
        raise ValueError("Set GEMINI_API_KEY only in this process environment")
    serving = read(serving_report)
    root = serving_report.parent
    bundle_manifest = Path(serving["bundle_manifest"])
    if not bundle_manifest.is_file():
        raise ValueError("The verified compact bundle is missing")
    cases = select_supported_cases(root) + [UNSUPPORTED]
    application = create_app(settings=Settings(
        _env_file=None,
        elevenlabs_api_key="unused-in-answer-check",
        gemini_api_key=api_key,
        compact_manifest=bundle_manifest,
        query_timeout_seconds=60,
    ))
    api_key = None
    results = []
    async with application.router.lifespan_context(application):
        async with httpx.AsyncClient(
            transport=httpx.ASGITransport(app=application), base_url="http://test",
        ) as client:
            for case in cases:
                started = perf_counter()
                response = await client.post("/api/queries", json={
                    "question": case["query"], "language": case["language"],
                })
                elapsed = (perf_counter() - started) * 1000
                try:
                    payload = response.json()
                except json.JSONDecodeError:
                    payload = {"invalid_response": response.text[:500]}
                if not isinstance(payload, dict):
                    payload = {"invalid_response": payload}
                citations = payload.get("citations", [])
                cited = [item.get("document_id") for item in citations if isinstance(item, dict)]
                saved = case.get("saved_ranking", [])
                relevant = set(case["relevant"])
                automatic = (
                    response.status_code == 200
                    and payload.get("status") == case["expected"]
                    and (case["expected"] == "no_answer" or (
                        bool(cited) and set(cited).issubset(saved)
                    ))
                )
                results.append({
                    "id": case["id"], "language": case["language"], "question": case["query"],
                    "expected": case["expected"], "http_status": response.status_code,
                    "response": payload, "latency_ms": elapsed,
                    "cited_ids": cited, "saved_top8": saved,
                    "cites_labeled_relevant": bool(relevant.intersection(cited)),
                    "automatic_contract_passed": automatic,
                })
                print(case["language"], case["id"], response.status_code,
                      payload.get("status"), f"{elapsed:.0f} ms", flush=True)
    report = {
        "schema_version": 1,
        "artifact_kind": "subarag_compact_answer_check",
        "serving_report": str(serving_report),
        "bundle_manifest_sha256": serving["bundle_manifest_sha256"],
        "model": "gemini-3.5-flash-lite",
        "requests": len(results),
        "automatic_contract_passed": all(item["automatic_contract_passed"] for item in results),
        "manual_review_required": [
            "Answer is correct and supported by the displayed citation text.",
            "Answer language matches the question language.",
            "No-answer behavior is appropriate for the fictional-star question.",
        ],
        "results": results,
        "limitations": "Six same-corpus labeled questions and one synthetic unsupported question; not a broad answer benchmark.",
    }
    output.mkdir(parents=True, exist_ok=True)
    report_path = output / "compact-answer-report.json"
    if report_path.exists():
        raise ValueError("Use a fresh answer-check output directory")
    write(report_path, report)
    print("REPORT:", report_path, flush=True)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--serving-report", type=Path, required=True)
    parser.add_argument("--output", type=Path)
    args = parser.parse_args()
    destination = args.output or Path(mkdtemp(prefix="subarag-answer-check-", dir="/content"))
    asyncio.run(run(args.serving_report.resolve(), destination.resolve()))
