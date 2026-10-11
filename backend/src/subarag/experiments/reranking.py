"""Measure always-on and uncertainty-gated cross-encoder reranking."""

from __future__ import annotations

import argparse
import json
from datetime import datetime, timezone
from pathlib import Path
from time import perf_counter

from ..retrieval import HnswDenseIndex, RetrievalHit, load_language_artifact
from ..retrieval.metrics import latency_summary, ranking_metrics
from .retrieval import _artifact_fingerprint, _calibration_split, _model_slug


def _evaluate(queries, rankings):
    metrics = ranking_metrics(queries, rankings, k=10)
    metrics["query_count"] = len(queries)
    return metrics


def _rerank(cross_encoder, query, hits, *, batch_size: int):
    pairs = [(query, hit.document.text) for hit in hits]
    started = perf_counter()
    scores = cross_encoder.predict(pairs, batch_size=batch_size, show_progress_bar=False)
    elapsed = (perf_counter() - started) * 1000
    ordered = sorted(zip(hits, scores, strict=True), key=lambda item: float(item[1]), reverse=True)
    reranked = [
        RetrievalHit(hit.document, float(score), rank, "reranked")
        for rank, (hit, score) in enumerate(ordered, 1)
    ]
    return reranked, elapsed


def run(args) -> dict:
    try:
        import numpy as np
        from sentence_transformers import CrossEncoder
    except ImportError as exc:
        raise RuntimeError("Install the experiments extra to run reranking") from exc

    documents, queries = load_language_artifact(
        args.manifest,
        args.language,
        max_documents=args.documents,
        max_queries=args.queries,
    )
    fingerprint = _artifact_fingerprint(documents, queries)
    embeddings_path = (
        args.embedding_cache
        / "embeddings"
        / args.language
        / _model_slug(args.embedder)
        / f"{len(documents)}-{fingerprint[:16]}.npz"
    )
    if not embeddings_path.is_file():
        raise ValueError(f"Run the retrieval benchmark first; cache is missing: {embeddings_path}")
    cached = np.load(embeddings_path)
    document_embeddings = cached["documents"]
    query_embeddings = cached["queries"]
    index = HnswDenseIndex(documents, document_embeddings)
    dense = {
        query.id: index.search(embedding, args.candidate_k)
        for query, embedding in zip(queries, query_embeddings, strict=True)
    }

    started = perf_counter()
    cross_encoder = CrossEncoder(args.reranker, device="cpu")
    load_ms = (perf_counter() - started) * 1000
    if queries:
        _rerank(cross_encoder, queries[0].query, dense[queries[0].id], batch_size=args.batch_size)
    reranked = {}
    timings = []
    for query in queries:
        reranked[query.id], elapsed = _rerank(
            cross_encoder,
            query.query,
            dense[query.id],
            batch_size=args.batch_size,
        )
        timings.append(elapsed)

    calibration, evaluation = _calibration_split(queries)
    margins = {
        query.id: dense[query.id][0].score - dense[query.id][1].score
        for query in queries
    }
    ordered_margins = sorted(margins[query.id] for query in calibration)
    thresholds = [ordered_margins[0] - 1e-6]
    thresholds.extend(
        ordered_margins[round((len(ordered_margins) - 1) * fraction)]
        for fraction in (0.25, 0.5, 0.75)
    )
    thresholds.append(ordered_margins[-1] + 1e-6)
    target = max(
        _evaluate(calibration, dense)["ndcg@10"],
        _evaluate(calibration, reranked)["ndcg@10"],
    )
    trials = []
    for threshold in thresholds:
        conditional = {
            query.id: reranked[query.id] if margins[query.id] <= threshold else dense[query.id]
            for query in calibration
        }
        metrics = _evaluate(calibration, conditional)
        trigger_rate = sum(margins[query.id] <= threshold for query in calibration) / len(calibration)
        trials.append({"max_margin": threshold, "trigger_rate": trigger_rate, **metrics})
    eligible = [trial for trial in trials if trial["ndcg@10"] >= target * 0.99]
    selected = min(eligible or trials, key=lambda trial: (trial["trigger_rate"], -trial["ndcg@10"]))
    conditional = {
        query.id: reranked[query.id]
        if margins[query.id] <= selected["max_margin"]
        else dense[query.id]
        for query in evaluation
    }
    evaluation_trigger_rate = (
        sum(margins[query.id] <= selected["max_margin"] for query in evaluation)
        / len(evaluation)
    )
    dense_evaluation = _evaluate(evaluation, dense)
    reranked_evaluation = _evaluate(evaluation, reranked)
    conditional_evaluation = _evaluate(evaluation, conditional)
    evaluation_target = max(dense_evaluation["ndcg@10"], reranked_evaluation["ndcg@10"])
    return {
        "schema_version": 1,
        "created_at": datetime.now(timezone.utc).isoformat(),
        "artifact": {
            "manifest": str(args.manifest.resolve()),
            "language": args.language,
            "documents": len(documents),
            "queries": len(queries),
            "fingerprint": fingerprint,
        },
        "configuration": {
            "embedder": args.embedder,
            "reranker": args.reranker,
            "candidate_k": args.candidate_k,
            "batch_size": args.batch_size,
        },
        "reranker_load_ms": load_ms,
        "reranker_latency_ms": latency_summary(timings),
        "all_queries": {
            "dense": _evaluate(queries, dense),
            "always_rerank": _evaluate(queries, reranked),
        },
        "conditional": {
            "calibration_query_count": len(calibration),
            "evaluation_query_count": len(evaluation),
            "quality_floor": target * 0.99,
            "trials": trials,
            "selected_max_margin": selected["max_margin"],
            "evaluation_trigger_rate": evaluation_trigger_rate,
            "dense_evaluation": dense_evaluation,
            "always_rerank_evaluation": reranked_evaluation,
            "conditional_evaluation": conditional_evaluation,
            "passed_quality_floor": conditional_evaluation["ndcg@10"] >= evaluation_target * 0.99,
        },
        "limitations": [
            "The selected cross-encoder is English-only and is not a production multilingual candidate.",
            "The pilot calibrates and evaluates on alternating rows of one validation slice.",
        ],
    }


def parser() -> argparse.ArgumentParser:
    value = argparse.ArgumentParser(description="Run the SubaRAG reranking ablation")
    value.add_argument("--manifest", type=Path, required=True)
    value.add_argument("--embedding-cache", type=Path, required=True)
    value.add_argument("--output", type=Path, required=True)
    value.add_argument("--language", choices=("en",), default="en")
    value.add_argument("--embedder", default="intfloat/multilingual-e5-small")
    value.add_argument("--reranker", default="cross-encoder/ms-marco-MiniLM-L6-v2")
    value.add_argument("--documents", type=int, default=2_000)
    value.add_argument("--queries", type=int, default=200)
    value.add_argument("--candidate-k", type=int, default=32)
    value.add_argument("--batch-size", type=int, default=32)
    return value


def main() -> None:
    args = parser().parse_args()
    report = run(args)
    args.output.mkdir(parents=True, exist_ok=True)
    destination = args.output / f"reranking-{args.language}-{report['artifact']['documents']}.json"
    temporary = destination.with_suffix(".json.tmp")
    temporary.write_text(json.dumps(report, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    temporary.replace(destination)
    print(destination.resolve())


if __name__ == "__main__":
    main()
