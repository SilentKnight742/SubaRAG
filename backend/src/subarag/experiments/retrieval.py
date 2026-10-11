"""Run SubaRAG retrieval ablations against a prepared corpus artifact."""

from __future__ import annotations

import argparse
import gc
import hashlib
import json
import os
import platform
import re
from datetime import datetime, timezone
from pathlib import Path
from time import perf_counter

from ..data.manifest import load_manifest
from ..retrieval import (
    AdaptiveRetriever,
    Bm25SparseIndex,
    ExactDenseIndex,
    HnswDenseIndex,
    SentenceTransformerEmbedder,
    extract_supporting_spans,
    load_language_artifact,
    reciprocal_rank_fusion,
    route_lexical,
    select_adaptive_k,
    unicode_tokens,
)
from ..retrieval.metrics import latency_summary, ranking_metrics


FIXED_K = (1, 2, 4, 6, 8, 12)
ADAPTIVE_GAPS = (0.02, 0.04, 0.06, 0.08, 0.12, 0.16)
PREFIX_FRACTIONS = (0.4, 0.6, 0.8)


def _model_slug(model: str) -> str:
    return re.sub(r"[^a-zA-Z0-9._-]+", "--", model)


def _artifact_fingerprint(documents, queries) -> str:
    digest = hashlib.sha256()
    for document in documents:
        digest.update(document.id.encode())
        digest.update(b"\0")
    for query in queries:
        digest.update(query.id.encode())
        digest.update(b"\0")
    return digest.hexdigest()


def _timed_rankings(index, queries, query_embeddings, candidate_k: int):
    rankings = {}
    timings = []
    if queries:
        index.search(query_embeddings[0], candidate_k)
    for query, embedding in zip(queries, query_embeddings, strict=True):
        started = perf_counter()
        rankings[query.id] = index.search(embedding, candidate_k)
        timings.append((perf_counter() - started) * 1000)
    return rankings, timings


def _timed_bm25(index, queries, candidate_k: int):
    rankings = {}
    timings = []
    if queries:
        index.search(queries[0].query, candidate_k)
    for query in queries:
        started = perf_counter()
        rankings[query.id] = index.search(query.query, candidate_k)
        timings.append((perf_counter() - started) * 1000)
    return rankings, timings


def _subset_rankings(rankings, k: int):
    return {identifier: hits[:k] for identifier, hits in rankings.items()}


def _adaptive_rankings(rankings, *, gap: float, buffer: int = 0):
    selected = {}
    ks = []
    for identifier, hits in rankings.items():
        choice = select_adaptive_k(hits, min_relative_gap=gap, buffer=buffer)
        selected[identifier] = hits[:choice.k]
        ks.append(choice.k)
    return selected, ks


def _evaluate_k(queries, rankings):
    metrics = ranking_metrics(queries, rankings, k=10)
    metrics["mean_k"] = sum(len(rankings[query.id]) for query in queries) / len(queries)
    return metrics


def _calibration_split(queries):
    calibration = [query for index, query in enumerate(queries) if index % 2 == 0]
    evaluation = [query for index, query in enumerate(queries) if index % 2 == 1]
    if not calibration or not evaluation:
        return queries, queries
    return calibration, evaluation


def _select_context_policy(queries, rankings):
    calibration, evaluation = _calibration_split(queries)
    fixed = {str(k): _evaluate_k(calibration, _subset_rankings(rankings, k)) for k in FIXED_K}
    best_quality = max(value["ndcg@10"] for value in fixed.values())
    trials = []
    for gap in ADAPTIVE_GAPS:
        for buffer in (0, 1):
            adaptive, _ = _adaptive_rankings(rankings, gap=gap, buffer=buffer)
            result = _evaluate_k(calibration, adaptive)
            trials.append({"gap": gap, "buffer": buffer, **result})
    candidates = [
        {"type": "fixed", "k": int(k), **result}
        for k, result in fixed.items()
    ] + [
        {"type": "adaptive", **trial}
        for trial in trials
    ]
    eligible = [candidate for candidate in candidates if candidate["ndcg@10"] >= best_quality * 0.99]
    selected = min(
        eligible or candidates,
        key=lambda candidate: (candidate["mean_k"], candidate["type"] != "fixed", -candidate["ndcg@10"]),
    )
    if selected["type"] == "fixed":
        evaluation_rankings = _subset_rankings(rankings, selected["k"])
        selected_policy = {"type": "fixed", "k": selected["k"]}
    else:
        evaluation_rankings, _ = _adaptive_rankings(
            rankings, gap=selected["gap"], buffer=selected["buffer"],
        )
        selected_policy = {
            "type": "adaptive", "gap": selected["gap"], "buffer": selected["buffer"],
        }
    evaluation_result = _evaluate_k(evaluation, evaluation_rankings)
    evaluation_query_ids = {query.id for query in evaluation}
    evaluation_ks = [len(hits) for identifier, hits in evaluation_rankings.items()
                     if identifier in evaluation_query_ids]
    evaluation_result["min_k"] = min(evaluation_ks)
    evaluation_result["max_k"] = max(evaluation_ks)
    return {
        "calibration_query_count": len(calibration),
        "evaluation_query_count": len(evaluation),
        "fixed_k_calibration": fixed,
        "adaptive_trials": trials,
        "quality_floor": best_quality * 0.99,
        "selected": selected_policy,
        "evaluation": evaluation_result,
    }


def _quantiles(values):
    ordered = sorted(values)
    if not ordered:
        return [0.0]
    return [ordered[round((len(ordered) - 1) * fraction)] for fraction in (0.25, 0.5, 0.75)]


def _conditional_rankings(queries, dense_rankings, hybrid, thresholds):
    conditional = {}
    triggered = 0
    reasons: dict[str, int] = {}
    for query in queries:
        decision = route_lexical(
            query.query, dense_rankings[query.id],
            min_top_score=thresholds["min_top_score"],
            min_margin=thresholds["min_margin"],
            max_spread=thresholds["max_spread"],
        )
        if decision.run_lexical:
            triggered += 1
            conditional[query.id] = hybrid[query.id]
            for reason in decision.reasons:
                reasons[reason] = reasons.get(reason, 0) + 1
        else:
            conditional[query.id] = dense_rankings[query.id]
    return conditional, triggered / len(queries), reasons


def _hybrid_experiment(queries, dense_rankings, bm25_rankings, candidate_k):
    dense_metrics = _evaluate_k(queries, dense_rankings)
    bm25_metrics = _evaluate_k(queries, bm25_rankings)
    hybrid = {
        query.id: reciprocal_rank_fusion(dense_rankings[query.id], bm25_rankings[query.id])[:candidate_k]
        for query in queries
    }
    hybrid_metrics = _evaluate_k(queries, hybrid)
    calibration, evaluation = _calibration_split(queries)
    top_scores = [dense_rankings[query.id][0].score for query in calibration]
    margins = [dense_rankings[query.id][0].score - dense_rankings[query.id][1].score
               for query in calibration if len(dense_rankings[query.id]) > 1]
    spreads = [dense_rankings[query.id][0].score - dense_rankings[query.id][min(7, len(dense_rankings[query.id]) - 1)].score
               for query in calibration]
    top_options = [min(top_scores) - 1e-6, *_quantiles(top_scores)]
    margin_options = [0.0, *_quantiles(margins)]
    spread_options = [0.0, *_quantiles(spreads)]
    target = max(
        _evaluate_k(calibration, dense_rankings)["ndcg@10"],
        _evaluate_k(calibration, hybrid)["ndcg@10"],
    )
    trials = []
    for top_score in top_options:
        for margin in margin_options:
            for spread in spread_options:
                thresholds = {
                    "min_top_score": top_score, "min_margin": margin, "max_spread": spread,
                }
                conditional, trigger_rate, _ = _conditional_rankings(
                    calibration, dense_rankings, hybrid, thresholds,
                )
                result = _evaluate_k(calibration, conditional)
                trials.append({**thresholds, "trigger_rate": trigger_rate, **result})
    dense_calibration = _evaluate_k(calibration, dense_rankings)
    dense_only_qualifies = dense_calibration["ndcg@10"] >= target * 0.99
    eligible = [trial for trial in trials if trial["ndcg@10"] >= target * 0.99]
    selected = min(eligible or trials, key=lambda trial: (trial["trigger_rate"], -trial["ndcg@10"]))
    thresholds = {key: selected[key] for key in ("min_top_score", "min_margin", "max_spread")}
    if dense_only_qualifies:
        conditional = {query.id: dense_rankings[query.id] for query in evaluation}
        all_conditional = dense_rankings
        trigger_rate = 0.0
        reasons = {}
        selected_route = "dense_only"
    else:
        conditional, trigger_rate, reasons = _conditional_rankings(
            evaluation, dense_rankings, hybrid, thresholds,
        )
        all_conditional, _, _ = _conditional_rankings(queries, dense_rankings, hybrid, thresholds)
        selected_route = "conditional_hybrid"
    conditional_metrics = _evaluate_k(evaluation, conditional)
    dense_evaluation = _evaluate_k(evaluation, dense_rankings)
    hybrid_evaluation = _evaluate_k(evaluation, hybrid)
    evaluation_target = max(dense_evaluation["ndcg@10"], hybrid_evaluation["ndcg@10"])
    return {
        "dense": dense_metrics,
        "bm25": bm25_metrics,
        "always_hybrid": hybrid_metrics,
        "conditional_hybrid": {
            **conditional_metrics,
            "calibration_query_count": len(calibration),
            "evaluation_query_count": len(evaluation),
            "selected_route": selected_route,
            "selected_thresholds": thresholds,
            "trigger_rate": trigger_rate,
            "trigger_reasons": reasons,
            "dense_evaluation": dense_evaluation,
            "always_hybrid_evaluation": hybrid_evaluation,
            "passed_quality_floor": conditional_metrics["ndcg@10"] >= evaluation_target * 0.99,
        },
        "conditional_rankings": all_conditional,
    }


def _compression_experiment(queries, rankings):
    original = compressed = 0
    empty = 0
    for query in queries:
        hits = rankings[query.id][:6]
        original += sum(len(unicode_tokens(hit.document.text)) for hit in hits)
        spans = extract_supporting_spans(query.query, hits)
        compressed += sum(len(unicode_tokens(text)) for _, text in spans)
        empty += not bool(spans)
    return {
        "original_tokens": original,
        "compressed_tokens": compressed,
        "reduction": 1 - compressed / original if original else 0,
        "empty_query_count": empty,
        "note": "Whitespace-independent Unicode token approximation; answer quality is not measured here.",
    }


def _prefixes(text: str):
    words = text.split()
    for fraction in PREFIX_FRACTIONS:
        length = max(1, round(len(words) * fraction))
        yield fraction, " ".join(words[:length])


def _voice_speculation(embedder, index, queries, final_rankings, *, k: int = 10):
    prefixed = [(query, fraction, text) for query in queries for fraction, text in _prefixes(query.query)]
    embeddings = embedder.embed_queries([text for _, _, text in prefixed])
    overlaps = {str(fraction): [] for fraction in PREFIX_FRACTIONS}
    exact_top1 = {str(fraction): 0 for fraction in PREFIX_FRACTIONS}
    counts = {str(fraction): 0 for fraction in PREFIX_FRACTIONS}
    for (query, fraction, _), embedding in zip(prefixed, embeddings, strict=True):
        key = str(fraction)
        partial = index.search(embedding, k)
        final = final_rankings[query.id][:k]
        partial_ids = {hit.document.id for hit in partial}
        final_ids = {hit.document.id for hit in final}
        overlaps[key].append(len(partial_ids & final_ids) / max(1, len(partial_ids | final_ids)))
        exact_top1[key] += partial[0].document.id == final[0].document.id
        counts[key] += 1
    return {
        key: {
            "mean_top10_jaccard": sum(values) / len(values),
            "same_top1_rate": exact_top1[key] / counts[key],
            "queries": counts[key],
        }
        for key, values in overlaps.items()
    }


def _cache_experiment(embedder, hnsw, bm25, queries, corpus_version, retriever_version):
    retriever = AdaptiveRetriever(
        embedder=embedder, dense_index=hnsw, lexical_index=bm25,
        corpus_version=corpus_version, retriever_version=retriever_version,
    )
    sample = queries[: min(20, len(queries))]
    cold = []
    warm = []
    for query in sample:
        started = perf_counter()
        retriever.retrieve(query.query, query.language)
        cold.append((perf_counter() - started) * 1000)
    for query in sample:
        started = perf_counter()
        outcome = retriever.retrieve(query.query, query.language)
        warm.append((perf_counter() - started) * 1000)
        if not outcome.telemetry.cache_hit:
            raise RuntimeError("Expected a versioned retrieval cache hit")
    return {"queries": len(sample), "cold_ms": latency_summary(cold), "warm_ms": latency_summary(warm)}


def _embedding_latency(embedder, queries, limit: int):
    sample = queries[: min(limit, len(queries))]
    if sample:
        embedder.embed_queries([sample[0].query])
    timings = []
    for query in sample:
        started = perf_counter()
        embedder.embed_queries([query.query])
        timings.append((perf_counter() - started) * 1000)
    return latency_summary(timings)


def _run_model(model, documents, queries, args, output_root, fingerprint, corpus_version):
    import numpy as np
    import psutil

    process = psutil.Process()
    model_started = perf_counter()
    embedder = SentenceTransformerEmbedder(model, batch_size=args.batch_size)
    model_load_ms = (perf_counter() - model_started) * 1000
    embedding_dir = output_root / "embeddings" / args.language / _model_slug(model)
    embedding_dir.mkdir(parents=True, exist_ok=True)
    cache_path = embedding_dir / f"{len(documents)}-{fingerprint[:16]}.npz"
    if cache_path.exists():
        cached = np.load(cache_path)
        document_embeddings = cached["documents"]
        query_embeddings = cached["queries"]
        embedding_build_ms = 0.0
        embeddings_cached = True
    else:
        started = perf_counter()
        document_embeddings = embedder.embed_documents([document.text for document in documents])
        query_embeddings = embedder.embed_queries([query.query for query in queries])
        embedding_build_ms = (perf_counter() - started) * 1000
        np.savez(cache_path, documents=document_embeddings, queries=query_embeddings)
        embeddings_cached = False
    rss_after_embeddings = process.memory_info().rss

    started = perf_counter()
    exact = ExactDenseIndex(documents, document_embeddings)
    exact_build_ms = (perf_counter() - started) * 1000
    exact_rankings, exact_timings = _timed_rankings(exact, queries, query_embeddings, args.candidate_k)

    started = perf_counter()
    hnsw = HnswDenseIndex(
        documents, document_embeddings, m=args.hnsw_m,
        ef_construction=args.hnsw_ef_construction, ef_search=args.hnsw_ef_search,
    )
    hnsw_build_ms = (perf_counter() - started) * 1000
    hnsw_rankings, hnsw_timings = _timed_rankings(hnsw, queries, query_embeddings, args.candidate_k)

    started = perf_counter()
    bm25 = Bm25SparseIndex(documents)
    bm25_build_ms = (perf_counter() - started) * 1000
    bm25_rankings, bm25_timings = _timed_bm25(bm25, queries, args.candidate_k)

    hybrid = _hybrid_experiment(queries, hnsw_rankings, bm25_rankings, args.candidate_k)
    conditional_rankings = hybrid.pop("conditional_rankings")
    context = _select_context_policy(queries, conditional_rankings)
    if context["selected"]["type"] == "fixed":
        context_rankings = _subset_rankings(conditional_rankings, context["selected"]["k"])
    else:
        context_rankings, _ = _adaptive_rankings(
            conditional_rankings, gap=context["selected"]["gap"],
            buffer=context["selected"]["buffer"],
        )

    result = {
        "model": model,
        "model_load_ms": model_load_ms,
        "dimension": embedder.dimension,
        "embedding_build_ms": embedding_build_ms,
        "embeddings_cached": embeddings_cached,
        "query_embedding_ms": _embedding_latency(embedder, queries, args.timing_queries),
        "rss_after_embeddings_bytes": rss_after_embeddings,
        "exact": {
            "build_ms": exact_build_ms,
            "index_bytes": exact.size_bytes,
            "search_ms": latency_summary(exact_timings),
            "metrics": ranking_metrics(queries, exact_rankings, k=10),
        },
        "hnsw": {
            "build_ms": hnsw_build_ms,
            "index_bytes": hnsw.size_bytes,
            "search_ms": latency_summary(hnsw_timings),
            "metrics": ranking_metrics(queries, hnsw_rankings, k=10),
        },
        "bm25": {
            "build_ms": bm25_build_ms,
            "search_ms": latency_summary(bm25_timings),
            "metrics": ranking_metrics(queries, bm25_rankings, k=10),
        },
        "hybrid": hybrid,
        "context_selection": context,
        "compression": _compression_experiment(queries, context_rankings),
        "voice_prefix_speculation": _voice_speculation(
            embedder, hnsw, queries[: args.speculation_queries], hnsw_rankings,
        ),
        "cache": _cache_experiment(
            embedder, hnsw, bm25, queries, corpus_version,
            f"{_model_slug(model)}-hnsw-m{args.hnsw_m}-ef{args.hnsw_ef_search}",
        ),
    }
    del bm25, hnsw, exact, embedder, document_embeddings, query_embeddings
    gc.collect()
    return result


def _selection(models):
    best_quality = max(result["hnsw"]["metrics"]["ndcg@10"] for result in models)
    eligible = [result for result in models if result["hnsw"]["metrics"]["ndcg@10"] >= best_quality * 0.98]
    chosen = min(eligible, key=lambda result: result["query_embedding_ms"]["p95"])
    return {
        "rule": "lowest query-embedding p95 within 2% of best HNSW nDCG@10",
        "best_ndcg@10": best_quality,
        "selected_model": chosen["model"],
    }


def run(args) -> dict:
    manifest = load_manifest(args.manifest, verify_files=True)
    documents, queries = load_language_artifact(
        args.manifest, args.language, max_documents=args.documents, max_queries=args.queries,
    )
    if not queries:
        raise ValueError("No validation queries have relevant passages in this corpus slice")
    output_root = args.output.resolve()
    output_root.mkdir(parents=True, exist_ok=True)
    fingerprint = _artifact_fingerprint(documents, queries)
    corpus_version = f"{manifest.source.revision[:12]}-{fingerprint[:12]}"
    model_results = [
        _run_model(model, documents, queries, args, output_root, fingerprint, corpus_version)
        for model in args.models
    ]
    return {
        "schema_version": 1,
        "created_at": datetime.now(timezone.utc).isoformat(),
        "environment": {
            "python": platform.python_version(), "platform": platform.platform(),
            "cpu_count": os.cpu_count(), "machine": platform.machine(),
        },
        "artifact": {
            "manifest": str(args.manifest.resolve()), "source_revision": manifest.source.revision,
            "fingerprint": fingerprint, "language": args.language,
            "documents": len(documents), "queries": len(queries),
        },
        "configuration": {
            "candidate_k": args.candidate_k, "batch_size": args.batch_size,
            "hnsw": {"m": args.hnsw_m, "ef_construction": args.hnsw_ef_construction,
                     "ef_search": args.hnsw_ef_search},
        },
        "models": model_results,
        "selection": _selection(model_results),
        "limitations": [
            "This pilot uses one validation slice for policy calibration and evaluation; final claims need a separate held-out set.",
            "Voice speculation uses text prefixes, not streaming ASR partials.",
            "Compression reports token reduction, not downstream answer quality.",
            "No generation model is included, so TTFT and end-to-end answer quality are not measured here.",
        ],
    }


def parser() -> argparse.ArgumentParser:
    value = argparse.ArgumentParser(description="Run reproducible SubaRAG retrieval experiments")
    value.add_argument("--manifest", type=Path, required=True)
    value.add_argument("--output", type=Path, required=True)
    value.add_argument("--language", choices=("en", "hi", "mr"), required=True)
    value.add_argument("--models", nargs="+", default=["intfloat/multilingual-e5-small"])
    value.add_argument("--documents", type=int, default=10_000)
    value.add_argument("--queries", type=int, default=200)
    value.add_argument("--candidate-k", type=int, default=32)
    value.add_argument("--batch-size", type=int, default=32)
    value.add_argument("--timing-queries", type=int, default=30)
    value.add_argument("--speculation-queries", type=int, default=40)
    value.add_argument("--hnsw-m", type=int, default=32)
    value.add_argument("--hnsw-ef-construction", type=int, default=200)
    value.add_argument("--hnsw-ef-search", type=int, default=64)
    return value


def main() -> None:
    args = parser().parse_args()
    if args.documents < 1 or args.queries < 1 or args.candidate_k < 1:
        raise SystemExit("Document, query and candidate counts must be positive")
    report = run(args)
    destination = args.output.resolve() / f"retrieval-{args.language}-{report['artifact']['documents']}.json"
    temporary = destination.with_suffix(".json.tmp")
    temporary.write_text(json.dumps(report, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    temporary.replace(destination)
    print(destination)


if __name__ == "__main__":
    main()
