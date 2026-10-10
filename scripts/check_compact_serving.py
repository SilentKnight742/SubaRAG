"""Package the provisional compact runtime and evaluate fresh queries in Colab.

No model export, quantization search, network API calls, artifact upload or deploy.
"""

from __future__ import annotations

import argparse
import asyncio
from concurrent.futures import ThreadPoolExecutor
import hashlib
import importlib.util
import json
import os
from pathlib import Path
import shutil
import sqlite3
import subprocess
import sys
import tempfile
import unicodedata


def read(path):
    return json.loads(Path(path).read_text())


def write(path, value):
    Path(path).write_text(json.dumps(value, indent=2, ensure_ascii=False) + "\n")


def digest(path):
    from subarag.retrieval.onnx_embedding import file_sha256
    return file_sha256(Path(path))


def preparation_helper():
    source = Path(__file__).resolve().parent / "check_embedding_tables.py"
    if digest(source) != "0fa3f32749925823f12ff530ddc8224842d66d36a1c81ad4065c61f4f42bb3ee":
        raise ValueError("Use the complete current source archive")
    spec = importlib.util.spec_from_file_location("verified_table_helper", source)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def make_bundle(work, output):
    from subarag.retrieval.compact import FILES, MODEL, verify_compact_bundle
    prior, disk_report, tracked, _baseline = preparation_helper().sources(work)
    model = prior.parent / "preprocessed-uint8/model"
    disk = disk_report.parent
    tokenizer = Path(read(disk / "screen-inputs.json")["tokenizer_report"]).parent
    stores = read(disk / "stores.json")
    bundle = output / "bundle"
    bundle.mkdir()
    sources = {f"encoder/{name}": model / name for name in ("model.onnx", "tokenizer.json", "embedding-manifest.json")}
    sources.update({f"tokenizer/{name}": tokenizer / name for name in (
        "float64-manifest.json", "vocabulary.marisa", "scores.bin", "pipeline.json")})
    languages = {}
    for lang, item in stores.items():
        index_manifest = Path(item["manifest_path"])
        sources[f"{lang}/index.faiss"] = index_manifest.parent / "index.faiss"
        sources[f"{lang}/documents.sqlite"] = disk / item["database"]
        languages[lang] = {"documents": item["count"], "corpus_version": read(index_manifest)["corpus_version"]}
    if set(sources) != FILES:
        raise ValueError("Incomplete compact bundle")
    for relative, source in sources.items():
        destination = bundle / relative
        destination.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(source, destination)
    metadata = {"schema_version": 1, "artifact_kind": "subarag_compact_bundle", "model_name": MODEL,
                "model_revision": read(model / "embedding-manifest.json")["model_revision"], "dimension": 384,
                "languages": languages, "provisional": True,
                "quality_note": "Accepted for portfolio validation, not a claim of meeting the old 2% per-language gate.",
                "source_preprocessing_report_sha256": digest(prior),
                "files": {name: {"bytes": (bundle / name).stat().st_size, "sha256": digest(bundle / name)} for name in sorted(FILES)}}
    write(bundle / "compact-bundle.json", metadata)
    verify_compact_bundle(bundle / "compact-bundle.json")
    if any(digest(path) != checksum for path, checksum in tracked.items()):
        raise ValueError("An original artifact changed while packaging")
    write(output / "inputs.json", {"preprocessing_report": str(prior), "disk_report": str(disk_report),
          "old_cases": read(work / "inputs.json")["cases"],
          "baseline_measurement": str(prior.parent / "preprocessed-uint8/measurement.json"),
          "fp32_directory": str(prior.parent / "fp32-preprocessed/model"),
          "source_sha256": {str(path): checksum for path, checksum in tracked.items()}})
    print("Packaged existing model, tokenizer, documents and indexes; no rebuild.", flush=True)


def fresh_cases(work, output):
    from subarag.data.manifest import verify_manifest
    corpus = Path(read(work / "inputs.json")["corpus"])
    manifest = verify_manifest(corpus)
    inputs = read(output / "inputs.json")
    inputs["source_sha256"][str(corpus)] = digest(corpus)
    for shard in manifest.shards:
        source = corpus.parent / shard.path
        inputs["source_sha256"][str(source)] = digest(source)
    write(output / "inputs.json", inputs)
    old = inputs["old_cases"]
    normalize = lambda text: " ".join(unicodedata.normalize("NFC", text).casefold().split())
    used_ids = {(case["language"], case["id"]) for case in old}
    used_texts = {(case["language"], normalize(case["query"])) for case in old}
    selected = []
    for lang in ("en", "hi", "mr"):
        rows = []
        for shard in manifest.shards:
            if shard.kind == "evaluation" and shard.language == lang and shard.split == "validation":
                with (corpus.parent / shard.path).open() as handle:
                    rows.extend(json.loads(line) for line in handle)
        rows.sort(key=lambda row: hashlib.sha256(("portfolio-v1:" + lang + ":" + row["id"]).encode()).hexdigest())
        count = 0
        connection = sqlite3.connect((output / "bundle" / lang / "documents.sqlite").as_uri() + "?mode=ro&immutable=1", uri=True)
        try:
            for row in rows:
                key = lang, normalize(row["query"])
                if (lang, row["id"]) in used_ids or key in used_texts:
                    continue
                relevant = [identifier for identifier in row["relevant_passage_ids"]
                            if connection.execute("SELECT 1 FROM documents WHERE id=?", (identifier,)).fetchone()]
                if not relevant:
                    continue
                selected.append({"id": row["id"], "language": lang, "query": row["query"], "relevant": relevant})
                used_ids.add((lang, row["id"]))
                used_texts.add(key)
                count += 1
                if count == 64:
                    break
        finally:
            connection.close()
        if count != 64:
            raise ValueError(f"Only {count} unused validation questions available for {lang}; no substitution with old queries")
    write(output / "fresh-cases.json", selected)
    print("Selected 64 fresh questions per language, excluding old IDs and duplicate question text.", flush=True)


def memory():
    return max(int(line.split()[1]) / 1024 for line in Path("/proc/self/status").read_text().splitlines()
               if line.startswith(("VmRSS:", "VmHWM:")))


async def measure(output, mode):
    # This standalone check must never inherit live application configuration.
    for key in tuple(os.environ):
        if key.startswith("SUBARAG_") or key in ("HF_TOKEN", "GEMINI_API_KEY", "ELEVENLABS_API_KEY", "SARVAM_API_KEY"):
            os.environ.pop(key)
    from subarag.api.app import create_app
    from subarag.api.settings import Settings
    from subarag.retrieval.compact import CompactRetriever
    from subarag.retrieval.onnx_embedding import OnnxQueryEmbedder
    from subarag.retrieval import EvaluationQuery, ranking_metrics, latency_summary
    from subarag.experiments.compact_embedding import PROBES
    from time import perf_counter
    config = read(output / "inputs.json")
    def factory(directory, **kwargs):
        return OnnxQueryEmbedder(Path(config["fp32_directory"]) if mode == "fp32" else directory, **kwargs)
    retriever = CompactRetriever(output / "bundle/compact-bundle.json", embedder_factory=factory)
    executor = ThreadPoolExecutor(max_workers=1)
    peak = memory()
    async def retrieve(query, language):
        nonlocal peak
        result = await asyncio.get_running_loop().run_in_executor(executor, retriever.retrieve, query, language)
        peak = max(peak, memory())
        return result
    app = create_app(settings=Settings(_env_file=None, elevenlabs_api_key="fixture-not-used"))
    metrics, timings, rankings = {}, {}, {}
    fresh = read(output / "fresh-cases.json")
    try:
        async with app.router.lifespan_context(app):
            old_match = None
            if mode == "compact":
                baseline = read(config["baseline_measurement"])["measurement"]
                old_match = True
                for case in config["old_cases"]:
                    hits = (await retrieve(case["query"], case["language"])).hits
                    old_match &= [hit.document.id for hit in hits] == baseline["rankings"][case["language"] + ":" + case["id"]]
                if not old_match:
                    raise ValueError("Serving runtime differs from the saved development rankings")
            for lang in ("en", "hi", "mr"):
                queries, found, elapsed = [], {}, []
                for case in (c for c in fresh if c["language"] == lang):
                    started = perf_counter()
                    outcome = await retrieve(case["query"], lang)
                    elapsed.append((perf_counter() - started) * 1000)
                    found[case["id"]] = outcome.hits
                    queries.append(EvaluationQuery(case["id"], lang, case["query"], frozenset(case["relevant"])))
                    rankings[lang + ":" + case["id"]] = [hit.document.id for hit in outcome.hits]
                metrics[lang] = ranking_metrics(queries, found, k=8)
                timings[lang] = latency_summary(elapsed)
                print(mode, lang, metrics[lang], flush=True)
            stress = []
            for _ in range(2):
                for lang, text in zip(("en", "hi", "mr"), PROBES, strict=True):
                    encoded = retriever.tokenizer.encode("query: " + text * 600)
                    if len(encoded.ids) != 512:
                        raise ValueError("Long probe no longer reaches 512 tokens")
                    started = perf_counter()
                    await retrieve(text * 600, lang)
                    stress.append((perf_counter() - started) * 1000)
                    await retrieve(text, lang)
            switched, stable = [], True
            for _ in range(3):
                for lang in ("en", "hi", "mr"):
                    case = next(c for c in fresh if c["language"] == lang)
                    started = perf_counter()
                    outcome = await retrieve(case["query"], lang)
                    switched.append((perf_counter() - started) * 1000)
                    stable &= [hit.document.id for hit in outcome.hits] == rankings[lang + ":" + case["id"]]
            forbidden = any(name in sys.modules for name in ("torch", "transformers", "sentence_transformers", "sentencepiece"))
            write(output / f"{mode}.json", {"mode": mode, "metrics": metrics, "retrieval_ms": timings,
                  "peak_rss_mib": max(peak, memory()), "old_rankings_identical": old_match,
                  "long_retrieval_ms": latency_summary(stress), "switching_ms": latency_summary(switched),
                  "switch_rankings_stable": stable, "heavyweight_imported": forbidden,
                  "fresh_cases_sha256": digest(output / "fresh-cases.json")})
    finally:
        executor.shutdown(wait=True, cancel_futures=True)
        retriever.close()


def run(work):
    output = Path(tempfile.mkdtemp(prefix="compact-serving-", dir=work))
    print("Output:", output, flush=True)
    make_bundle(work, output)
    fresh_cases(work, output)
    env = dict(os.environ, HF_HUB_OFFLINE="1", TRANSFORMERS_OFFLINE="1", OMP_NUM_THREADS="1",
               OPENBLAS_NUM_THREADS="1", TOKENIZERS_PARALLELISM="false")
    env.pop("HF_TOKEN", None)
    for mode in ("compact", "fp32"):
        subprocess.run([sys.executable, "-u", str(Path(__file__).resolve()), "--work", str(work),
                        "--output", str(output), "--child", mode], env=env, check=True)
    compact, reference = read(output / "compact.json"), read(output / "fp32.json")
    tracked = read(output / "inputs.json")["source_sha256"]
    unchanged = all(digest(path) == checksum for path, checksum in tracked.items())
    report = {"schema_version": 1, "script_sha256": digest(__file__), "source_sha256": tracked,
              "original_files_unchanged": unchanged, "bundle_manifest": str(output / "bundle/compact-bundle.json"),
              "bundle_manifest_sha256": digest(output / "bundle/compact-bundle.json"),
              "fresh_queries_per_language": 64, "old_query_ids_and_text_excluded": True,
              "compact": compact, "fp32": reference,
              "relative_ndcg_change": {lang: compact["metrics"][lang]["ndcg@8"] / reference["metrics"][lang]["ndcg@8"] - 1
                                       if reference["metrics"][lang]["ndcg@8"] else None for lang in ("en", "hi", "mr")},
              "within_400_mib_screen": compact["peak_rss_mib"] <= 400,
              "answer_quality": "Not evaluated: no generation API calls made",
              "hosting_memory_limit": "Not yet verified: Colab process RSS is not a 512 MiB container test",
              "note": "Provisional portfolio candidate. No 2% quality gate, no automatic release. Fresh same-corpus retrieval check, not an independently curated answer benchmark. No upload or deployment."}
    write(output / "compact-serving-report.json", report)
    print("REPORT:", output / "compact-serving-report.json", flush=True)
    if not unchanged or not compact["old_rankings_identical"] or compact["heavyweight_imported"] or not compact["switch_rankings_stable"]:
        raise ValueError("Compact serving correctness check failed")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--work", type=Path, required=True)
    parser.add_argument("--output", type=Path)
    parser.add_argument("--child", choices=("compact", "fp32"))
    args = parser.parse_args()
    if args.child:
        if args.output is None:
            parser.error("Child measurement requires --output")
        asyncio.run(measure(args.output, args.child))
    else:
        run(args.work.resolve())
