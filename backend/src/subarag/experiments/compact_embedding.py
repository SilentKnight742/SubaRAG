"""Colab-only export and isolated CPU memory/quality screen for compact E5.

Export, reference, and candidate run in separate processes. No paid APIs,
generation, automatic upload, or changes to the serving defaults occur here.
"""

from __future__ import annotations

import argparse
import asyncio
import gc
import hashlib
import importlib.metadata
import json
import os
from pathlib import Path
import resource
import sys
from time import perf_counter

from ..retrieval.onnx_embedding import OnnxQueryEmbedder, file_sha256


MODEL = "intfloat/multilingual-e5-small"
MODEL_REVISION = "614241f622f53c4eeff9890bdc4f31cfecc418b3"
DATA_REPO = "silentkirmada/SubaRAG-data"
DATA_REVISION = "e29f5c990a4fc0d8e05771be4007f1c6560adfc2"
INDEX_REVISION = "9f6d55f53b9479930613e4c849e5e834a4230d0a"
INDEX_PATH = "retrieval/e7a5c4e6bd668ccfeadbe8f2178d20f9a8570e51937d3592813af55ded2fac68"
LANGUAGES = ("en", "hi", "mr")
PROBES = ("What is a corporation?", "कंपनी क्या है?", "कंपनी म्हणजे काय?")


def write_json(path: Path, value) -> None:
    path.write_text(json.dumps(value, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")


def process_peak_rss_mib() -> float:
    # Linux getrusage can retain the launcher's pre-exec high-water mark.
    # VmHWM belongs to this process's current address space, so an earlier
    # notebook import cannot falsely fail an otherwise small candidate.
    for line in Path("/proc/self/status").read_text().splitlines():
        if line.startswith("VmHWM:"):
            return int(line.split()[1]) / 1024
    raise RuntimeError("This memory screen requires Linux /proc VmHWM (available in Colab)")


async def prepare(work: Path) -> None:
    from ..data.remote import fetch_prepared_corpus
    from ..data.manifest import load_manifest
    from ..retrieval.remote import fetch_retrieval_bundle

    token = os.environ.get("HF_TOKEN")
    if not token:
        raise ValueError("Supply the read-only HF token in the child process environment")
    corpus = await fetch_prepared_corpus(
        repo_id=DATA_REPO, revision=DATA_REVISION, token=token, cache_dir=work / "corpus",
    )
    indexes = await fetch_retrieval_bundle(
        repo_id=DATA_REPO, revision=INDEX_REVISION, path=INDEX_PATH, token=token,
        cache_dir=work / "indexes", prepared_manifest=corpus,
        prepared_repo=DATA_REPO, prepared_revision=DATA_REVISION,
    )
    manifest = load_manifest(corpus)
    cases = []
    for language in LANGUAGES:
        # Freeze deterministic, dispersed validation queries, not a tuned hand-picked set.
        rows = []
        for shard in manifest.shards:
            if shard.kind == "evaluation" and shard.language == language and shard.split == "validation":
                with (corpus.parent / shard.path).open() as handle:
                    rows.extend(json.loads(line) for line in handle)
        rows = [row for row in rows if row["relevant_passage_ids"]]
        rows.sort(key=lambda row: hashlib.sha256(row["id"].encode()).hexdigest())
        if len(rows) < 64:
            raise ValueError("Not enough labeled validation queries for the frozen screen")
        cases.extend({"id": row["id"], "language": language, "query": row["query"],
                      "relevant": row["relevant_passage_ids"]} for row in rows[:64])
    write_json(work / "inputs.json", {
        "corpus": str(corpus), "indexes": {key: str(value) for key, value in indexes.items()},
        "cases": cases, "prepared_revision": DATA_REVISION,
        "retrieval_revision": INDEX_REVISION, "retrieval_path": INDEX_PATH,
    })
    print("Verified existing corpus/indexes; froze 64 queries per language.", flush=True)


def export(work: Path) -> None:
    import numpy as np
    import onnxruntime as ort
    from onnxruntime.quantization import QuantType, quantize_dynamic
    import torch
    from transformers import AutoModel, AutoTokenizer
    from ..retrieval.embedding import SentenceTransformerEmbedder

    ort.disable_telemetry_events()
    output = work / "compact-model"
    if output.exists():
        raise ValueError("Model output already exists; keep it and use a fresh work directory for a new export")
    output.mkdir()
    torch.set_num_threads(1)
    tokenizer = AutoTokenizer.from_pretrained(MODEL, revision=MODEL_REVISION)
    model = AutoModel.from_pretrained(MODEL, revision=MODEL_REVISION, attn_implementation="eager")
    model.eval()

    class PooledEncoder(torch.nn.Module):
        def __init__(self, encoder):
            super().__init__()
            self.encoder = encoder

        def forward(self, input_ids, attention_mask):
            hidden = self.encoder(input_ids=input_ids, attention_mask=attention_mask, return_dict=False)[0]
            mask = attention_mask.unsqueeze(-1).to(hidden.dtype)
            pooled = (hidden * mask).sum(dim=1) / mask.sum(dim=1).clamp(min=1e-9)
            return torch.nn.functional.normalize(pooled, p=2, dim=1)

    wrapper = PooledEncoder(model).eval()
    texts = ["query: " + item for item in PROBES]
    tokens = tokenizer(texts, padding=True, truncation=True, max_length=512, return_tensors="pt")
    fp32 = work / "e5-fp32.onnx"
    print("Exporting pinned E5 with masked mean pooling and normalized output…", flush=True)
    with torch.no_grad():
        torch.onnx.export(
            wrapper, (tokens["input_ids"], tokens["attention_mask"]), str(fp32),
            input_names=["input_ids", "attention_mask"], output_names=["sentence_embedding"],
            dynamic_axes={"input_ids": {0: "batch", 1: "sequence"},
                          "attention_mask": {0: "batch", 1: "sequence"},
                          "sentence_embedding": {0: "batch"}},
            opset_version=17, dynamo=False, external_data=False,
        )
    del wrapper, model
    gc.collect()
    # Compare the unquantized export to the actual serving embedder, not just
    # our pooling wrapper. This catches prefix, tokenization, and pooling drift.
    baseline = SentenceTransformerEmbedder(MODEL, revision=MODEL_REVISION, batch_size=1)
    expected = baseline.embed_queries(PROBES)
    del baseline
    gc.collect()
    session = ort.InferenceSession(str(fp32), providers=["CPUExecutionProvider"])
    actual = session.run(["sentence_embedding"], {
        name: tokens[name].numpy() for name in ("input_ids", "attention_mask")
    })[0]
    parity = float(np.min(np.sum(actual * expected, axis=1)))
    if parity < 0.9999 or not np.isfinite(actual).all():
        raise ValueError(f"FP32 export failed Sentence Transformers parity: {parity}")
    del session
    gc.collect()
    print("Quantizing matrix weights AND embedding lookups…", flush=True)
    quantize_dynamic(
        str(fp32), str(output / "model.onnx"),
        op_types_to_quantize=["MatMul", "Gather"], weight_type=QuantType.QUInt8,
        per_channel=False, use_external_data_format=False,
        extra_options={"MatMulConstBOnly": True},
    )
    tokenizer.backend_tokenizer.save(str(output / "tokenizer.json"))
    write_json(output / "embedding-manifest.json", {
        "schema_version": 1, "model_name": MODEL, "model_revision": MODEL_REVISION,
        "dimension": 384, "max_length": 512, "query_prefix": "query: ",
        "pooling": "masked-mean-l2", "quantization": "uint8-matmul-gather",
        "fp32_probe_min_cosine": parity,
        "packages": {name: importlib.metadata.version(name)
                     for name in ("torch", "transformers", "onnx", "onnxruntime", "tokenizers")},
        "files": {name: {"bytes": (output / name).stat().st_size,
                         "sha256": file_sha256(output / name)}
                  for name in ("model.onnx", "tokenizer.json")},
    })
    print("Compact model exported. No upload or deployment.", flush=True)


def measure(work: Path, *, compact: bool) -> None:
    import numpy as np
    import faiss
    from ..retrieval import load_hnsw_artifact, latency_summary, ranking_metrics
    from ..retrieval.models import EvaluationQuery
    from ..api.app import create_app
    from ..api.settings import Settings

    started = perf_counter()
    faiss.omp_set_num_threads(1)
    inputs = json.loads((work / "inputs.json").read_text())
    # Load the model before indexes so temporary session-construction buffers
    # don't overlap the entire corpus. Serving must adopt this order if promoted.
    if compact:
        embedder = OnnxQueryEmbedder(work / "compact-model", expected_revision=MODEL_REVISION)
    else:
        import torch
        torch.set_num_threads(1)
        from ..retrieval.embedding import SentenceTransformerEmbedder
        embedder = SentenceTransformerEmbedder(MODEL, revision=MODEL_REVISION, batch_size=1)
    artifacts = {lang: load_hnsw_artifact(path, expected_model=MODEL)
                 for lang, path in inputs["indexes"].items()}
    if any(item.model_revision != MODEL_REVISION for item in artifacts.values()):
        raise ValueError("Index and query model revisions differ")

    # Include base web/transcription imports and actual HTTP client startup in
    # both processes. The candidate is not allowed to import torch/transformers.
    app = create_app(settings=Settings(_env_file=None, elevenlabs_api_key="fixture-not-used"))
    async def warm_api():
        async with app.router.lifespan_context(app):
            return None
    asyncio.run(warm_api())
    cold_start_ms = (perf_counter() - started) * 1000
    for probe in PROBES:
        embedder.embed_queries([probe])
    timings, metrics, vectors, ranked = {}, {}, {}, {}
    for language in LANGUAGES:
        cases = [case for case in inputs["cases"] if case["language"] == language]
        documents = {item.id for item in artifacts[language].index.documents}
        queries, rankings, elapsed = [], {}, []
        for case in cases:
            relevant = frozenset(case["relevant"]) & documents
            if not relevant:
                raise ValueError("Frozen evaluation query has no relevant document in its index")
            start = perf_counter()
            embedding = embedder.embed_queries([case["query"]])[0]
            hits = artifacts[language].index.search(embedding, 8)
            elapsed.append((perf_counter() - start) * 1000)
            key = language + ":" + case["id"]
            vectors[key] = embedding.tolist()
            ranked[key] = [hit.document.id for hit in hits]
            queries.append(EvaluationQuery(case["id"], language, case["query"], relevant))
            rankings[case["id"]] = hits
        timings[language] = latency_summary(elapsed)
        metrics[language] = ranking_metrics(queries, rankings, k=8)
        print(language, metrics[language], "p95 ms:", round(timings[language]["p95"], 2), flush=True)
    # Exercise the 512-token ceiling, then alternate short and long queries.
    # One batch-size-one inference at a time matches the planned CPU serving path.
    for _ in range(2):
        for probe in PROBES:
            embedder.embed_queries([probe * 600])
            embedder.embed_queries([probe])
    forbidden = any(name in sys.modules for name in ("torch", "transformers", "sentence_transformers"))
    if compact and forbidden:
        raise RuntimeError("Compact measurement accidentally imported the heavyweight runtime")
    report = {
        "runtime": "onnx-uint8" if compact else "sentence-transformers-fp32",
        "model_revision": MODEL_REVISION, "inputs_sha256": file_sha256(work / "inputs.json"),
        "query_count": len(inputs["cases"]), "metrics": metrics, "retrieval_ms": timings,
        "cold_start_ms": cold_start_ms,
        "load_order": "API imports, query model, all three indexes, API client startup",
        "peak_rss_mib": process_peak_rss_mib(),
        "memory_measurement": "Linux /proc/self/status VmHWM, current process address space",
        "getrusage_peak_mib_including_launcher": resource.getrusage(resource.RUSAGE_SELF).ru_maxrss / 1024,
        "heavyweight_runtime_imported": forbidden,
        "includes": "all three full indexes, documents, web imports/startup, encoder, short and 512-token probes",
        "excludes": "live generation, concurrent transcription, response buffers, container page-cache accounting",
        "vectors": vectors, "rankings": ranked,
    }
    write_json(work / ("candidate.json" if compact else "reference.json"), report)
    print("Peak process RSS MiB:", round(report["peak_rss_mib"], 2), flush=True)


def summarize(work: Path) -> None:
    import numpy as np
    reference = json.loads((work / "reference.json").read_text())
    candidate = json.loads((work / "candidate.json").read_text())
    if candidate["inputs_sha256"] != reference["inputs_sha256"]:
        raise ValueError("Reference and candidate used different frozen inputs")
    quality = {lang: candidate["metrics"][lang]["ndcg@8"] >= 0.98 * reference["metrics"][lang]["ndcg@8"]
               and reference["metrics"][lang]["ndcg@8"] > 0 for lang in LANGUAGES}
    cosines = [float(np.dot(candidate["vectors"][key], reference["vectors"][key]))
               for key in reference["vectors"]]
    gates = {
        "peak_rss_at_most_400_mib": candidate["peak_rss_mib"] <= 400,
        "per_language_ndcg_within_2_percent": quality,
        "per_language_retrieval_p95_at_most_500_ms": {
            lang: candidate["retrieval_ms"][lang]["p95"] <= 500 for lang in LANGUAGES},
        "no_heavyweight_runtime": not candidate["heavyweight_runtime_imported"],
    }
    passed = (gates["peak_rss_at_most_400_mib"] and all(quality.values())
              and all(gates["per_language_retrieval_p95_at_most_500_ms"].values())
              and gates["no_heavyweight_runtime"])
    for item in (reference, candidate):
        item.pop("vectors")
        item.pop("rankings")
    report = {
        "schema_version": 1, "screen_passed": passed, "gates": gates,
        "embedding_cosine_min": min(cosines), "embedding_cosine_mean": sum(cosines) / len(cosines),
        "reference": reference, "candidate": candidate,
        "embedding_manifest": json.loads((work / "compact-model/embedding-manifest.json").read_text()),
        "prepared_revision": DATA_REVISION, "retrieval_revision": INDEX_REVISION,
        "retrieval_path": INDEX_PATH,
        "limitation": "Colab process-level screen, not a Render/container memory or latency guarantee; 192 queries are not full answer evaluation.",
        "next_step": "If passed, run a memory-limited serving container check before enabling RAG. If failed, do not deploy this candidate.",
    }
    write_json(work / "compact-report.json", report)
    print(json.dumps({"screen_passed": passed, "gates": gates}, indent=2), flush=True)
    print("Share compact-report.json; keep the runtime open. No upload or deployment.", flush=True)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("stage", choices=("prepare", "export", "reference", "candidate", "report"))
    parser.add_argument("--work", type=Path, required=True)
    args = parser.parse_args()
    args.work.mkdir(parents=True, exist_ok=True)
    if args.stage == "prepare":
        asyncio.run(prepare(args.work))
    elif args.stage == "export":
        export(args.work)
    elif args.stage == "report":
        summarize(args.work)
    else:
        measure(args.work, compact=args.stage == "candidate")


if __name__ == "__main__":
    main()
