"""Upload this single file to the existing compact-check Colab runtime.

Run: python diagnose_compact_runtime.py --work <the existing WORK directory>
No credentials, network, passage rebuilds, or deployment. Existing files stay intact.
"""

from __future__ import annotations

import argparse
import gc
import hashlib
import json
import os
from pathlib import Path
import shutil
import subprocess
import sys
import tempfile
from time import perf_counter
from unittest.mock import patch


def memory() -> dict[str, float]:
    values = {}
    for line in Path("/proc/self/status").read_text().splitlines():
        if line.startswith(("VmRSS:", "VmHWM:")):
            values[line.split(":")[0]] = int(line.split()[1]) / 1024
    if set(values) != {"VmRSS", "VmHWM"}:
        raise RuntimeError("This diagnostic requires the existing Linux Colab runtime")
    return {"rss_mib": values["VmRSS"], "peak_mib": values["VmHWM"]}


def write_json(path: Path, value) -> None:
    path.write_text(json.dumps(value, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")


def quantize_matrices_only(work: Path, output: Path) -> None:
    import onnxruntime as ort
    from onnxruntime.quantization import QuantType, quantize_dynamic
    from subarag.retrieval.onnx_embedding import file_sha256

    ort.disable_telemetry_events()
    source = work / "e5-fp32.onnx"
    destination = output / "matrix-only-model"
    destination.mkdir()
    original = json.loads((work / "compact-model/embedding-manifest.json").read_text())
    tokenizer = work / "compact-model/tokenizer.json"
    if file_sha256(tokenizer) != original["files"]["tokenizer.json"]["sha256"]:
        raise ValueError("Existing tokenizer has changed; diagnostic stopped")
    print("Preparing matrix-only comparison from the existing FP32 graph…", flush=True)
    quantize_dynamic(
        str(source), str(destination / "model.onnx"),
        op_types_to_quantize=["MatMul"], weight_type=QuantType.QUInt8,
        per_channel=False, use_external_data_format=False,
        extra_options={"MatMulConstBOnly": True},
    )
    shutil.copy2(tokenizer, destination / "tokenizer.json")
    original.update(
        quantization="uint8-matmul-only-fp32-embedding-table-diagnostic",
        fp32_graph_sha256=file_sha256(source),
        files={name: {"bytes": (destination / name).stat().st_size,
                      "sha256": file_sha256(destination / name)}
               for name in ("model.onnx", "tokenizer.json")},
    )
    write_json(destination / "embedding-manifest.json", original)


def run_measurement(work: Path, output: Path, mode: str) -> None:
    stages = []
    started = perf_counter()

    def mark(stage: str) -> None:
        item = {"stage": stage, "elapsed_ms": (perf_counter() - started) * 1000, **memory()}
        stages.append(item)
        print(f"{mode}: {stage}: RSS {item['rss_mib']:.1f} MiB; peak {item['peak_mib']:.1f} MiB", flush=True)

    mark("python_started")
    import numpy  # noqa: F401 - import cost is part of this measurement
    mark("numpy_imported")
    import faiss  # noqa: F401
    mark("faiss_imported")
    import onnxruntime as ort
    ort.disable_telemetry_events()
    mark("onnxruntime_imported")
    import tokenizers
    mark("tokenizers_imported")
    from subarag.experiments import compact_embedding as experiment
    from subarag import retrieval
    from subarag.retrieval.dense import HnswDenseIndex
    from subarag.api.app import create_app  # noqa: F401
    mark("api_and_retrieval_imported")

    model_dir = output / "matrix-only-model" if mode == "matrix-only" else work / "compact-model"
    metadata = json.loads((model_dir / "embedding-manifest.json").read_text())
    for name, item in metadata["files"].items():
        if name not in ("model.onnx", "tokenizer.json"):
            raise ValueError("Unexpected embedding file")
        if experiment.file_sha256(model_dir / name) != item["sha256"]:
            raise ValueError("Embedding artifact changed; diagnostic stopped")
    mark("embedding_checksums_verified")
    real_session = ort.InferenceSession
    real_tokenizer = tokenizers.Tokenizer
    real_embedder = experiment.OnnxQueryEmbedder
    real_dense_load = HnswDenseIndex.load
    real_artifact_load = retrieval.load_hnsw_artifact
    cached_tokenizer = None
    if mode == "tokenizer-first":
        cached_tokenizer = real_tokenizer.from_file(str(model_dir / "tokenizer.json"))
        mark("tokenizer_loaded_before_model")

    def traced_session(*args, **kwargs):
        mark("before_model_session")
        session = real_session(*args, **kwargs)
        mark("model_session_loaded")
        return session

    class TracedTokenizer:
        @staticmethod
        def from_file(path):
            value = cached_tokenizer if cached_tokenizer is not None else real_tokenizer.from_file(path)
            mark("tokenizer_ready")
            return value

    def traced_dense(documents, path, **kwargs):
        language = Path(path).parent.name
        mark(f"{language}_documents_loaded_before_faiss")
        value = real_dense_load(documents, path, **kwargs)
        mark(f"{language}_faiss_loaded")
        return value

    def traced_artifact(*args, **kwargs):
        value = real_artifact_load(*args, **kwargs)
        mark(f"{value.language}_artifact_ready")
        return value

    class TracedEmbedder:
        def __init__(self, _directory, *, expected_revision):
            self.inner = real_embedder(model_dir, expected_revision=expected_revision)
            self.short_seen = False
            self.long_seen = False

        def embed_queries(self, texts):
            long = any(len(text) > 2000 for text in texts)
            first_long = long and not self.long_seen
            if first_long:
                mark("before_first_512_token_probe")
            result = self.inner.embed_queries(texts)
            if not self.short_seen:
                mark("first_short_query_complete")
                self.short_seen = True
            if long:
                mark("long_query_complete")
                self.long_seen = True
            return result

    measured = {}
    # Reuse the exact frozen 192-query measurement without overwriting its
    # original candidate.json/reference.json. Only our new output folder changes.
    with patch.object(ort, "InferenceSession", traced_session), \
            patch.object(tokenizers, "Tokenizer", TracedTokenizer), \
            patch.object(HnswDenseIndex, "load", traced_dense), \
            patch.object(retrieval, "load_hnsw_artifact", traced_artifact), \
            patch.object(experiment, "OnnxQueryEmbedder", TracedEmbedder), \
            patch.object(experiment, "write_json", lambda _path, value: measured.update(value)):
        experiment.measure(work, compact=True)
        mark("measurement_complete_objects_released")
    gc.collect()
    mark("after_gc")
    if any(name in sys.modules for name in ("torch", "transformers", "sentence_transformers")):
        raise RuntimeError("Heavy runtime imported; diagnostic measurement invalid")
    measured.pop("vectors", None)
    measured.pop("rankings", None)
    measured["runtime"] = mode
    measured["load_order"] = "tokenizer before model" if mode == "tokenizer-first" else "model before tokenizer"
    measured["stage_memory"] = stages
    measured["model_bytes"] = metadata["files"]["model.onnx"]["bytes"]
    measured["embedding_manifest_sha256"] = experiment.file_sha256(model_dir / "embedding-manifest.json")
    write_json(output / f"{mode}.json", measured)


def run(work: Path) -> None:
    required = ("inputs.json", "reference.json", "candidate.json", "compact-report.json",
                "e5-fp32.onnx", "compact-model/embedding-manifest.json")
    for name in required:
        if not (work / name).is_file():
            raise ValueError(f"Missing {name}. Run this in the same Colab runtime with its original WORK directory.")
    output = Path(tempfile.mkdtemp(prefix="memory-diagnostic-", dir=work))
    print("Diagnostic output:", output, flush=True)
    env = dict(os.environ, OMP_NUM_THREADS="1", OPENBLAS_NUM_THREADS="1", TOKENIZERS_PARALLELISM="false",
               HF_HUB_OFFLINE="1", TRANSFORMERS_OFFLINE="1")
    env.pop("HF_TOKEN", None)
    modes = ("current", "tokenizer-first", "quantize", "matrix-only")
    for mode in modes:
        subprocess.run([sys.executable, "-u", str(Path(__file__).resolve()), "--work", str(work),
                        "--output", str(output), "--child", mode], env=env, check=True)
    reference = json.loads((work / "reference.json").read_text())
    variants = {}
    for mode in ("current", "tokenizer-first", "matrix-only"):
        value = json.loads((output / f"{mode}.json").read_text())
        if value["inputs_sha256"] != reference["inputs_sha256"]:
            raise ValueError("Frozen input fingerprints differ; cannot compare quality")
        value["relative_ndcg_change_percent"] = {
            lang: 100 * (value["metrics"][lang]["ndcg@8"] / reference["metrics"][lang]["ndcg@8"] - 1)
            if reference["metrics"][lang]["ndcg@8"] else None for lang in ("en", "hi", "mr")}
        value["within_original_quality_gate"] = {
            lang: change is not None and change >= -2 for lang, change in value["relative_ndcg_change_percent"].items()}
        variants[mode] = value
    result = {
        "schema_version": 1, "diagnostic_only": True, "variants": variants,
        "diagnostic_script_sha256": hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
        "model_revision": reference["model_revision"], "inputs_sha256": reference["inputs_sha256"],
        "original_report": json.loads((work / "compact-report.json").read_text()),
        "limitation": "Fresh-process snapshots and high-water marks are not additive component sizes or a container guarantee. Matrix-only isolates embedding-table quantization; it is not a selected serving model.",
    }
    report = output / "memory-diagnostic.json"
    write_json(report, result)
    print("REPORT:", report, flush=True)
    print("Share memory-diagnostic.json. Original files unchanged; no upload/deployment.", flush=True)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--work", type=Path, required=True)
    parser.add_argument("--output", type=Path)
    parser.add_argument("--child", choices=("current", "tokenizer-first", "quantize", "matrix-only"))
    args = parser.parse_args()
    if args.child:
        if args.output is None:
            parser.error("Child stages require --output")
        if args.child == "quantize":
            quantize_matrices_only(args.work, args.output)
        else:
            run_measurement(args.work, args.output, args.child)
    else:
        run(args.work.resolve())
