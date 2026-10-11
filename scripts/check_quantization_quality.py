"""Colab-only quality screen: per-channel matrices or full-precision matrices.

Keeps the original uint8 embedding tables, float64 tokenizer, disk documents,
one index, and requested-size allocator. No serving changes or cloud uploads.
"""

from __future__ import annotations

import argparse
from collections import Counter
import hashlib
import importlib.util
import json
import os
from pathlib import Path
import shutil
import subprocess
import sys
import tempfile
from unittest.mock import patch

MEMORY_SHA = "cf88c2b249e519c1023c04b16243d9bbcf2d68198607b80a4798017449738389"
MODES = ("control", "per-channel", "fp32-matrices")


def read(path):
    return json.loads(Path(path).read_text())


def write(path, value):
    Path(path).write_text(json.dumps(value, indent=2) + "\n")


def digest(path):
    result = hashlib.sha256()
    with Path(path).open("rb") as handle:
        while chunk := handle.read(1024 * 1024):
            result.update(chunk)
    return result.hexdigest()


def helper():
    for path in Path(__file__).resolve().parent.glob("check_inference_memory*.py"):
        if digest(path) == MEMORY_SHA:
            spec = importlib.util.spec_from_file_location("verified_memory_helper", path)
            module = importlib.util.module_from_spec(spec)
            spec.loader.exec_module(module)
            return module
    raise ValueError("Keep check_inference_memory.py and earlier helper uploads beside this script.")


def audit_graph(path):
    import onnx
    graph = onnx.load(str(path)).graph
    initializers = {item.name: item for item in graph.initializer}
    gathers, matrix_channels, float_weights = [], [], {}
    for node in graph.node:
        if node.op_type == "MatMul" and node.input[1] in initializers:
            tensor = initializers[node.input[1]]
            # Quantization may explicitly set protobuf data_location=DEFAULT.
            # Compare exact tensor bytes/type/shape, not optional storage flags.
            array = onnx.numpy_helper.to_array(tensor)
            float_weights[tensor.name] = {"shape": list(tensor.dims), "data_type": tensor.data_type,
                                          "sha256": hashlib.sha256(array.tobytes()).hexdigest()}
        if node.op_type not in ("Gather", "MatMulInteger"):
            continue
        name = node.input[0] if node.op_type == "Gather" else node.input[1]
        if name not in initializers or initializers[name].data_type != onnx.TensorProto.UINT8:
            continue
        if not name.endswith("_quantized"):
            raise ValueError("Unexpected quantized initializer naming")
        base = name.removesuffix("_quantized")
        scale = initializers[base + "_scale"]
        if node.op_type == "Gather":
            tensors = [initializers[name], scale, initializers[base + "_zero_point"]]
            gathers.append({"node_sha256": hashlib.sha256(node.SerializeToString()).hexdigest(),
                            "weight": name,
                            "tensors_sha256": [hashlib.sha256(t.SerializeToString()).hexdigest() for t in tensors]})
        else:
            matrix_channels.append({"weight": name, "scale_shape": list(scale.dims)})
    return {"operators": dict(Counter(node.op_type for node in graph.node)),
            "gathers": sorted(gathers, key=lambda item: (item["weight"], item["node_sha256"])),
            "matrix_scales": matrix_channels, "float_matrix_weights": float_weights}


def prepare(work, output, mode):
    from onnxruntime.quantization import QuantType, quantize_dynamic
    destination = output / mode / "model"
    destination.mkdir()
    source = work / "e5-fp32.onnx"
    print(f"Preparing {mode} from the saved FP32 graph (no download/export)…", flush=True)
    quantize_dynamic(
        str(source), str(destination / "model.onnx"),
        op_types_to_quantize=["MatMul", "Gather"] if mode == "per-channel" else ["Gather"],
        weight_type=QuantType.QUInt8, per_channel=mode == "per-channel", reduce_range=False,
        use_external_data_format=False, extra_options={"MatMulConstBOnly": True},
    )
    baseline = audit_graph(work / "compact-model/model.onnx")
    candidate = audit_graph(destination / "model.onnx")
    if not baseline["gathers"] or candidate["gathers"] != baseline["gathers"]:
        raise ValueError("Quantized embedding tables or their Gather nodes changed")
    if mode == "per-channel":
        if (not candidate["matrix_scales"]
                or len(candidate["matrix_scales"]) != len(baseline["matrix_scales"])
                or not all(len(item["scale_shape"]) == 1 and item["scale_shape"][0] > 1
                           for item in candidate["matrix_scales"])):
            raise ValueError("Per-channel matrix conversion did not match the expected graph")
    else:
        fp32 = audit_graph(source)
        if (candidate["operators"].get("MatMulInteger", 0) or candidate["operators"].get("DynamicQuantizeLinear", 0)
                or not fp32["float_matrix_weights"] or candidate["float_matrix_weights"] != fp32["float_matrix_weights"]):
            raise ValueError("Full-precision matrix weights were not preserved")
    shutil.copy2(work / "compact-model/tokenizer.json", destination / "tokenizer.json")
    metadata = read(work / "compact-model/embedding-manifest.json")
    metadata.update(quantization=mode + "-uint8-gather-diagnostic", fp32_graph_sha256=digest(source),
                    files={name: {"bytes": (destination / name).stat().st_size,
                                  "sha256": digest(destination / name)}
                           for name in ("model.onnx", "tokenizer.json")})
    write(destination / "embedding-manifest.json", metadata)
    write(output / mode / "graph-audit.json", {"embedding_tables_and_gathers_unchanged": True,
                                               "baseline": baseline, "candidate": candidate})


def measure(work, output, mode):
    memory = helper()
    disk = memory.helper()
    import onnxruntime as ort
    destination = work / "compact-model" if mode == "control" else output / mode / "model"
    metadata = read(destination / "embedding-manifest.json")
    baseline_metadata = read(work / "compact-model/embedding-manifest.json")
    for field in ("schema_version", "model_name", "model_revision", "dimension", "max_length",
                  "query_prefix", "pooling"):
        if metadata[field] != baseline_metadata[field]:
            raise ValueError("Candidate changed the embedding contract")
    if set(metadata["files"]) != {"model.onnx", "tokenizer.json"}:
        raise ValueError("Unexpected candidate files")
    for name, item in metadata["files"].items():
        if (destination / name).stat().st_size != item["bytes"] or digest(destination / name) != item["sha256"]:
            raise ValueError("Candidate checksum mismatch")
    if metadata["files"]["tokenizer.json"] != baseline_metadata["files"]["tokenizer.json"]:
        raise ValueError("Candidate tokenizer changed")
    actual_session = ort.InferenceSession
    redirected = []

    def redirect(path, **kwargs):
        if Path(path).resolve() != (work / "compact-model/model.onnx").resolve() or redirected:
            raise ValueError("Unexpected session request")
        redirected.append(str(destination / "model.onnx"))
        return actual_session(str(destination / "model.onnx"), **kwargs)

    result, searches = {}, []
    actual_search = disk.DiskIndex.search

    def search(index, vector, k):
        hits = actual_search(index, vector, k)
        searches.append({"language": index.language, "ids": [hit.document.id for hit in hits]})
        return hits
    def capture(path, value):
        if Path(path) != output / mode / "requested-arena.json" or result:
            raise ValueError("Unexpected experiment output")
        result.update(value)

    # The verified adapter retains its tokenizer, pooling and shape checks. Only
    # the model opened by ORT changes, after validating its own manifest above.
    # Candidate export/audit imports never enter this measurement process.
    with patch.object(ort, "InferenceSession", redirect), patch.object(memory, "write", capture), \
            patch.object(memory, "helper", lambda: disk), patch.object(disk.DiskIndex, "search", search):
        memory.child(work, output / mode, "requested-arena")
    if len(redirected) != 1 or not result:
        raise ValueError("Incomplete candidate measurement")
    count = len(read(work / "inputs.json")["cases"])
    if len(searches) != count + 9:
        raise ValueError("Unexpected retrieval schedule")
    first = {lang: next(row for row in searches[:count] if row["language"] == lang) for lang in disk.LANGUAGES}
    stable = all(row == first[row["language"]] for row in searches[count:])
    value = result["measurement"]
    value["mixed_language_rankings_identical_to_original_uint8"] = value.pop("mixed_language_rankings_identical")
    result.update(mode=mode, actual_model=redirected[0], model_manifest=metadata,
                  candidate_rankings_stable_after_language_switch=stable)
    write(output / mode / "measurement.json", result)


def summarize(item, original, reference):
    value = item["measurement"]
    if value["inputs_sha256"] != original["inputs_sha256"] or value["model_revision"] != original["model_revision"]:
        raise ValueError("Comparison provenance differs")
    item["identical_frozen_vectors_to_original_uint8"] = value.pop("vectors") == original["vectors"]
    item["identical_frozen_rankings_to_original_uint8"] = value.pop("rankings") == original["rankings"]
    # A changed quantizer is expected to change outputs. Gate against the FP32
    # retrieval reference, not exact equality with the lower-quality uint8 model.
    quality = {lang: reference["metrics"][lang]["ndcg@8"] > 0 and
               value["metrics"][lang]["ndcg@8"] >= .98 * reference["metrics"][lang]["ndcg@8"]
               for lang in ("en", "hi", "mr")}
    gates = {"within_400_mib": value["peak_rss_mib"] <= 400,
             "grouped_p95_within_500_ms": all(v["p95"] <= 500 for v in value["retrieval_ms"].values()),
             "switching_p95_within_500_ms": value["mixed_language_ms"]["p95"] <= 500,
             "one_resident_index": value["max_resident_index_count"] == 1,
             "no_heavyweight_runtime": not value["heavyweight_runtime_imported"],
             "candidate_switch_rankings_stable": item["candidate_rankings_stable_after_language_switch"]}
    item.update(gates=gates, quality_gate_by_language=quality,
                relative_ndcg_change={lang: value["metrics"][lang]["ndcg@8"] / reference["metrics"][lang]["ndcg@8"] - 1
                                      for lang in quality if reference["metrics"][lang]["ndcg@8"] > 0},
                screen_passed=all(gates.values()) and all(quality.values()))
    return item


def run(work):
    helper()
    reports = [p for p in work.glob("inference-memory-*/inference-memory-report.json")
               if read(p).get("script_sha256") == MEMORY_SHA]
    if not reports:
        raise ValueError("Keep the completed allocator screen in WORK; stop if the runtime has reset.")
    prior = max(reports, key=lambda p: p.stat().st_mtime)
    previous = read(prior)
    selected = previous["variants"]["requested-arena"]
    if not previous["original_files_unchanged"] or selected["status"] != "completed" or not all(selected["gates"].values()):
        raise ValueError("The requested-size allocator has not passed its memory/parity screen")
    tracked = {Path(name): checksum for name, checksum in previous["source_sha256"].items()}
    tracked[prior] = digest(prior)
    tracked[work / "e5-fp32.onnx"] = digest(work / "e5-fp32.onnx")
    disk_reports = [p for p in tracked if p.name == "disk-retrieval-report.json"]
    if len(disk_reports) != 1 or any(not p.resolve().is_relative_to(work) or digest(p) != checksum for p, checksum in tracked.items()):
        raise ValueError("Input provenance changed")
    output = Path(tempfile.mkdtemp(prefix="quantization-quality-", dir=work))
    print("Output:", output, flush=True)
    env = dict(os.environ, HF_HUB_OFFLINE="1", TRANSFORMERS_OFFLINE="1", TOKENIZERS_PARALLELISM="false",
               OMP_NUM_THREADS="1", OPENBLAS_NUM_THREADS="1")
    env.pop("HF_TOKEN", None)
    variants = {}
    original, reference = read(work / "candidate.json"), read(work / "reference.json")
    for mode in MODES:
        (output / mode).mkdir()
        write(output / mode / "screen-inputs.json", {"disk_report": str(disk_reports[0])})
        for stage in (("measure",) if mode == "control" else ("prepare", "measure")):
            process = subprocess.run([sys.executable, "-u", str(Path(__file__).resolve()), "--work", str(work),
                                      "--output", str(output), "--mode", mode, "--child", stage], env=env)
            if process.returncode:
                variants[mode] = {"status": "failed", "stage": stage, "exit_code": process.returncode}
                if mode == "control":
                    raise RuntimeError("Control failed; keep the files and share the actual error above.")
                break
        else:
            item = summarize(read(output / mode / "measurement.json"), original, reference)
            if mode == "control" and not (item["identical_frozen_vectors_to_original_uint8"] and item["identical_frozen_rankings_to_original_uint8"]):
                raise ValueError("Fresh control no longer reproduces the original uint8 results")
            if mode != "control":
                item["graph_audit"] = read(output / mode / "graph-audit.json")
            variants[mode] = item
    if any(digest(p) != checksum for p, checksum in tracked.items()):
        raise ValueError("An original file changed during this screen")
    write(output / "quantization-quality-report.json", {
        "schema_version": 1, "diagnostic_only": True, "script_sha256": digest(__file__),
        "memory_helper_sha256": MEMORY_SHA, "original_files_unchanged": True,
        "source_sha256": {str(p): checksum for p, checksum in tracked.items()}, "variants": variants,
        "limitation": "192 reused development queries, not held-out proof or answer evaluation. Any candidate passing this screen still needs independent quality and memory-limited serving validation. Model conversion memory is isolated from inference. Full-precision matrix variant may exceed the memory gate. No index rebuild, upload, deployment, or automatic selection.",
    })
    print("REPORT:", output / "quantization-quality-report.json", flush=True)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--work", type=Path, required=True)
    parser.add_argument("--output", type=Path)
    parser.add_argument("--mode", choices=MODES)
    parser.add_argument("--child", choices=("prepare", "measure"))
    args = parser.parse_args()
    if args.child:
        if args.output is None or args.mode is None:
            parser.error("Child mode requires --output and --mode")
        (prepare if args.child == "prepare" else measure)(args.work, args.output, args.mode)
    else:
        run(args.work.resolve())
