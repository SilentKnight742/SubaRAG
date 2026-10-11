"""Colab-only isolation of the three E5 embedding tables after preprocessing.

Restore one table at a time from the verified FP32 graph. Keep every unrelated
node and initializer byte-identical; no re-quantization, downloads, or deployment.
"""

from __future__ import annotations

import argparse
from copy import deepcopy
import hashlib
import importlib.util
import json
import os
from pathlib import Path
import shutil
import subprocess
import sys
import tempfile

PREPROCESS_SHA = "92ad1be0da7e11a32d3848cf447a2458003e310794c5f6846cebc5dcdf0fef22"
MODES = ("preprocessed-control", "fp32-position", "fp32-token-type", "fp32-word")
ROLES = {"fp32-position": "position_embeddings", "fp32-token-type": "token_type_embeddings",
         "fp32-word": "word_embeddings"}


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
    for path in Path(__file__).resolve().parent.glob("check_onnx_preprocessing*.py"):
        if digest(path) == PREPROCESS_SHA:
            spec = importlib.util.spec_from_file_location("verified_preprocessing_helper", path)
            module = importlib.util.module_from_spec(spec)
            spec.loader.exec_module(module)
            return module
    raise ValueError("Keep the updated preprocessing helper and its dependencies in /content.")


def fingerprint(value):
    return hashlib.sha256(value.SerializeToString()).hexdigest()


def restore_table(model, fp32, role):
    """Strictly replace one uint8 Gather + DequantizeLinear with its FP32 Gather."""
    import onnx
    weights = {item.name: item for item in fp32.graph.initializer}
    quantized = {item.name: item for item in model.graph.initializer}
    tables = [node for node in fp32.graph.node if node.op_type == "Gather"
              and node.input[0] in weights and weights[node.input[0]].data_type == onnx.TensorProto.FLOAT
              and len(weights[node.input[0]].dims) == 2 and weights[node.input[0]].dims[1] == 384]
    matches = {tag: [node for node in tables if tag in node.input[0]] for tag in ROLES.values()}
    if len(tables) != 3 or any(len(nodes) != 1 for nodes in matches.values()):
        raise ValueError("Expected exactly the word, position and token-type embedding tables; will not guess")
    original = matches[role][0]
    name = original.input[0]
    qname, scale, zero = name + "_quantized", name + "_scale", name + "_zero_point"
    if (not all(key in quantized for key in (qname, scale, zero))
            or quantized[qname].data_type != onnx.TensorProto.UINT8
            or list(quantized[qname].dims) != list(weights[name].dims)):
        raise ValueError("Unexpected quantized table encoding")
    nodes = list(model.graph.node)
    selected = [i for i, node in enumerate(nodes) if node.op_type == "Gather" and node.input[0] == qname]
    if len(selected) != 1:
        raise ValueError("Quantized table must have exactly one Gather")
    index = selected[0]
    gather = nodes[index]
    expected = deepcopy(original)
    expected.input[0], expected.output[0] = qname, original.output[0] + "_quantized"
    if fingerprint(gather) != fingerprint(expected):
        raise ValueError("Gather changed beyond the known quantization rewrite")
    consumers = [i for i, node in enumerate(nodes) if gather.output[0] in node.input]
    if len(consumers) != 1:
        raise ValueError("Quantized Gather output must feed exactly one dequantizer")
    dq_index = consumers[0]
    dq = nodes[dq_index]
    if (dq.op_type != "DequantizeLinear" or list(dq.input) != [gather.output[0], scale, zero]
            or list(dq.output) != list(original.output) or dq.attribute):
        raise ValueError("Unexpected Gather dequantization boundary")
    removed = {qname, scale, zero}
    untouched_nodes = [fingerprint(node) for i, node in enumerate(nodes) if i not in (index, dq_index)]
    untouched_weights = {key: fingerprint(tensor) for key, tensor in quantized.items() if key not in removed}
    replacement = [deepcopy(original) if i == index else node for i, node in enumerate(nodes) if i != dq_index]
    if any(key in node.input for node in replacement for key in removed):
        raise ValueError("A table initializer is shared with another operation")
    del model.graph.node[:]
    model.graph.node.extend(replacement)
    retained = [tensor for tensor in model.graph.initializer if tensor.name not in removed]
    del model.graph.initializer[:]
    model.graph.initializer.extend(retained)
    model.graph.initializer.append(weights[name])
    infos = [item for item in model.graph.value_info if item.name != gather.output[0]]
    del model.graph.value_info[:]
    model.graph.value_info.extend(infos)
    if ([fingerprint(node) for node in model.graph.node if node.name != original.name] != untouched_nodes
            or {t.name: fingerprint(t) for t in model.graph.initializer if t.name != name} != untouched_weights
            or fingerprint(next(t for t in model.graph.initializer if t.name == name)) != fingerprint(weights[name])):
        raise ValueError("An unrelated node/weight changed or FP32 table bytes differ")
    onnx.checker.check_model(model)
    return {"role": role, "weight": name, "shape": list(weights[name].dims),
            "fp32_weight_sha256": fingerprint(weights[name]),
            "removed_initializer_names": sorted(removed), "removed_dequantizer": dq.name,
            "unrelated_nodes_and_initializers_identical": True, "restored_weight_exact": True}


def prepare(work, output, mode):
    selected = read(output / "screen-inputs.json")
    prior = Path(selected["preprocessing_report"]).parent
    source = prior / "preprocessed-uint8/model"
    destination = output / mode / "model"
    destination.mkdir()
    if mode == "preprocessed-control":
        for name in ("model.onnx", "tokenizer.json", "embedding-manifest.json"):
            shutil.copy2(source / name, destination / name)
        audit = {"model_byte_identical": digest(destination / "model.onnx") == digest(source / "model.onnx")}
    else:
        import onnx
        fp32 = onnx.load(str(prior / "fp32-preprocessed/model/model.onnx"))
        model = onnx.load(str(source / "model.onnx"))
        audit = restore_table(model, fp32, ROLES[mode])
        onnx.save(model, str(destination / "model.onnx"))
        # Check serialization too, before the separate measured process loads it.
        onnx.checker.check_model(str(destination / "model.onnx"))
        shutil.copy2(source / "tokenizer.json", destination / "tokenizer.json")
        metadata = read(source / "embedding-manifest.json")
        metadata.update(quantization=mode + "-isolation-diagnostic", source_graph_sha256=digest(source / "model.onnx"),
                        files={name: {"bytes": (destination / name).stat().st_size,
                                      "sha256": digest(destination / name)}
                               for name in ("model.onnx", "tokenizer.json")})
        write(destination / "embedding-manifest.json", metadata)
    audit.update(model_sha256=digest(destination / "model.onnx"), model_bytes=(destination / "model.onnx").stat().st_size)
    write(output / mode / "graph-audit.json", audit)


def sources(work):
    reports = [p for p in work.glob("onnx-preprocessing-*/onnx-preprocessing-report.json")
               if read(p).get("script_sha256") == PREPROCESS_SHA and read(p).get("status") == "completed"]
    if not reports:
        raise ValueError("Keep the completed shape-merge run in WORK; stop if Colab reset")
    prior = max(reports, key=lambda p: p.stat().st_mtime)
    report = read(prior)
    if not (report["original_files_unchanged"] and report["restored_control_passed"] and report["fp32_parity"]["passed"]
            and all(report["variants"]["preprocessed-uint8"]["gates"].values())):
        raise ValueError("Preprocessing control/parity/runtime checks have not passed")
    tracked = {Path(name): checksum for name, checksum in report["source_sha256"].items()}
    tracked[prior] = digest(prior)
    for mode in ("fp32-preprocessed", "preprocessed-uint8"):
        root = prior.parent / mode / "model"
        metadata = read(root / "embedding-manifest.json")
        tracked[root / "embedding-manifest.json"] = digest(root / "embedding-manifest.json")
        if metadata != report["variants"][mode]["model_manifest"]:
            raise ValueError("Saved model manifest differs from the completed report")
        if set(metadata["files"]) != {"model.onnx", "tokenizer.json"}:
            raise ValueError("Unexpected model files")
        for name, item in metadata["files"].items():
            tracked[root / name] = item["sha256"]
            if (root / name).stat().st_size != item["bytes"]:
                raise ValueError("Model file size changed")
    baseline = prior.parent / "preprocessed-uint8/measurement.json"
    tracked[baseline] = digest(baseline)
    disk = [p for p in tracked if p.name == "disk-retrieval-report.json"]
    if len(disk) != 1 or any(not p.resolve().is_relative_to(work) or digest(p) != h for p, h in tracked.items()):
        raise ValueError("Experiment provenance changed")
    return prior, disk[0], tracked, read(baseline)


def run(work):
    preprocessing = helper()
    quality = preprocessing.helper()
    quality.helper().helper().tokenizer_helper().load_helper()
    prior, disk_report, tracked, baseline = sources(work)
    output = Path(tempfile.mkdtemp(prefix="embedding-tables-", dir=work))
    write(output / "screen-inputs.json", {"preprocessing_report": str(prior)})
    print("Output:", output, flush=True)
    report = {"schema_version": 1, "diagnostic_only": True, "script_sha256": digest(__file__),
              "preprocessing_helper_sha256": PREPROCESS_SHA, "status": "running", "variants": {},
              "source_sha256": {str(p): h for p, h in tracked.items()},
              "limitation": "One-table interventions, not additive attribution: errors interact with downstream quantized matrices. Reused development queries, not held-out proof. FP32 word-table variant may exceed memory limits and is diagnostic only. Sequential Colab RSS excludes serving concurrency/container accounting. No download, index rebuild, re-quantization, automatic selection, upload, or deployment."}
    report_path = output / "embedding-table-report.json"
    env = dict(os.environ, HF_HUB_OFFLINE="1", TRANSFORMERS_OFFLINE="1", TOKENIZERS_PARALLELISM="false",
               OMP_NUM_THREADS="1", OPENBLAS_NUM_THREADS="1")
    env.pop("HF_TOKEN", None)
    original, reference = read(work / "candidate.json"), read(work / "reference.json")
    try:
        for mode in MODES:
            (output / mode).mkdir()
            write(output / mode / "screen-inputs.json", {"disk_report": str(disk_report)})
            for stage in ("prepare", "measure"):
                print(f"Starting {mode}: {stage}", flush=True)
                result = subprocess.run([sys.executable, "-u", str(Path(__file__).resolve()), "--work", str(work),
                                         "--output", str(output), "--mode", mode, "--child", stage], env=env)
                if result.returncode:
                    report["variants"][mode] = {"status": "failed", "stage": stage, "exit_code": result.returncode}
                    raise RuntimeError(f"{mode} {stage} failed; share the progress/error lines above")
            item = read(output / mode / "measurement.json")
            value, old = item["measurement"], baseline["measurement"]
            item["identical_vectors_to_preprocessed_control"] = value["vectors"] == old["vectors"]
            item["identical_rankings_to_preprocessed_control"] = value["rankings"] == old["rankings"]
            signature = lambda data: [(r["tokens"], r["shape"], r["dtype"], r["output_sha256"]) for r in data["trace"]]
            item["identical_all_outputs_to_preprocessed_control"] = signature(item) == signature(baseline)
            item["ndcg_change_from_preprocessed_control"] = {
                lang: value["metrics"][lang]["ndcg@8"] - old["metrics"][lang]["ndcg@8"] for lang in ("en", "hi", "mr")}
            quality.summarize(item, original, reference)
            item["graph_audit"] = read(output / mode / "graph-audit.json")
            report["variants"][mode] = item
            if mode == "preprocessed-control":
                report["fresh_control_passed"] = bool(item["identical_vectors_to_preprocessed_control"]
                    and item["identical_rankings_to_preprocessed_control"]
                    and item["identical_all_outputs_to_preprocessed_control"] and all(item["gates"].values()))
                if not report["fresh_control_passed"]:
                    raise ValueError("Fresh control changed; stop before table interventions")
            write(report_path, report)
        report["status"] = "completed"
    except Exception as error:
        report.update(status="stopped", error=str(error))
        raise
    finally:
        report["original_files_unchanged"] = all(p.is_file() and digest(p) == h for p, h in tracked.items())
        if not report["original_files_unchanged"]:
            report.update(status="stopped", error="An original artifact changed")
        write(report_path, report)
        print("REPORT:", report_path, flush=True)
    if not report["original_files_unchanged"]:
        raise ValueError("An original artifact changed")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--work", type=Path, required=True)
    parser.add_argument("--output", type=Path)
    parser.add_argument("--mode", choices=MODES)
    parser.add_argument("--child", choices=("prepare", "measure"))
    args = parser.parse_args()
    if args.child:
        if args.output is None or args.mode is None:
            parser.error("Child stage requires --output and --mode")
        if args.child == "prepare":
            prepare(args.work, args.output, args.mode)
        else:
            helper().helper().measure(args.work, args.output, args.mode)
    else:
        run(args.work.resolve())
