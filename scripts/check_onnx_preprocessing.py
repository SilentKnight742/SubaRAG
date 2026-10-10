"""Offline Colab experiment: restored control, FP32 preprocessing, then uint8.

Uses the restored disk-retrieval checkpoint directly. Never rebuilds indexes,
downloads models, edits original artifacts, or promotes a serving candidate.
"""

from __future__ import annotations

import argparse
from collections import Counter
import hashlib
import importlib.util
import json
import logging
import os
from pathlib import Path
import shutil
import subprocess
import sys
import tempfile
from unittest.mock import patch

QUALITY_SHA = "5c4df33a42d92fcd1aa796506c88c5de5b71b6769047051d45994dde74d34775"
DISK_SHA = "5ebc4544ae6b4ad5b692c00ce4b6c0f15901f14c395899c6cb7c39f7f0b6d63d"
MODES = ("control", "fp32-original", "fp32-preprocessed", "preprocessed-uint8")


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
    for path in Path(__file__).resolve().parent.glob("check_quantization_quality*.py"):
        if digest(path) == QUALITY_SHA:
            spec = importlib.util.spec_from_file_location("verified_quality_helper", path)
            module = importlib.util.module_from_spec(spec)
            spec.loader.exec_module(module)
            return module
    raise ValueError("Upload check_quantization_quality.py and check_inference_memory.py beside this helper.")


def graph_summary(path):
    import onnx
    model = onnx.load(str(path))
    onnx.checker.check_model(model)
    tensors = {tensor.name: tensor for tensor in model.graph.initializer}
    return {
        "sha256": digest(path), "bytes": path.stat().st_size, "onnx_version": onnx.__version__,
        "operators": dict(Counter(node.op_type for node in model.graph.node)),
        "value_info_count": len(model.graph.value_info),
        "preprocessing_marker": any(item.key == "onnx.quant.pre_process" and item.value == "onnxruntime.quant"
                                    for item in model.metadata_props),
        "uint8_gathers": sum(node.op_type == "Gather" and node.input[0] in tensors
                             and tensors[node.input[0]].data_type == onnx.TensorProto.UINT8
                             for node in model.graph.node),
    }


def prepare(work, output, mode):
    import onnxruntime as ort
    ort.disable_telemetry_events()
    destination = output / mode / "model"
    destination.mkdir()
    model = destination / "model.onnx"
    source = work / "e5-fp32.onnx"
    if mode == "fp32-original":
        shutil.copy2(source, model)
    elif mode == "fp32-preprocessed":
        from onnxruntime.quantization.shape_inference import quant_pre_process
        # Actually perform all three stages. Never add the marker ourselves or
        # silently retry with a stage disabled just to remove the warning.
        errors = []

        class Errors(logging.Filter):
            def filter(self, record):
                if record.levelno >= logging.ERROR:
                    errors.append(record.getMessage())
                return True  # Preserve the caller's normal warning/error output.

        logger = logging.getLogger("onnxruntime.quantization.shape_inference")
        capture = Errors()
        logger.addFilter(capture)
        try:
            quant_pre_process(
                input_model=str(source), output_model_path=str(model),
                skip_optimization=False, skip_symbolic_shape=False, skip_onnx_shape=False,
                auto_merge=False, guess_output_rank=False, save_as_external_data=False,
            )
        finally:
            logger.removeFilter(capture)
        # ORT can catch an optimization exception and still save a marked graph.
        if errors:
            raise ValueError("Preprocessing logged an error; rejecting its partial output. See logs above.")
    elif mode == "preprocessed-uint8":
        from onnxruntime.quantization import QuantType, quantize_dynamic
        source = output / "fp32-preprocessed/model/model.onnx"
        if not read(output / "fp32-parity.json")["passed"]:
            raise ValueError("FP32 parity must pass before quantization")
        quantize_dynamic(
            str(source), str(model), op_types_to_quantize=["MatMul", "Gather"],
            weight_type=QuantType.QUInt8, per_channel=False, reduce_range=False,
            use_external_data_format=False, extra_options={"MatMulConstBOnly": True},
        )
    else:
        raise ValueError("Control does not need preparation")
    audit = graph_summary(model)
    if mode != "fp32-original" and not audit["preprocessing_marker"]:
        raise ValueError("Expected preprocessing metadata is missing")
    if mode == "preprocessed-uint8" and (not audit["uint8_gathers"]
                                        or not audit["operators"].get("MatMulInteger")):
        raise ValueError("Expected uint8 Gather and matrix quantization did not occur")
    shutil.copy2(work / "compact-model/tokenizer.json", destination / "tokenizer.json")
    metadata = read(work / "compact-model/embedding-manifest.json")
    metadata.update(quantization=mode + "-diagnostic", source_graph_sha256=digest(source),
                    files={name: {"bytes": (destination / name).stat().st_size,
                                  "sha256": digest(destination / name)}
                           for name in ("model.onnx", "tokenizer.json")})
    write(destination / "embedding-manifest.json", metadata)
    write(output / mode / "graph-audit.json", audit)


def measure(work, output, mode):
    quality = helper()
    if not mode.startswith("fp32-"):
        # Exact prior measurement harness; conversion imports never enter here.
        quality.measure(work, output, mode)
        return
    import numpy as np
    import onnxruntime as ort
    actual_session = ort.InferenceSession
    raw_outputs = []

    class Capture:
        def __init__(self, session):
            self.session = session

        def __getattr__(self, name):
            return getattr(self.session, name)

        def run(self, *args, **kwargs):
            values = self.session.run(*args, **kwargs)
            if len(values) != 1 or values[0].shape != (1, 384):
                raise ValueError("Unexpected FP32 output shape")
            raw_outputs.append(values[0].copy())
            return values

    with patch.object(ort, "InferenceSession", lambda *a, **k: Capture(actual_session(*a, **k))):
        quality.measure(work, output, mode)
    expected = len(read(work / "inputs.json")["cases"]) + 24
    if len(raw_outputs) != expected:
        raise ValueError("Incomplete FP32 request schedule")
    np.save(output / mode / "raw-outputs.npy", np.concatenate(raw_outputs), allow_pickle=False)


def compare_fp32(output, reference):
    import numpy as np
    before, after = [read(output / mode / "measurement.json") for mode in MODES[1:3]]
    left, right = [np.load(output / mode / "raw-outputs.npy", allow_pickle=False).astype(np.float64)
                   for mode in MODES[1:3]]
    if (left.shape != right.shape or left.ndim != 2 or left.shape[1] != 384
            or not np.isfinite(left).all() or not np.isfinite(right).all()
            or np.any(np.linalg.norm(left, axis=1) == 0) or np.any(np.linalg.norm(right, axis=1) == 0)):
        raise ValueError("Invalid FP32 comparison arrays")
    cosine = np.sum(left * right, axis=1) / (np.linalg.norm(left, axis=1) * np.linalg.norm(right, axis=1))
    signature = lambda item: [(r["tokens"], r["shape"], r["dtype"]) for r in item["trace"]]
    a, b = before["measurement"], after["measurement"]
    gates = {
        "same_request_schedule": signature(before) == signature(after),
        "same_inputs_and_revision": a["inputs_sha256"] == b["inputs_sha256"] == reference["inputs_sha256"]
        and a["model_revision"] == b["model_revision"] == reference["model_revision"],
        "all_outputs_close": bool(np.allclose(left, right, rtol=1e-4, atol=1e-5)),
        "all_cosines_at_least_0_999999": bool(np.all(cosine >= .999999)),
        "identical_ordered_rankings": a["rankings"] == b["rankings"],
        "original_matches_saved_fp32_rankings": a["rankings"] == reference["rankings"],
        "stable_language_switches": all(item["candidate_rankings_stable_after_language_switch"]
                                        for item in (before, after)),
    }
    value = {"gates": gates, "passed": all(gates.values()), "requests": len(left),
             "max_absolute_difference": float(np.abs(left - right).max()),
             "minimum_cosine": float(cosine.min()), "bitwise_identical": bool(np.array_equal(left, right)),
             "rtol": 1e-4, "atol": 1e-5,
             "note": "Includes all six 512-token probes; FP32 memory is diagnostic, not the uint8 serving gate."}
    write(output / "fp32-parity.json", value)
    return value


def sources(work):
    reports = [p for p in work.glob("disk-retrieval-*/disk-retrieval-report.json")
               if read(p).get("script_sha256") == DISK_SHA]
    if not reports:
        raise ValueError("Restore the checkpoint first; no verified disk-retrieval report found.")
    prior = max(reports, key=lambda p: p.stat().st_mtime)
    report = read(prior)
    one = report["variants"]["one-index"]
    if not (report["original_files_unchanged"] and one["identical_vectors_to_original_uint8"]
            and one["identical_rankings_to_original_uint8"]):
        raise ValueError("Checkpoint lacks a passing disk-retrieval parity screen")
    tracked = {Path(name): checksum for name, checksum in report["source_sha256"].items()}
    for path in (prior, prior.parent / "stores.json", prior.parent / "screen-inputs.json",
                 work / "inputs.json", work / "candidate.json", work / "reference.json", work / "e5-fp32.onnx",
                 work / "compact-model/embedding-manifest.json"):
        tracked.setdefault(path, digest(path))
    for item in report["stores"].values():
        tracked[prior.parent / item["database"]] = item["database_sha256"]
    if any(not path.resolve().is_relative_to(work) or digest(path) != checksum for path, checksum in tracked.items()):
        raise ValueError("Checkpoint input changed or points outside WORK")
    return prior, tracked


def run(work):
    quality = helper()
    quality.helper().helper().tokenizer_helper().load_helper()  # Check all five helper hashes up front.
    prior, tracked = sources(work)
    original, reference = read(work / "candidate.json"), read(work / "reference.json")
    output = Path(tempfile.mkdtemp(prefix="onnx-preprocessing-", dir=work))
    print("Output:", output, flush=True)
    env = dict(os.environ, HF_HUB_OFFLINE="1", TRANSFORMERS_OFFLINE="1", TOKENIZERS_PARALLELISM="false",
               OMP_NUM_THREADS="1", OPENBLAS_NUM_THREADS="1")
    env.pop("HF_TOKEN", None)
    report = {"schema_version": 1, "diagnostic_only": True, "script_sha256": digest(__file__),
              "quality_helper_sha256": QUALITY_SHA, "source_sha256": {str(p): h for p, h in tracked.items()},
              "variants": {}, "status": "running",
              "preprocessing": {"symbolic_shape_inference": True, "basic_graph_optimization": True,
                                "onnx_shape_inference": True, "auto_merge": False},
              "limitation": "Reused development queries, not held-out evaluation. Sequential process RSS is not a container/concurrency guarantee. FP32 measurements retain raw outputs for comparison; uint8 measurements do not. No automatic selection, uploads, index rebuild, or serving changes."}
    report_path = output / "onnx-preprocessing-report.json"
    try:
        for mode in MODES:
            (output / mode).mkdir()
            write(output / mode / "screen-inputs.json", {"disk_report": str(prior)})
            for stage in (("measure",) if mode == "control" else ("prepare", "measure")):
                print(f"Starting {mode}: {stage}", flush=True)
                process = subprocess.run([sys.executable, "-u", str(Path(__file__).resolve()), "--work", str(work),
                                          "--output", str(output), "--mode", mode, "--child", stage], env=env)
                if process.returncode:
                    report["variants"][mode] = {"status": "failed", "stage": stage, "exit_code": process.returncode}
                    raise RuntimeError(f"{mode} {stage} failed; keep the files and share the error above.")
            item = quality.summarize(read(output / mode / "measurement.json"), original, reference)
            if mode != "control":
                item["graph_audit"] = read(output / mode / "graph-audit.json")
            report["variants"][mode] = item
            if mode == "control":
                report["restored_control_passed"] = bool(item["identical_frozen_vectors_to_original_uint8"]
                    and item["identical_frozen_rankings_to_original_uint8"] and all(item["gates"].values()))
                if not report["restored_control_passed"]:
                    raise ValueError("Restored control failed numerical/runtime gates; no conversion attempted.")
            if mode == "fp32-preprocessed":
                report["fp32_parity"] = compare_fp32(output, reference)
                if not report["fp32_parity"]["passed"]:
                    raise ValueError("FP32 parity failed; quantization was not attempted.")
            write(report_path, report)
        report["status"] = "completed"
    except Exception as error:
        report.update(status="stopped", error=str(error))
        raise
    finally:
        report["original_files_unchanged"] = all(p.is_file() and digest(p) == h for p, h in tracked.items())
        if not report["original_files_unchanged"]:
            report.update(status="stopped", error="Original artifact checksum changed")
        write(report_path, report)
        print("REPORT:", report_path, flush=True)
    if not report["original_files_unchanged"]:
        raise ValueError("Original artifact changed")


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
