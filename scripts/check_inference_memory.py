"""Offline screen of CPU allocator settings on the unchanged query model.

Reuses the verified disk-backed, one-index runtime. Each setting runs in a fresh
process with identical queries and the same 512-token ceiling. No deployment.
"""

from __future__ import annotations

import argparse
import hashlib
import importlib.util
import json
import os
from pathlib import Path
import subprocess
import sys
import tempfile
from time import perf_counter
from unittest.mock import patch

DISK_SHA = "5ebc4544ae6b4ad5b692c00ce4b6c0f15901f14c395899c6cb7c39f7f0b6d63d"
PROFILE_SHA = "5f092db8661f3a40c8ee3b02d903cf8912c695dc18f4f1210379e861c0309544"
MODES = ("control", "requested-arena", "requested-arena-shrink")


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
    for path in Path(__file__).resolve().parent.glob("check_disk_retrieval*.py"):
        if digest(path) == DISK_SHA:
            spec = importlib.util.spec_from_file_location("verified_disk_helper", path)
            module = importlib.util.module_from_spec(spec)
            spec.loader.exec_module(module)
            return module
    raise ValueError("Keep the prior check_disk_retrieval.py and tokenizer helper uploads beside this script.")


def child(work, output, mode):
    disk = helper()
    import onnxruntime as ort
    ort.disable_telemetry_events()
    selected = read(output / "screen-inputs.json")
    disk_dir = Path(selected["disk_report"]).parent
    trace, sessions, captured = [], [], {}
    actual_session = ort.InferenceSession
    use_arena, shrink = mode != "control", mode == "requested-arena-shrink"
    arena_config = {"max_mem": 0, "arena_extend_strategy": 1, "max_dead_bytes_per_chunk": 256}
    if use_arena:
        # kSameAsRequested = 1. A fresh child owns this environment allocator;
        # it cannot affect another variant or the notebook's Python process.
        info = ort.OrtMemoryInfo("Cpu", ort.OrtAllocatorType.ORT_ARENA_ALLOCATOR, 0, ort.OrtMemType.DEFAULT)
        ort.create_and_register_allocator(info, ort.OrtArenaCfg(arena_config))
    run_options = ort.RunOptions()
    if shrink:
        run_options.add_run_config_entry("memory.enable_memory_arena_shrinkage", "cpu:0")

    class SessionProxy:
        def __init__(self, session):
            self.session = session

        def __getattr__(self, name):
            return getattr(self.session, name)

        def run(self, names, feeds):
            before, started = disk.memory(), perf_counter()
            # Pass equivalent empty RunOptions to control as well, for fairness.
            result = self.session.run(names, feeds, run_options)
            elapsed = (perf_counter() - started) * 1000
            after = disk.memory()
            if len(result) != 1:
                raise ValueError("Unexpected model outputs")
            array = result[0]
            trace.append({"request": len(trace) + 1, "tokens": int(feeds["input_ids"].shape[1]),
                          "shape": list(array.shape), "dtype": str(array.dtype),
                          "output_sha256": hashlib.sha256(array.tobytes()).hexdigest(),
                          "inference_ms": elapsed, "before": before, "after": after})
            return result

    def session_factory(path, *, sess_options, providers):
        if (Path(path).resolve() != (work / "compact-model/model.onnx").resolve()
                or providers != ["CPUExecutionProvider"] or sessions):
            raise ValueError("Unexpected inference session")
        if sess_options.enable_cpu_mem_arena or sess_options.enable_mem_pattern:
            raise ValueError("Baseline session settings changed")
        if use_arena:
            sess_options.enable_cpu_mem_arena = True
            sess_options.add_session_config_entry("session.use_env_allocators", "1")
        before, started = disk.memory(), perf_counter()
        session = actual_session(path, sess_options=sess_options, providers=providers)
        options = session.get_session_options()
        sessions.append({"before": before, "after": disk.memory(),
                         "load_ms": (perf_counter() - started) * 1000,
                         "enable_cpu_mem_arena": options.enable_cpu_mem_arena,
                         "enable_mem_pattern": options.enable_mem_pattern,
                         "enable_mem_reuse": options.enable_mem_reuse,
                         "graph_optimization_level": str(options.graph_optimization_level),
                         "intra_op_threads": options.intra_op_num_threads,
                         "inter_op_threads": options.inter_op_num_threads,
                         "use_env_allocators": options.get_session_config_entry("session.use_env_allocators") if use_arena else None})
        return SessionProxy(session)

    def capture(path, value):
        if Path(path) != disk_dir / "one-index.json" or captured:
            raise ValueError("Unexpected measurement write")
        captured.update(value)

    with patch.object(ort, "InferenceSession", session_factory), patch.object(disk, "write", capture):
        disk.measure(work, disk_dir, "one-index")
    count = len(read(work / "inputs.json")["cases"])
    if len(trace) != count + 24 or len(sessions) != 1 or not captured:
        raise ValueError("Unexpected benchmark schedule")
    long_rows = [trace[3 + count + offset] for offset in range(0, 12, 2)]
    if any(row["tokens"] != 512 for row in long_rows):
        raise ValueError("A long probe no longer reaches 512 tokens")
    observed_peak = max([disk.memory()["peak_mib"], captured["peak_rss_mib"]]
                        + [r[side]["peak_mib"] for r in trace + sessions for side in ("before", "after")])
    # Report the maximum observed high-water reading, including transient peaks
    # sampled after long inference, even if a later /proc reading is smaller.
    captured["peak_rss_mib"] = observed_peak
    from subarag.retrieval import latency_summary
    write(output / f"{mode}.json", {"status": "completed", "measurement": captured,
          "mode": mode, "onnxruntime_version": ort.__version__, "sessions": sessions, "trace": trace,
          "arena_config": arena_config if use_arena else None, "shrink_after_run": shrink,
          "long_inference_ms": latency_summary([row["inference_ms"] for row in long_rows])})
    print(f"{mode}: observed peak {observed_peak:.1f} MiB; all six 512-token probes completed.", flush=True)


def run(work):
    helper()
    profiles = [p for p in work.glob("query-memory-*/query-memory-report.json")
                if read(p).get("script_sha256") == PROFILE_SHA]
    if not profiles:
        raise ValueError("Keep the completed memory profile in WORK. If Colab reset, restore the checkpoint first.")
    prior = max(profiles, key=lambda p: p.stat().st_mtime)
    profile = read(prior)
    if not (profile["original_files_unchanged"] and profile["measurement"]["identical_vectors_to_original_uint8"]
            and profile["measurement"]["identical_rankings_to_original_uint8"]):
        raise ValueError("The memory profile did not preserve results")
    tracked = {Path(name): checksum for name, checksum in profile["source_sha256"].items()}
    tracked[prior] = digest(prior)
    reports = [p for p in tracked if p.name == "disk-retrieval-report.json"]
    if len(reports) != 1:
        raise ValueError("Ambiguous disk-retrieval provenance")
    if any(not p.resolve().is_relative_to(work) or digest(p) != checksum for p, checksum in tracked.items()):
        raise ValueError("An original input changed or points outside WORK")
    output = Path(tempfile.mkdtemp(prefix="inference-memory-", dir=work))
    write(output / "screen-inputs.json", {"disk_report": str(reports[0])})
    print("Output:", output, flush=True)
    env = dict(os.environ, HF_HUB_OFFLINE="1", TRANSFORMERS_OFFLINE="1", TOKENIZERS_PARALLELISM="false",
               OMP_NUM_THREADS="1", OPENBLAS_NUM_THREADS="1")
    env.pop("HF_TOKEN", None)
    for mode in MODES:
        process = subprocess.run([sys.executable, "-u", str(Path(__file__).resolve()), "--work", str(work),
                                  "--output", str(output), "--child", mode], env=env)
        if process.returncode:
            write(output / f"{mode}.json", {"status": "failed", "exit_code": process.returncode,
                                           "message": "See the child error in notebook output. No fallback settings were used."})
            if mode == "control":
                raise RuntimeError("Control failed; keep the files and share the actual error above.")
    control = read(output / "control.json")
    original, reference = read(work / "candidate.json"), read(work / "reference.json")
    variants = {}
    signature = lambda rows: [(r["tokens"], r["shape"], r["dtype"], r["output_sha256"]) for r in rows]
    for mode in MODES:
        item = read(output / f"{mode}.json")
        if item["status"] != "completed":
            variants[mode] = item
            continue
        value = item["measurement"]
        if value["inputs_sha256"] != original["inputs_sha256"] or value["model_revision"] != original["model_revision"]:
            raise ValueError("Comparison provenance differs")
        parity = value.pop("vectors") == original["vectors"]
        ranking_parity = value.pop("rankings") == original["rankings"]
        gates = {"identical_frozen_vectors": parity, "identical_frozen_rankings": ranking_parity,
                 "identical_all_inference_outputs_to_control": signature(item["trace"]) == signature(control["trace"]),
                 "identical_switch_rankings": value["mixed_language_rankings_identical"],
                 "within_400_mib": value["peak_rss_mib"] <= 400,
                 "grouped_p95_within_500_ms": all(v["p95"] <= 500 for v in value["retrieval_ms"].values()),
                 "switching_p95_within_500_ms": value["mixed_language_ms"]["p95"] <= 500,
                 "no_heavyweight_runtime": not value["heavyweight_runtime_imported"],
                 "one_resident_index": value["max_resident_index_count"] == 1}
        quality = {lang: reference["metrics"][lang]["ndcg@8"] > 0 and
                   value["metrics"][lang]["ndcg@8"] >= .98 * reference["metrics"][lang]["ndcg@8"]
                   for lang in ("en", "hi", "mr")}
        item.update(gates=gates, quality_gate_by_language=quality,
                    screen_passed=all(gates.values()) and all(quality.values()))
        variants[mode] = item
    if not (variants["control"]["gates"]["identical_frozen_vectors"]
            and variants["control"]["gates"]["identical_frozen_rankings"]):
        raise ValueError("Fresh control no longer reproduces the verified baseline")
    if any(digest(p) != checksum for p, checksum in tracked.items()):
        raise ValueError("An original input changed during the experiment")
    write(output / "inference-memory-report.json", {
        "schema_version": 1, "diagnostic_only": True, "script_sha256": digest(__file__),
        "disk_helper_sha256": DISK_SHA, "original_files_unchanged": True,
        "source_sha256": {str(p): checksum for p, checksum in tracked.items()}, "variants": variants,
        "limitation": "Fresh isolated Colab processes, single sequential requests, warm OS file cache. Allocator settings only: no graph, precision, tokenizer, 512-token limit or index changes. Instrumentation adds some overhead; process RSS is not a container guarantee. max_mem=0 is the allocator default, not a process memory cap. Existing Marathi quality failure remains. No deployment or automatic promotion.",
    })
    print("REPORT:", output / "inference-memory-report.json", flush=True)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--work", type=Path, required=True)
    parser.add_argument("--output", type=Path)
    parser.add_argument("--child", choices=MODES)
    args = parser.parse_args()
    if args.child:
        if args.output is None:
            parser.error("Child mode requires --output")
        child(args.work, args.output, args.child)
    else:
        run(args.work.resolve())
