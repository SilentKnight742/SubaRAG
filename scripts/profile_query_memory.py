"""Offline phase-level memory profile of the verified one-index retrieval run.

No model/index rebuild, query-length changes, uploads, or serving changes.
The measurements add instrumentation overhead; they are not a latency benchmark.
"""

from __future__ import annotations

import argparse
from contextlib import contextmanager
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
    raise ValueError("Keep the previously uploaded check_disk_retrieval.py beside this script.")


def child(work, output):
    disk = helper()
    events, context = [], {}

    @contextmanager
    def phase(name, **details):
        before, started = disk.memory(), perf_counter()
        yield
        after = disk.memory()
        events.append({"phase": name, **context, **details, "before": before, "after": after,
                       "elapsed_ms": (perf_counter() - started) * 1000,
                       "new_peak_mib": max(0, after["peak_mib"] - before["peak_mib"])})

    with phase("runtime_imports"):
        import onnxruntime as ort
        from subarag.experiments import compact_embedding as experiment
        f64 = disk.tokenizer_helper()
    report_path = Path(read(output / "profile-inputs.json")["disk_report"])
    cases = read(work / "inputs.json")["cases"]
    scheduled = [{"request_group": "warmup", "language": lang} for lang in disk.LANGUAGES]
    scheduled += [{"request_group": "frozen", "language": lang, "case_id": case["id"]}
                  for lang in disk.LANGUAGES for case in cases if case["language"] == lang]
    scheduled += [{"request_group": kind, "language": lang}
                  for _ in range(2) for lang in disk.LANGUAGES for kind in ("long_probe", "short_after_long")]
    scheduled += [{"request_group": "language_switch", "language": lang}
                  for _ in range(3) for lang in disk.LANGUAGES]
    request_count = 0
    actual_session, actual_encoder = ort.InferenceSession, experiment.OnnxQueryEmbedder
    actual_tokenizer_class = f64.tokenizer_class
    actual_get, actual_search = disk.IndexPool.get, disk.DiskIndex.search

    class SessionProxy:
        def __init__(self, session):
            self.wrapped = session

        def __getattr__(self, name):
            return getattr(self.wrapped, name)

        def run(self, outputs, feeds, *args, **kwargs):
            context["encoded_tokens"] = int(feeds["input_ids"].shape[1])
            with phase("onnx_inference"):
                return self.wrapped.run(outputs, feeds, *args, **kwargs)

    def create_session(*args, **kwargs):
        with phase("onnx_session_creation"):
            session = actual_session(*args, **kwargs)
        return SessionProxy(session)

    def tokenizer_class(light):
        base = actual_tokenizer_class(light)

        class ProfiledTokenizer(base):
            def __init__(self, *args, **kwargs):
                with phase("compact_tokenizer_loading"):
                    super().__init__(*args, **kwargs)

            def encode(self, text):
                with phase("tokenization", input_characters=len(text)):
                    encoded = super().encode(text)
                    context["encoded_tokens"] = len(encoded.ids)
                return encoded
        return ProfiledTokenizer

    class ProfiledEncoder(actual_encoder):
        def __init__(self, *args, **kwargs):
            with phase("encoder_construction"):
                super().__init__(*args, **kwargs)

        def embed_queries(self, texts):
            nonlocal request_count
            if len(texts) != 1 or request_count >= len(scheduled):
                raise ValueError("Unexpected benchmark request schedule; do not guess phase labels")
            context.clear()
            context.update(scheduled[request_count], request_number=request_count + 1)
            request_count += 1
            with phase("query_encoding"):
                return super().embed_queries(texts)

    def index_get(pool, language):
        with phase("index_access", requested_language=language, cache_miss=language not in pool.loaded):
            return actual_get(pool, language)

    def index_search(index, vector, k):
        with phase("search_and_document_reads"):
            return actual_search(index, vector, k)

    result = {}
    def capture(path, value):
        if Path(path) != report_path.parent / "one-index.json" or result:
            raise ValueError("Unexpected diagnostic write")
        result.update(value)

    with patch.object(disk, "tokenizer_helper", lambda: f64), \
            patch.object(f64, "tokenizer_class", tokenizer_class), \
            patch.object(ort, "InferenceSession", create_session), \
            patch.object(experiment, "OnnxQueryEmbedder", ProfiledEncoder), \
            patch.object(disk.IndexPool, "get", index_get), \
            patch.object(disk.DiskIndex, "search", index_search), \
            patch.object(disk, "write", capture):
        disk.measure(work, report_path.parent, "one-index")
    if request_count != len(scheduled) or not result:
        raise ValueError("Incomplete profile")
    if any(name in sys.modules for name in ("torch", "transformers", "sentence_transformers", "sentencepiece")):
        raise ValueError("Unexpected heavyweight import")
    write(output / "profile-measurement.json", {"events": events, "measurement": result,
                                               "onnxruntime_version": ort.__version__})


def run(work):
    disk = helper()
    candidates = [p for p in work.glob("disk-retrieval-*/disk-retrieval-report.json")
                  if read(p).get("script_sha256") == DISK_SHA]
    if not candidates:
        raise ValueError("Keep the completed disk-retrieval run in WORK; restore the checkpoint if missing.")
    prior = max(candidates, key=lambda p: p.stat().st_mtime)
    previous = read(prior)
    baseline = previous["variants"]["one-index"]
    if not (previous["original_files_unchanged"] and baseline["identical_vectors_to_original_uint8"]
            and baseline["identical_rankings_to_original_uint8"] and baseline["mixed_language_rankings_identical"]):
        raise ValueError("Previous one-index run did not preserve results")
    tracked = {Path(name): checksum for name, checksum in previous["source_sha256"].items()}
    for name in ("screen-inputs.json", "stores.json", "one-index.json", "disk-retrieval-report.json"):
        tracked[prior.parent / name] = digest(prior.parent / name)
    for item in previous["stores"].values():
        tracked[prior.parent / item["database"]] = item["database_sha256"]
    if any(not p.resolve().is_relative_to(work) or digest(p) != checksum for p, checksum in tracked.items()):
        raise ValueError("Original artifacts changed or refer outside WORK")
    output = Path(tempfile.mkdtemp(prefix="query-memory-", dir=work))
    write(output / "profile-inputs.json", {"disk_report": str(prior)})
    print("Output:", output, flush=True)
    env = dict(os.environ, HF_HUB_OFFLINE="1", TRANSFORMERS_OFFLINE="1", TOKENIZERS_PARALLELISM="false",
               OMP_NUM_THREADS="1", OPENBLAS_NUM_THREADS="1")
    env.pop("HF_TOKEN", None)
    subprocess.run([sys.executable, "-u", str(Path(__file__).resolve()), "--work", str(work),
                    "--output", str(output), "--child"], env=env, check=True)
    measured = read(output / "profile-measurement.json")
    value, original = measured["measurement"], read(work / "candidate.json")
    if value["inputs_sha256"] != original["inputs_sha256"] or value["model_revision"] != original["model_revision"]:
        raise ValueError("Profile provenance differs")
    value["identical_vectors_to_original_uint8"] = value.pop("vectors") == original["vectors"]
    value["identical_rankings_to_original_uint8"] = value.pop("rankings") == original["rankings"]
    value["within_400_mib"] = value["peak_rss_mib"] <= 400
    # Nested phase windows overlap: never add their peak increments together.
    events = measured["events"]
    summaries = {name: {"calls": len(rows), "largest_new_peak_mib": max(r["new_peak_mib"] for r in rows),
                        "highest_recorded_peak_mib": max(r["after"]["peak_mib"] for r in rows)}
                 for name in sorted({event["phase"] for event in events})
                 if (rows := [event for event in events if event["phase"] == name])}
    if any(digest(p) != checksum for p, checksum in tracked.items()):
        raise ValueError("An original file changed during profiling")
    write(output / "query-memory-report.json", {
        "schema_version": 1, "diagnostic_only": True, "script_sha256": digest(__file__),
        "helper_sha256": DISK_SHA, "original_files_unchanged": True,
        "source_sha256": {str(p): checksum for p, checksum in tracked.items()},
        "previous_peak_mib": baseline["peak_rss_mib"], "measurement": value,
        "phase_summary": summaries, "events": events,
        "onnxruntime_version": measured["onnxruntime_version"],
        "limitation": "Process RSS with instrumentation overhead, not container memory or a latency benchmark. Nested phases overlap. HWM identifies when a new peak occurred, not individual allocation/operator ownership. Model, 512-token ceiling, indexes, and existing Marathi quality failure are unchanged. No deployment.",
    })
    print("REPORT:", output / "query-memory-report.json", flush=True)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--work", type=Path, required=True)
    parser.add_argument("--output", type=Path)
    parser.add_argument("--child", action="store_true")
    args = parser.parse_args()
    if args.child:
        if args.output is None:
            parser.error("Child mode requires --output")
        child(args.work, args.output)
    else:
        run(args.work.resolve())
