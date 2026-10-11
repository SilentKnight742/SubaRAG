"""Compare the existing FP32 ONNX export and tokenizers in the same Colab runtime.

Run: python check_onnx_parity.py --work <existing WORK directory>
Offline, diagnostic only; existing reports, models, and indexes stay unchanged.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
from pathlib import Path
import shutil
import subprocess
import sys
import tempfile
from unittest.mock import patch


def read(path: Path):
    return json.loads(path.read_text(encoding="utf-8"))


def write(path: Path, value) -> None:
    path.write_text(json.dumps(value, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        while chunk := handle.read(1024 * 1024):
            digest.update(chunk)
    return digest.hexdigest()


def token_cases(work: Path):
    cases = [(case["language"] + ":" + case["id"], case["query"])
             for case in read(work / "inputs.json")["cases"]]
    edges = {
        "empty": "", "whitespace": " \t\n", "spacing": "  hello\tworld\n ",
        "unicode": "café cafe\u0301 नमस्ते मराठी 👩🏽‍💻",
        "numbers": "₹1,234.56 / 2026-10-07 +42%",
        "special_tokens": "<s> </s> <pad> <unk> <mask>",
        "mask_spacing": "hello  <mask>world <mask> world",
        "controls": "a\x00b\u200bc\u200dd\u00a0e",
        "long_en": "What is a corporation? " * 600,
        "long_hi": "कंपनी क्या है? " * 600,
        "long_mr": "कंपनी म्हणजे काय? " * 600,
    }
    return cases + [("edge:" + key, value) for key, value in edges.items()]


def tokens(work: Path, output: Path, mode: str) -> None:
    metadata = read(work / "compact-model/embedding-manifest.json")
    if mode == "raw-tokenizer":
        from tokenizers import Tokenizer
        tokenizer = Tokenizer.from_file(str(work / "compact-model/tokenizer.json"))
        tokenizer.no_padding()
        tokenizer.enable_truncation(max_length=512)

        def encode(text):
            value = tokenizer.encode(text)
            return {"input_ids": value.ids, "attention_mask": value.attention_mask}
    else:
        from transformers import AutoTokenizer
        # Missing cached files must fail, never trigger a new model download.
        tokenizer = AutoTokenizer.from_pretrained(
            metadata["model_name"], revision=metadata["model_revision"], local_files_only=True,
        )

        def encode(text):
            value = tokenizer(text, padding=False, truncation=True, max_length=512)
            return {key: value[key] for key in ("input_ids", "attention_mask")}

    results = {key: encode("query: " + text) for key, text in token_cases(work)}
    write(output / f"{mode}.json", results)
    print(f"{mode}: checked {len(results)} inputs", flush=True)


def prepare_model(work: Path, output: Path) -> None:
    metadata = read(work / "compact-model/embedding-manifest.json")
    destination = output / "fp32-model"
    destination.mkdir()
    for source, name in ((work / "e5-fp32.onnx", "model.onnx"),
                         (work / "compact-model/tokenizer.json", "tokenizer.json")):
        try:
            os.link(source, destination / name)
        except OSError:
            shutil.copy2(source, destination / name)
    metadata.update(quantization="none-fp32-diagnostic", files={
        name: {"bytes": (destination / name).stat().st_size,
               "sha256": sha256(destination / name)} for name in ("model.onnx", "tokenizer.json")})
    write(destination / "embedding-manifest.json", metadata)


def measure(work: Path, output: Path) -> None:
    from subarag.experiments import compact_embedding as experiment
    real_embedder = experiment.OnnxQueryEmbedder
    result = {}

    def fp32_embedder(_directory, *, expected_revision):
        return real_embedder(output / "fp32-model", expected_revision=expected_revision)

    with patch.object(experiment, "OnnxQueryEmbedder", fp32_embedder), \
            patch.object(experiment, "write_json", lambda _path, value: result.update(value)):
        experiment.measure(work, compact=True)
    result["runtime"] = "onnx-fp32"
    write(output / "fp32.json", result)


def compare(reference, actual):
    import numpy as np
    if (actual["inputs_sha256"] != reference["inputs_sha256"]
            or actual["model_revision"] != reference["model_revision"]
            or set(actual["vectors"]) != set(reference["vectors"])
            or set(actual["rankings"]) != set(reference["vectors"])
            or set(reference["rankings"]) != set(reference["vectors"])):
        raise ValueError("Reference/candidate provenance or query IDs differ")
    languages = {}
    for language in ("en", "hi", "mr"):
        keys = [key for key in reference["vectors"] if key.startswith(language + ":")]
        left = np.asarray([reference["vectors"][key] for key in keys], dtype=np.float64)
        right = np.asarray([actual["vectors"][key] for key in keys], dtype=np.float64)
        if (left.shape != right.shape or left.ndim != 2 or left.shape[1] != 384
                or not np.isfinite(left).all() or not np.isfinite(right).all()
                or np.any(np.linalg.norm(left, axis=1) == 0)
                or np.any(np.linalg.norm(right, axis=1) == 0)):
            raise ValueError("Invalid embedding vectors in comparison")
        cosine = np.sum(left * right, axis=1) / (np.linalg.norm(left, axis=1) * np.linalg.norm(right, axis=1))
        baseline = reference["metrics"][language]["ndcg@8"]
        ndcg = actual["metrics"][language]["ndcg@8"]
        languages[language] = {
            "queries": len(keys), "embedding_cosine_min": float(cosine.min()),
            "embedding_cosine_mean": float(cosine.mean()),
            "all_cosines_at_least_0_9999": bool(np.all(cosine >= 0.9999)),
            "max_absolute_component_difference": float(np.abs(left - right).max()),
            "max_l2_difference": float(np.linalg.norm(left - right, axis=1).max()),
            "identical_ordered_top8": sum(reference["rankings"][key] == actual["rankings"][key] for key in keys),
            "identical_top8_sets": sum(set(reference["rankings"][key]) == set(actual["rankings"][key]) for key in keys),
            "reference_ndcg_at_8": baseline, "actual_ndcg_at_8": ndcg,
            "relative_ndcg_change_percent": 100 * (ndcg / baseline - 1) if baseline else None,
            "within_original_quality_gate": baseline > 0 and ndcg >= baseline * 0.98,
        }
    return languages


def run(work: Path) -> None:
    required = ("inputs.json", "reference.json", "candidate.json", "e5-fp32.onnx",
                "compact-model/embedding-manifest.json", "compact-model/tokenizer.json")
    for name in required:
        if not (work / name).is_file():
            raise ValueError(f"Missing {name}; use the same Colab runtime and original WORK directory.")
    original_hashes = {name: sha256(work / name) for name in required}
    metadata = read(work / "compact-model/embedding-manifest.json")
    if original_hashes["compact-model/tokenizer.json"] != metadata["files"]["tokenizer.json"]["sha256"]:
        raise ValueError("Existing tokenizer checksum mismatch")
    reference = read(work / "reference.json")
    if (reference["inputs_sha256"] != original_hashes["inputs.json"]
            or reference["model_revision"] != metadata["model_revision"]):
        raise ValueError("Frozen inputs/model do not match the saved reference")
    output = Path(tempfile.mkdtemp(prefix="onnx-parity-", dir=work))
    print("Comparison output:", output, flush=True)
    prepare_model(work, output)
    env = dict(os.environ, HF_HUB_OFFLINE="1", TRANSFORMERS_OFFLINE="1",
               OMP_NUM_THREADS="1", OPENBLAS_NUM_THREADS="1", TOKENIZERS_PARALLELISM="false")
    env.pop("HF_TOKEN", None)
    # Tokenizer imports and their allocation peaks never enter the inference process.
    for mode in ("raw-tokenizer", "hf-tokenizer", "fp32"):
        print("Starting", mode, "(offline)…", flush=True)
        subprocess.run([sys.executable, "-u", str(Path(__file__).resolve()), "--work", str(work),
                        "--output", str(output), "--child", mode], env=env, check=True)
    raw, hf = (read(output / f"{mode}-tokenizer.json") for mode in ("raw", "hf"))
    if set(raw) != set(hf):
        raise ValueError("Tokenizer comparison IDs differ")
    mismatches = [key for key in raw if raw[key] != hf[key]]
    fp32 = read(output / "fp32.json")
    comparisons = {"onnx_fp32": compare(reference, fp32),
                   "original_uint8": compare(reference, read(work / "candidate.json"))}
    tokenizer_json = read(work / "compact-model/tokenizer.json")
    unchanged = all(sha256(work / name) == digest for name, digest in original_hashes.items())
    if not unchanged:
        raise RuntimeError("An original input changed during the comparison; do not use these results")
    fp32.pop("vectors")
    fp32.pop("rankings")
    report = {
        "schema_version": 1, "diagnostic_only": True,
        "script_sha256": sha256(Path(__file__)), "original_files_unchanged": unchanged,
        "source_sha256": original_hashes, "fp32_measurement": fp32,
        "comparison_to_sentence_transformers": comparisons,
        "tokenizer_parity": {"cases": len(raw), "exact_match": not mismatches,
                             "mismatched_case_ids": mismatches,
                             "comparison": "Saved raw tokenizer vs cached pinned AutoTokenizer, query prefix, no padding, truncation at 512; not a replacement-tokenizer approval."},
        "tokenizer_inventory": {
            "model_type": tokenizer_json.get("model", {}).get("type"),
            "vocabulary_entries": len(tokenizer_json.get("model", {}).get("vocab", [])),
            "added_tokens": len(tokenizer_json.get("added_tokens", [])),
            "normalizer": tokenizer_json.get("normalizer", {}).get("type") if tokenizer_json.get("normalizer") else None,
            "pre_tokenizer": tokenizer_json.get("pre_tokenizer"),
            "post_processor": tokenizer_json.get("post_processor"),
        },
        "limitation": "192 frozen retrieval queries plus tokenizer edge cases are a diagnostic, not full correctness or a deployment gate. FP32 memory is measured for context, not proposed as a free-tier serving solution. No tokenizer replacement is made.",
    }
    write(output / "onnx-parity-report.json", report)
    print("REPORT:", output / "onnx-parity-report.json", flush=True)
    print("Share this report. No upload or deployment; original files unchanged.", flush=True)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--work", type=Path, required=True)
    parser.add_argument("--output", type=Path)
    parser.add_argument("--child", choices=("raw-tokenizer", "hf-tokenizer", "fp32"))
    args = parser.parse_args()
    if args.child:
        if args.output is None:
            parser.error("Child stages require --output")
        if args.child == "fp32":
            measure(args.work, args.output)
        else:
            tokens(args.work, args.output, args.child)
    else:
        run(args.work.resolve())
