"""Offline Colab screen: compact vocabulary with original float64 scoring.

Requires marisa-trie and the previous check_light_tokenizer.py upload.
Reuses the saved normalizer and existing models/indexes, never serving defaults.
"""

from __future__ import annotations

import argparse
import hashlib
import importlib.metadata
import importlib.util
import json
import math
import mmap
import os
from pathlib import Path
import re
import shutil
import struct
import subprocess
import sys
import tempfile
from unittest.mock import patch

HELPER_SHA = "b8b66f5b22ddd80be600400324b0df8b9002de4eb523f1900f76c9cb66b846f8"
RECORD = "<Id"  # original vocabulary ID plus IEEE-754 binary64 score


def digest(path):
    value = hashlib.sha256()
    with Path(path).open("rb") as handle:
        while chunk := handle.read(1024 * 1024):
            value.update(chunk)
    return value.hexdigest()


def read(path):
    return json.loads(Path(path).read_text(encoding="utf-8"))


def write(path, value):
    Path(path).write_text(json.dumps(value, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")


def load_helper():
    for path in Path(__file__).resolve().parent.glob("check_light_tokenizer*.py"):
        if digest(path) == HELPER_SHA:
            spec = importlib.util.spec_from_file_location("verified_light_helper", path)
            helper = importlib.util.module_from_spec(spec)
            spec.loader.exec_module(helper)
            return helper
    raise ValueError("Keep the previous check_light_tokenizer.py upload in the same Colab folder.")


def prepare(work, output):
    import marisa_trie
    from tokenizers import Tokenizer
    selected = read(output / "screen-inputs.json")
    prior = Path(selected["previous_report"])
    metadata = read(prior.parent / "light-manifest.json")
    source = work / "compact-model/tokenizer.json"
    if digest(source) != metadata["source_tokenizer_sha256"]:
        raise ValueError("Tokenizer source changed")
    pipeline = prior.parent / "pipeline.json"
    if digest(pipeline) != metadata["files"]["pipeline.json"]:
        raise ValueError("Saved normalization pipeline changed")
    source_model = read(source)["model"]
    # Capture the reference runtime's parsed binary64 values, not an assumption
    # that Python and Rust JSON parsers round every decimal identically. Loading
    # the large original vocabulary happens only in this preparation process.
    reference_tokenizer = Tokenizer.from_file(str(source))
    model = json.loads(reference_tokenizer.to_str())["model"]
    if model["type"] != "Unigram" or model["unk_id"] != 3 or model.get("byte_fallback", False):
        raise ValueError("Unsupported vocabulary contract")
    vocab = model["vocab"]
    if (len(vocab) != len(source_model["vocab"])
            or any(a[0] != b[0] for a, b in zip(vocab, source_model["vocab"]))):
        raise ValueError("Reference parser changed vocabulary identities")
    parser_differences = sum(a[1] != b[1] for a, b in zip(vocab, source_model["vocab"]))
    if not vocab or any(not piece or not math.isfinite(score) for piece, score in vocab):
        raise ValueError("Empty token or non-finite score in vocabulary")
    trie = marisa_trie.Trie(piece for piece, _ in vocab)
    if len(trie) != len(vocab):
        raise ValueError("Duplicate vocabulary piece")
    records = bytearray(len(vocab) * struct.calcsize(RECORD))
    for index, (piece, score) in enumerate(vocab):
        struct.pack_into(RECORD, records, trie[piece] * struct.calcsize(RECORD), index, score)
    trie.save(str(output / "vocabulary.marisa"))
    (output / "scores.bin").write_bytes(records)
    # Verify the serialized representation too, not only the in-memory builder.
    restored = Vocabulary(output)
    if any(restored[piece] != [(index, score)] for index, (piece, score) in enumerate(vocab)):
        raise ValueError("Serialized vocabulary changed an ID/score")
    shutil.copy2(pipeline, output / "pipeline.json")
    write(output / "float64-manifest.json", {
        "schema_version": 1, "kind": "marisa-unigram-float64", "record_format": RECORD,
        "source_tokenizer_sha256": digest(source), "vocabulary_entries": len(vocab),
        "scores_preserved_exactly": True, "unknown_id": 3,
        "score_source": "Loaded reference tokenizer binary64 scores, serialized without float32 conversion",
        "python_vs_reference_json_score_differences": parser_differences,
        "unknown_score": min(score for _, score in vocab) - 10.0,
        "max_piece_characters": max(len(piece) for piece, _ in vocab),
        "special_tokens": metadata["special_tokens"], "lstrip_tokens": metadata["lstrip_tokens"],
        "files": {name: digest(output / name) for name in ("vocabulary.marisa", "scores.bin", "pipeline.json")},
    })
    print(f"Preserved all {len(vocab)} IDs and scores exactly; no float32 conversion.", flush=True)


class Vocabulary:
    """Plain Unicode trie plus a fixed-width, read-only ID/score table.

    Keep binary records separate from keys so control characters in input cannot
    be confused with the separators used by a combined key/value trie.
    """

    def __init__(self, output):
        import marisa_trie
        self.trie = marisa_trie.Trie().mmap(str(output / "vocabulary.marisa"))
        with (output / "scores.bin").open("rb") as handle:
            self.records = mmap.mmap(handle.fileno(), 0, access=mmap.ACCESS_READ)
        if len(self.records) != len(self.trie) * struct.calcsize(RECORD):
            raise ValueError("Vocabulary/score table dimensions differ")

    def prefixes(self, text):
        return self.trie.prefixes(text)

    def __getitem__(self, text):
        return [struct.unpack_from(RECORD, self.records, self.trie[text] * struct.calcsize(RECORD))]

    def get(self, text):
        try:
            return self[text]
        except KeyError:
            return None


class Float64Segmenter:
    """Deterministic unigram best-path scoring; no input-specific exceptions.

    Match the original's left-to-right traversal, strict-greater replacement,
    binary64 sums, single-character unknown fallback, and unknown-span fusion.
    Unicode character boundaries have the same traversal order as UTF-8 boundaries.
    """

    def __init__(self, trie, *, unknown_id, unknown_score, max_piece_characters):
        self.trie = trie
        self.unknown_id = unknown_id
        self.unknown_score = unknown_score
        self.max_piece_characters = max_piece_characters

    def encode(self, text, *, out_type=int):
        if out_type is not int:
            raise ValueError("Only token IDs are supported")
        size = len(text)
        best = [0.0] * (size + 1)
        starts = [-1] * (size + 1)
        ids = [self.unknown_id] * (size + 1)
        for start in range(size):
            single = False
            # Bound copied suffixes by the vocabulary's longest piece. All
            # possible prefixes still participate; this does not truncate input.
            suffix = text[start:start + self.max_piece_characters]
            for piece in sorted(self.trie.prefixes(suffix), key=len):
                token_id, score = self.trie[piece][0]
                end = start + len(piece)
                candidate = score + best[start]
                if starts[end] == -1 or candidate > best[end]:
                    best[end], starts[end], ids[end] = candidate, start, token_id
                single = single or len(piece) == 1
            if not single:
                end = start + 1
                candidate = self.unknown_score + best[start]
                if starts[end] == -1 or candidate > best[end]:
                    best[end], starts[end], ids[end] = candidate, start, self.unknown_id
        spans = []
        end = size
        while end:
            start = starts[end]
            if start < 0 or start >= end:
                raise ValueError("Invalid segmentation backtrace")
            spans.append((start, end, ids[end]))
            end = start
        # Fuse adjacent unknown spans before their final vocabulary lookup, as
        # the original encoder does. Do not fuse across pre-tokenized pieces.
        result, unknown_start, unknown_end = [], None, None
        for start, end, token_id in reversed(spans):
            if token_id == self.unknown_id:
                unknown_start = start if unknown_start is None else unknown_start
                unknown_end = end
            else:
                if unknown_start is not None:
                    found = self.trie.get(text[unknown_start:unknown_end])
                    result.append(found[0][0] if found else self.unknown_id)
                    unknown_start = None
                result.append(token_id)
        if unknown_start is not None:
            found = self.trie.get(text[unknown_start:unknown_end])
            result.append(found[0][0] if found else self.unknown_id)
        return result


def tokenizer_class(helper):
    # Reuse the previously checked prefix, normalization, special-token and
    # truncation behavior verbatim. Replace only the vocabulary/segmenter.
    class Float64Tokenizer(helper.LightTokenizer):
        def __init__(self, output):
            from tokenizers import Tokenizer
            metadata = read(output / "float64-manifest.json")
            if metadata.get("kind") != "marisa-unigram-float64" or metadata.get("record_format") != RECORD:
                raise ValueError("Unsupported compact tokenizer artifact")
            if set(metadata["files"]) != {"vocabulary.marisa", "scores.bin", "pipeline.json"}:
                raise ValueError("Unexpected compact tokenizer files")
            for name, checksum in metadata["files"].items():
                if digest(output / name) != checksum:
                    raise ValueError("Compact tokenizer checksum mismatch")
            self.pipeline = Tokenizer.from_file(str(output / "pipeline.json"))
            self.processor = Float64Segmenter(
                Vocabulary(output),
                unknown_id=metadata["unknown_id"], unknown_score=metadata["unknown_score"],
                max_piece_characters=metadata["max_piece_characters"],
            )
            self.special = metadata["special_tokens"]
            self.lstrip_tokens = frozenset(metadata["lstrip_tokens"])
            self.pattern = re.compile("(" + "|".join(re.escape(key) for key in self.special) + ")")
            self.max_length = 512
    return Float64Tokenizer


def expanded_cases(helper, work):
    result = helper.cases(work)
    # Keep the failed synthetic input and add an entire family of punctuation
    # lengths, rather than hard-coding a special answer for its six characters.
    result.extend((f"precision:{count}:{suffix}", "?!" * count + suffix)
                  for count in range(1, 33) for suffix in ("", "hello🙂"))
    return result


def child(work, output, mode):
    helper = load_helper()
    if mode == "prepare":
        prepare(work, output)
        return
    all_cases = expanded_cases(helper, work)
    with patch.object(helper, "LightTokenizer", tokenizer_class(helper)), \
            patch.object(helper, "cases", lambda _work: all_cases):
        if mode in ("raw", "light"):
            helper.tokenize(work, output, mode)
        else:
            helper.measure(work, output)
    if any(name in sys.modules for name in ("torch", "transformers", "sentence_transformers", "sentencepiece")):
        raise RuntimeError("Unexpected heavyweight or superseded tokenizer import")


def run(work):
    helper = load_helper()
    reports = [path for path in work.glob("light-tokenizer-*/light-tokenizer-report.json")
               if read(path).get("script_sha256") == HELPER_SHA]
    if not reports:
        raise ValueError("Keep the original light-tokenizer report and artifacts in WORK.")
    prior = max(reports, key=lambda path: path.stat().st_mtime)
    previous = read(prior)
    names = ("inputs.json", "candidate.json", "reference.json", "compact-model/model.onnx",
             "compact-model/tokenizer.json", "compact-model/embedding-manifest.json")
    hashes = {name: digest(work / name) for name in names}
    if any(previous["source_sha256"][name] != value for name, value in hashes.items()):
        raise ValueError("Existing inputs changed since the light-tokenizer run")
    original, reference = read(work / "candidate.json"), read(work / "reference.json")
    if any(item["inputs_sha256"] != hashes["inputs.json"] for item in (original, reference)):
        raise ValueError("Frozen query fingerprints differ")
    tracked = {path: digest(path) for path in (prior, prior.parent / "light-manifest.json", prior.parent / "pipeline.json")}
    output = Path(tempfile.mkdtemp(prefix="float64-tokenizer-", dir=work))
    write(output / "screen-inputs.json", {"previous_report": str(prior)})
    print("Output:", output, flush=True)
    env = dict(os.environ, HF_HUB_OFFLINE="1", TRANSFORMERS_OFFLINE="1", TOKENIZERS_PARALLELISM="false",
               OMP_NUM_THREADS="1", OPENBLAS_NUM_THREADS="1")
    env.pop("HF_TOKEN", None)

    def stage(mode):
        subprocess.run([sys.executable, "-u", str(Path(__file__).resolve()), "--work", str(work),
                        "--output", str(output), "--child", mode], env=env, check=True)

    for mode in ("prepare", "raw", "light"):
        stage(mode)
    raw, light = read(output / "raw.json"), read(output / "light.json")
    if set(raw["tokens"]) != set(light["tokens"]):
        raise ValueError("Tokenizer case IDs differ")
    mismatches = [key for key in raw["tokens"] if raw["tokens"][key] != light["tokens"][key]]
    parity = {"cases": len(raw["tokens"]), "exact_match": not mismatches, "mismatched_case_ids": mismatches}
    measurement = None
    if not mismatches:
        print("Exact parity passed; measuring the existing uint8 pipeline with float64 tokenization.", flush=True)
        stage("measure")
        measurement = read(output / "measurement.json")
        if measurement["inputs_sha256"] != hashes["inputs.json"] or measurement["model_revision"] != original["model_revision"]:
            raise ValueError("Pipeline comparison provenance differs")
        measurement["runtime"] = "onnx-uint8-marisa-float64-tokenizer"
        measurement["identical_vectors_to_original_uint8"] = measurement.pop("vectors") == original["vectors"]
        measurement["identical_rankings_to_original_uint8"] = measurement.pop("rankings") == original["rankings"]
        measurement["within_400_mib"] = measurement["peak_rss_mib"] <= 400
        measurement["within_original_quality_gate"] = {
            lang: reference["metrics"][lang]["ndcg@8"] > 0 and measurement["metrics"][lang]["ndcg@8"] >=
            0.98 * reference["metrics"][lang]["ndcg@8"] for lang in ("en", "hi", "mr")}
    else:
        print("Token mismatch: rejected, full inference skipped.", flush=True)
    for item in (raw, light):
        item.pop("tokens")
    if (any(digest(work / name) != value for name, value in hashes.items())
            or any(digest(path) != value for path, value in tracked.items())):
        raise RuntimeError("An original artifact changed during the screen")
    write(output / "float64-tokenizer-report.json", {
        "schema_version": 1, "diagnostic_only": True, "script_sha256": digest(__file__), "helper_sha256": HELPER_SHA,
        "source_sha256": hashes, "original_files_unchanged": True, "tokenizer_parity": parity,
        "tokenizer_processes": {"raw": raw, "float64": light}, "full_pipeline": measurement,
        "manifest": read(output / "float64-manifest.json"),
        "packages": {name: importlib.metadata.version(name) for name in ("marisa-trie", "tokenizers")},
        "limitation": "Finite parity is not universal equivalence. Memory-mapped file pages still count toward container memory when resident. The unchanged uint8 model already failed Marathi quality. No serving promotion, upload, or deployment.",
    })
    print("REPORT:", output / "float64-tokenizer-report.json", flush=True)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--work", type=Path, required=True)
    parser.add_argument("--output", type=Path)
    parser.add_argument("--child", choices=("prepare", "raw", "light", "measure"))
    args = parser.parse_args()
    if args.child:
        if args.output is None:
            parser.error("Child stages require --output")
        child(args.work, args.output, args.child)
    else:
        run(args.work.resolve())
