"""Colab-only lighter-tokenizer screen; no model downloads or serving changes.

Run with --work pointing to the existing compact experiment. Requires
sentencepiece==0.2.1 plus the dependencies already installed by that notebook.
"""

from __future__ import annotations

import argparse
import hashlib
import importlib.metadata
import json
import math
import os
from pathlib import Path
import random
import re
import subprocess
import sys
import tempfile
from time import perf_counter
from types import SimpleNamespace
from unittest.mock import patch


def read(path):
    return json.loads(Path(path).read_text(encoding="utf-8"))


def write(path, value):
    Path(path).write_text(json.dumps(value, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")


def digest(path):
    result = hashlib.sha256()
    with Path(path).open("rb") as handle:
        while block := handle.read(1024 * 1024):
            result.update(block)
    return result.hexdigest()


def memory():
    values = {line.split(":")[0]: int(line.split()[1]) / 1024
              for line in Path("/proc/self/status").read_text().splitlines()
              if line.startswith(("VmRSS:", "VmHWM:"))}
    return {"rss_mib": values["VmRSS"], "peak_mib": values["VmHWM"]}


def convert(work, output):
    # Conversion runs outside the memory measurement process. Preserve IDs,
    # report float32 score rounding, and require exact tokens before inference.
    from sentencepiece import sentencepiece_model_pb2 as pb
    data = read(work / "compact-model/tokenizer.json")
    model = data["model"]
    if model["type"] != "Unigram" or model["unk_id"] != 3 or model.get("byte_fallback", False):
        raise ValueError("Expected the pinned E5 Unigram vocabulary without byte fallback")
    vocab = model["vocab"]
    special = {item["content"]: item for item in data["added_tokens"]}
    expected = {"<s>": 0, "<pad>": 1, "</s>": 2, "<unk>": 3, "<mask>": len(vocab) - 1}
    if set(special) != set(expected):
        raise ValueError("Unexpected added tokens; do not guess their behavior")
    for text, index in expected.items():
        item = special[text]
        if (item["id"] != index or vocab[index][0] != text or not item["special"]
                or item["normalized"] or item["single_word"] or item["rstrip"]
                or not isinstance(item["lstrip"], bool)
                or (text != "<mask>" and item["lstrip"])):
            raise ValueError(f"Unsupported special-token behavior for {text}: {item}")
    if data["post_processor"]["single"] != [
        {"SpecialToken": {"id": "<s>", "type_id": 0}},
        {"Sequence": {"id": "A", "type_id": 0}},
        {"SpecialToken": {"id": "</s>", "type_id": 0}},
    ]:
        raise ValueError("Unsupported single-sequence template")
    proto = pb.ModelProto()
    proto.trainer_spec.model_type = pb.TrainerSpec.UNIGRAM
    proto.trainer_spec.vocab_size = len(vocab)
    proto.trainer_spec.unk_id = 3
    proto.trainer_spec.bos_id = -1
    proto.trainer_spec.eos_id = -1
    proto.trainer_spec.pad_id = -1
    # The original Rust normalizer/pre-tokenizer runs first. SentencePiece must
    # not normalize, add spaces, collapse spaces, or remap token IDs a second time.
    proto.normalizer_spec.name = "identity"
    proto.normalizer_spec.add_dummy_prefix = False
    proto.normalizer_spec.remove_extra_whitespaces = False
    proto.normalizer_spec.escape_whitespaces = False
    rounding = {"target_dtype": "float32", "changed_scores": 0,
                "max_absolute_error": 0.0, "max_relative_error": 0.0, "examples": []}
    for index, (piece, score) in enumerate(vocab):
        if not math.isfinite(score):
            raise ValueError(f"Non-finite source vocabulary score at ID {index}")
        kind = pb.ModelProto.SentencePiece.NORMAL
        if index == 3:
            kind = pb.ModelProto.SentencePiece.UNKNOWN
        stored = proto.pieces.add(piece=piece, score=score, type=kind)
        if not math.isfinite(stored.score):
            raise ValueError(f"Vocabulary score overflows float32 at ID {index}")
        if stored.score != score:
            error = abs(stored.score - score)
            rounding["changed_scores"] += 1
            rounding["max_absolute_error"] = max(rounding["max_absolute_error"], error)
            rounding["max_relative_error"] = max(rounding["max_relative_error"], error / abs(score))
            if len(rounding["examples"]) < 5:
                rounding["examples"].append({"vocabulary_id": index, "source_score": score,
                                             "stored_score": stored.score})
    print(f"Score conversion: {rounding['changed_scores']}/{len(vocab)} rounded to float32; "
          f"maximum absolute change {rounding['max_absolute_error']:.9g}. "
          "Exact token parity is still required.", flush=True)
    (output / "vocabulary.model").write_bytes(proto.SerializeToString())
    # A one-token placeholder retains only the original normalizer and
    # pre-tokenizer. Its model is never used for segmentation.
    pipeline = {**data, "model": {"type": "WordLevel", "vocab": {"[UNK]": 0}, "unk_token": "[UNK]"},
                "added_tokens": [], "post_processor": None, "decoder": None,
                "truncation": None, "padding": None}
    write(output / "pipeline.json", pipeline)
    write(output / "light-manifest.json", {
        "source_tokenizer_sha256": digest(work / "compact-model/tokenizer.json"),
        "special_tokens": expected, "vocabulary_entries": len(vocab),
        "score_conversion": rounding,
        "lstrip_tokens": [text for text, item in special.items() if item["lstrip"]],
        "files": {name: digest(output / name) for name in ("vocabulary.model", "pipeline.json")},
    })


class LightTokenizer:
    def __init__(self, output):
        import sentencepiece as spm
        from tokenizers import Tokenizer
        metadata = read(output / "light-manifest.json")
        for name, checksum in metadata["files"].items():
            if name not in ("vocabulary.model", "pipeline.json") or digest(output / name) != checksum:
                raise ValueError("Light tokenizer checksum mismatch")
        self.pipeline = Tokenizer.from_file(str(output / "pipeline.json"))
        self.processor = spm.SentencePieceProcessor(model_file=str(output / "vocabulary.model"))
        self.special = metadata["special_tokens"]
        self.lstrip_tokens = frozenset(metadata["lstrip_tokens"])
        self.pattern = re.compile("(" + "|".join(re.escape(key) for key in self.special) + ")")
        self.max_length = 512

    def no_padding(self):
        return None

    def enable_truncation(self, *, max_length):
        if max_length != 512:
            raise ValueError("Only the existing 512-token single-query contract is supported")

    def encode(self, text):
        ids = []
        parts = self.pattern.split(text)
        for index, part in enumerate(parts):
            if part in self.special:
                ids.append(self.special[part])
                continue
            if index + 1 < len(parts) and parts[index + 1] in self.lstrip_tokens:
                part = part.rstrip()
            normalized = self.pipeline.normalizer.normalize_str(part) if self.pipeline.normalizer else part
            pieces = self.pipeline.pre_tokenizer.pre_tokenize_str(normalized)
            for piece, _offset in pieces:
                ids.extend(self.processor.encode(piece, out_type=int))
        ids = [0] + ids[:510] + [2]
        return SimpleNamespace(ids=ids, attention_mask=[1] * len(ids))


def cases(work):
    result = [(case["language"] + ":" + case["id"], case["query"])
              for case in read(work / "inputs.json")["cases"]]
    edges = ["", " \t\n", "  hello\tworld\n ", "café cafe\u0301 नमस्ते मराठी 👩🏽‍💻",
             "₹1,234.56 / 2026-10-07 +42%", "<s> </s> <pad> <unk> <mask>",
             "hello  <mask>world <mask> world", "a\x00b\u200bc\u200dd\u00a0e",
             "▁word ▁ ▁hello", "中文 日本語 한국어 العربية", "\uff1c\uff4d\uff41\uff53\uff4b\uff1e"]
    for probe in ("What is a corporation?", "कंपनी क्या है?", "कंपनी म्हणजे काय?"):
        edges.extend((probe * 600, probe + " " * 512, probe * 31, probe * 32))
    for token in ("<s>", "</s>", "<pad>", "<unk>", "<mask>"):
        edges.extend((token, token * 2, "hello" + token + "world", "\t" + token + "\n"))
    result.extend((f"edge:{index}", text) for index, text in enumerate(edges))
    rng = random.Random(742)
    fragments = ["hello", "कंपनी", "मराठी", "é", "e\u0301", "🙂", "<mask>", "</s>",
                 "\u200d", "\x00", "\u00a0", "\u202f", "\u3000", "▁", " ", "\n", "\t", "123", "?!"]
    result.extend((f"fuzz:{index}", "".join(rng.choices(fragments, k=rng.randint(1, 60))))
                  for index in range(200))
    return result


def tokenize(work, output, mode):
    start = memory()
    started = perf_counter()
    if mode == "raw":
        from tokenizers import Tokenizer
        tokenizer = Tokenizer.from_file(str(work / "compact-model/tokenizer.json"))
        tokenizer.no_padding()
        tokenizer.enable_truncation(max_length=512)
    else:
        tokenizer = LightTokenizer(output)
    loaded = memory()
    load_ms = (perf_counter() - started) * 1000
    results = {}
    started = perf_counter()
    for key, query in cases(work):
        encoded = tokenizer.encode("query: " + query)
        results[key] = {"ids": encoded.ids, "attention_mask": encoded.attention_mask}
    write(output / f"{mode}.json", {"tokens": results, "before_load": start, "after_load": loaded,
                                   "after_queries": memory(), "load_ms": load_ms,
                                   "all_queries_ms": (perf_counter() - started) * 1000})
    print(f"{mode}: {len(results)} cases; peak {memory()['peak_mib']:.1f} MiB", flush=True)


def measure(work, output):
    from subarag.experiments import compact_embedding as experiment
    import tokenizers
    light = LightTokenizer(output)
    result = {}

    class Substitute:
        @staticmethod
        def from_file(path):
            if Path(path).resolve() != (work / "compact-model/tokenizer.json").resolve():
                raise ValueError("Unexpected tokenizer load")
            return light

    with patch.object(tokenizers, "Tokenizer", Substitute), \
            patch.object(experiment, "write_json", lambda _path, value: result.update(value)):
        experiment.measure(work, compact=True)
    result["runtime"] = "onnx-uint8-light-tokenizer"
    result["load_order"] = "light tokenizer, API imports, query model, indexes, API startup"
    write(output / "measurement.json", result)


def run(work):
    required = ("inputs.json", "candidate.json", "reference.json", "compact-model/model.onnx",
                "compact-model/tokenizer.json", "compact-model/embedding-manifest.json")
    for name in required:
        if not (work / name).is_file():
            raise ValueError(f"Missing {name}. Keep the existing Colab runtime and WORK directory.")
    hashes = {name: digest(work / name) for name in required}
    metadata = read(work / "compact-model/embedding-manifest.json")
    for name, item in metadata["files"].items():
        if name not in ("model.onnx", "tokenizer.json") or hashes["compact-model/" + name] != item["sha256"]:
            raise ValueError("Original model/tokenizer checksum mismatch")
    original, reference = read(work / "candidate.json"), read(work / "reference.json")
    if any(item["inputs_sha256"] != hashes["inputs.json"] or item["model_revision"] != metadata["model_revision"]
           for item in (original, reference)):
        raise ValueError("Frozen inputs or model revisions differ")
    output = Path(tempfile.mkdtemp(prefix="light-tokenizer-", dir=work))
    print("Output:", output, flush=True)
    env = dict(os.environ, HF_HUB_OFFLINE="1", TRANSFORMERS_OFFLINE="1", TOKENIZERS_PARALLELISM="false",
               OMP_NUM_THREADS="1", OPENBLAS_NUM_THREADS="1")
    env.pop("HF_TOKEN", None)

    def child(mode):
        subprocess.run([sys.executable, "-u", str(Path(__file__).resolve()), "--work", str(work),
                        "--output", str(output), "--child", mode], env=env, check=True)

    for mode in ("convert", "raw", "light"):
        child(mode)
    raw, light = read(output / "raw.json"), read(output / "light.json")
    if set(raw["tokens"]) != set(light["tokens"]):
        raise ValueError("Token comparison case IDs differ")
    mismatches = [key for key in raw["tokens"] if raw["tokens"][key] != light["tokens"][key]]
    parity = {"cases": len(raw["tokens"]), "exact_match": not mismatches, "mismatched_case_ids": mismatches,
              "first_mismatches": [{"case_id": key, "expected": raw["tokens"][key], "actual": light["tokens"][key]}
                                   for key in mismatches[:5]]}
    for item in (raw, light):
        item.pop("tokens")
    measurement = None
    if not mismatches:
        print("Exact token parity passed. Measuring the full unchanged uint8 pipeline…", flush=True)
        child("measure")
        measurement = read(output / "measurement.json")
        if (measurement["inputs_sha256"] != hashes["inputs.json"]
                or set(measurement["vectors"]) != set(original["vectors"])):
            raise ValueError("Inference comparison inputs differ")
        measurement["identical_vectors_to_original_uint8"] = measurement.pop("vectors") == original["vectors"]
        measurement["identical_rankings_to_original_uint8"] = measurement.pop("rankings") == original["rankings"]
        measurement["within_400_mib"] = measurement["peak_rss_mib"] <= 400
        measurement["within_original_quality_gate"] = {
            lang: reference["metrics"][lang]["ndcg@8"] > 0 and measurement["metrics"][lang]["ndcg@8"] >=
            0.98 * reference["metrics"][lang]["ndcg@8"] for lang in ("en", "hi", "mr")}
    else:
        print("Token mismatch: candidate rejected; skipping full inference.", flush=True)
    if any(digest(work / name) != value for name, value in hashes.items()):
        raise RuntimeError("An original input changed during this diagnostic")
    write(output / "light-tokenizer-report.json", {
        "schema_version": 1, "diagnostic_only": True, "script_sha256": digest(__file__),
        "source_sha256": hashes, "original_files_unchanged": True, "tokenizer_parity": parity,
        "tokenizer_processes": {"raw": raw, "light": light}, "full_pipeline": measurement,
        "light_manifest": read(output / "light-manifest.json"),
        "packages": {name: importlib.metadata.version(name) for name in ("sentencepiece", "tokenizers", "protobuf")},
        "limitation": "Finite token parity is not universal equivalence. The original uint8 model already failed Marathi quality; this experiment only changes tokenization. RSS is not a container guarantee. No serving promotion or upload.",
    })
    print("REPORT:", output / "light-tokenizer-report.json", flush=True)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--work", type=Path, required=True)
    parser.add_argument("--output", type=Path)
    parser.add_argument("--child", choices=("convert", "raw", "light", "measure"))
    args = parser.parse_args()
    if args.child:
        if args.output is None:
            parser.error("Child stages require --output")
        if args.child == "convert":
            convert(args.work, args.output)
        elif args.child == "measure":
            measure(args.work, args.output)
        else:
            tokenize(args.work, args.output, args.child)
    else:
        run(args.work.resolve())
