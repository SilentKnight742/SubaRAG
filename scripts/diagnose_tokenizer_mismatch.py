"""Offline Colab diagnosis of the failed lighter-tokenizer screen; no inference.

Reuses check_light_tokenizer.py and its completed report/artifacts in WORK.
Only writes a new diagnostic report. Does not change either tokenizer.
"""

from __future__ import annotations

import argparse
from decimal import Decimal
import hashlib
import importlib.util
import json
import os
from pathlib import Path
import struct
import tempfile

HELPER_SHA = "b8b66f5b22ddd80be600400324b0df8b9002de4eb523f1900f76c9cb66b846f8"


def digest(path):
    value = hashlib.sha256()
    with Path(path).open("rb") as handle:
        while chunk := handle.read(1024 * 1024):
            value.update(chunk)
    return value.hexdigest()


def read(path):
    return json.loads(Path(path).read_text(encoding="utf-8"))


def f32(value):
    return struct.unpack("<f", struct.pack("<f", value))[0]


def minimize(text, differs, *, limit=256):
    """Bounded deletion reduction, not a claim of globally shortest input."""
    evaluations, chunks = 0, 2
    while len(text) > 1 and evaluations < limit:
        width = max(1, (len(text) + chunks - 1) // chunks)
        reduced = False
        for start in range(0, len(text), width):
            candidate = text[:start] + text[start + width:]
            if not candidate:
                continue
            evaluations += 1
            if differs(candidate):
                text, chunks, reduced = candidate, max(2, chunks - 1), True
                break
            if evaluations >= limit:
                break
        if not reduced:
            if chunks >= len(text):
                break
            chunks = min(len(text), chunks * 2)
    return text, evaluations


def core_ids(raw, light, text):
    return ([token.id for token in raw.model.tokenize(text)],
            light.processor.encode(text, out_type=int))


def subset_comparison(text, raw, light, vocab):
    """Retain every possible vocabulary match plus unknown/min-score entries.

    IDs are mapped back to the full vocabulary. Agreement with both full models
    is reported explicitly before interpreting the reduced example.
    """
    from tokenizers.models import Unigram
    import sentencepiece as spm
    from sentencepiece import sentencepiece_model_pb2 as pb

    selected = {3, min(range(len(vocab)), key=lambda index: vocab[index][1])}
    for start in range(len(text)):
        for end in range(start + 1, len(text) + 1):
            index = raw.token_to_id(text[start:end])
            if index is not None:
                selected.add(index)
    global_ids = sorted(selected)
    entries = [tuple(vocab[index]) for index in global_ids]
    unknown = global_ids.index(3)
    rounded = [(piece, f32(score)) for piece, score in entries]

    def rust_ids(values):
        model = Unigram(values, unk_id=unknown, byte_fallback=False)
        return [global_ids[token.id] for token in model.tokenize(text)]

    proto = pb.ModelProto()
    proto.trainer_spec.model_type = pb.TrainerSpec.UNIGRAM
    proto.trainer_spec.vocab_size = len(entries)
    proto.trainer_spec.unk_id = unknown
    proto.trainer_spec.bos_id = -1
    proto.trainer_spec.eos_id = -1
    proto.trainer_spec.pad_id = -1
    proto.normalizer_spec.name = "identity"
    proto.normalizer_spec.add_dummy_prefix = False
    proto.normalizer_spec.remove_extra_whitespaces = False
    proto.normalizer_spec.escape_whitespaces = False
    for index, (piece, score) in enumerate(rounded):
        proto.pieces.add(piece=piece, score=score,
                         type=pb.ModelProto.SentencePiece.UNKNOWN if index == unknown
                         else pb.ModelProto.SentencePiece.NORMAL)
    native = spm.SentencePieceProcessor(model_proto=proto.SerializeToString())
    native_ids = [global_ids[index] for index in native.encode(text, out_type=int)]
    original_ids, full_native_ids = core_ids(raw, light, text)
    exact_ids, rounded_ids = rust_ids(entries), rust_ids(rounded)
    return {
        "text": text, "vocabulary": [{"id": index, "piece": vocab[index][0],
                                       "source_score": vocab[index][1], "float32_score": f32(vocab[index][1])}
                                      for index in global_ids],
        "full_raw_ids": original_ids, "full_native_ids": full_native_ids,
        "subset_raw_original_scores_ids": exact_ids, "subset_raw_float32_scores_ids": rounded_ids,
        "subset_native_ids": native_ids,
        "subset_reproduces_both_full_models": exact_ids == original_ids and native_ids == full_native_ids,
        "rounding_scores_changes_raw_output": exact_ids != rounded_ids,
        "same_stored_scores_still_different": rounded_ids != native_ids,
    }


def path_details(ids, vocab):
    source = [vocab[index][1] for index in ids]
    total32, total64 = 0.0, 0.0
    cumulative32, cumulative64 = [], []
    for score in source:
        total32 = f32(f32(score) + total32)
        total64 = score + total64
        cumulative32.append(total32)
        cumulative64.append(total64)
    return {"ids": ids, "pieces": [vocab[index][0] for index in ids], "scores": source,
            "decimal_sum_of_stored_source_values": str(sum((Decimal.from_float(score) for score in source), Decimal(0))),
            "cumulative_float32": cumulative32, "cumulative_float64": cumulative64,
            "note": "These are path sums, not a replay of every competing Viterbi update. Unknown-token penalties are not modeled here."}


def run(work):
    # Reuse the exact helper that produced the submitted report, including when
    # Colab renamed an upload. Never silently import a different local script.
    helper_path = next((path for path in Path(__file__).resolve().parent.glob("check_light_tokenizer*.py")
                        if digest(path) == HELPER_SHA), None)
    if helper_path is None:
        raise ValueError("Keep the latest check_light_tokenizer.py upload in the same Colab folder.")
    spec = importlib.util.spec_from_file_location("verified_light_helper", helper_path)
    helper = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(helper)
    completed = []
    for path in work.glob("light-tokenizer-*/light-tokenizer-report.json"):
        value = read(path)
        if value.get("script_sha256") == HELPER_SHA and value.get("tokenizer_parity", {}).get("mismatched_case_ids"):
            completed.append(path)
    if not completed:
        raise ValueError("No completed mismatching report from the expected helper was found.")
    report_path = max(completed, key=lambda path: path.stat().st_mtime)
    prior = read(report_path)
    source = work / "compact-model/tokenizer.json"
    tracked = [source, work / "inputs.json", report_path,
               report_path.parent / "light-manifest.json", report_path.parent / "pipeline.json",
               report_path.parent / "vocabulary.model"]
    original_hashes = {str(path): digest(path) for path in tracked}
    for name in ("inputs.json", "compact-model/tokenizer.json"):
        if original_hashes[str(work / name)] != prior["source_sha256"][name]:
            raise ValueError("Saved source differs from the failed run")
    from tokenizers import Tokenizer
    raw = Tokenizer.from_file(str(source))
    raw.no_padding()
    raw.enable_truncation(max_length=512)
    light = helper.LightTokenizer(report_path.parent)
    vocab = read(source)["model"]["vocab"]
    inputs = dict(helper.cases(work))
    def differs(text):
        left, right = core_ids(raw, light, text)
        return left != right

    results = []
    for case_id in prior["tokenizer_parity"]["mismatched_case_ids"]:
        # Only synthetic cases are exported; dataset query text stays private.
        if not case_id.startswith(("fuzz:", "edge:")):
            raise ValueError("This diagnostic only exports synthetic mismatches")
        text = "query: " + inputs[case_id]
        raw_ids, light_ids = raw.encode(text).ids, light.encode(text).ids
        if raw_ids == light_ids:
            raise ValueError("Saved mismatch did not reproduce; stop before attributing a cause")
        pieces, assembled_raw, assembled_light = [], [], []
        parts = light.pattern.split(text)
        for index, part in enumerate(parts):
            if part in light.special:
                assembled_raw.append(light.special[part])
                assembled_light.append(light.special[part])
                continue
            if index + 1 < len(parts) and parts[index + 1] in light.lstrip_tokens:
                part = part.rstrip()
            normalized = light.pipeline.normalizer.normalize_str(part) if light.pipeline.normalizer else part
            for piece, _ in light.pipeline.pre_tokenizer.pre_tokenize_str(normalized):
                left, right = core_ids(raw, light, piece)
                assembled_raw.extend(left)
                assembled_light.extend(right)
                if left != right:
                    reduced, calls = minimize(piece, differs)
                    if len(reduced) > 256:
                        raise ValueError("Reduced segment is too long for a bounded subset diagnostic")
                    pieces.append({"pretokenized_text": piece, "raw_path": path_details(left, vocab),
                                   "native_path": path_details(right, vocab), "reduction_evaluations": calls,
                                   "reduced_example": subset_comparison(reduced, raw, light, vocab)})
        results.append({"case_id": case_id, "synthetic_input": text,
                        "raw_ids": raw_ids, "light_ids": light_ids,
                        "raw_core_reconstruction_matches": [0] + assembled_raw[:510] + [2] == raw_ids,
                        "light_core_reconstruction_matches": [0] + assembled_light[:510] + [2] == light_ids,
                        "differing_segments": pieces})
    if any(digest(path) != value for path, value in original_hashes.items()):
        raise RuntimeError("An original tokenizer/report changed during diagnosis")
    output = Path(tempfile.mkdtemp(prefix="tokenizer-mismatch-", dir=work))
    helper.write(output / "tokenizer-mismatch-report.json", {
        "schema_version": 1, "diagnostic_only": True, "script_sha256": digest(__file__),
        "helper_sha256": HELPER_SHA, "previous_report_sha256": original_hashes[str(report_path)],
        "source_sha256": original_hashes, "original_files_unchanged": True,
        "cases": results,
        "limitation": "Synthetic tokenizer-only diagnosis, not a memory benchmark or replacement adapter. Path sums do not establish the full Viterbi decision trace. No model/index loading, network, upload, or deployment.",
    })
    print("Reproduced", len(results), "synthetic mismatch(es).", flush=True)
    print("REPORT:", output / "tokenizer-mismatch-report.json", flush=True)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--work", type=Path, required=True)
    args = parser.parse_args()
    os.environ["HF_HUB_OFFLINE"] = "1"
    os.environ["TRANSFORMERS_OFFLINE"] = "1"
    run(args.work.resolve())
