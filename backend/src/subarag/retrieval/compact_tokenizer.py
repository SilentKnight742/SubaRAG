"""Compact Unigram tokenizer with the experimentally verified float64 scoring."""

from __future__ import annotations

import json
import mmap
from pathlib import Path
import re
import struct
from types import SimpleNamespace

from .onnx_embedding import file_sha256

RECORD = "<Id"


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



class CompactTokenizer:
    """Preserve original IDs, normalization, special tokens, and 512-token limit."""

    def __init__(self, directory: Path):
        from tokenizers import Tokenizer
        directory = Path(directory).resolve()
        metadata = json.loads((directory / "float64-manifest.json").read_text())
        if metadata.get("kind") != "marisa-unigram-float64" or metadata.get("record_format") != RECORD:
            raise ValueError("Unsupported compact tokenizer")
        if set(metadata["files"]) != {"vocabulary.marisa", "scores.bin", "pipeline.json"}:
            raise ValueError("Unexpected compact tokenizer files")
        for name, checksum in metadata["files"].items():
            path = (directory / name).resolve()
            if path.parent != directory or file_sha256(path) != checksum:
                raise ValueError("Compact tokenizer checksum mismatch")
        self.source_tokenizer_sha256 = metadata["source_tokenizer_sha256"]
        self.pipeline = Tokenizer.from_file(str(directory / "pipeline.json"))
        self.vocabulary = Vocabulary(directory)
        self.processor = Float64Segmenter(
            self.vocabulary, unknown_id=metadata["unknown_id"],
            unknown_score=metadata["unknown_score"], max_piece_characters=metadata["max_piece_characters"],
        )
        self.special = metadata["special_tokens"]
        self.lstrip_tokens = frozenset(metadata["lstrip_tokens"])
        self.pattern = re.compile("(" + "|".join(re.escape(key) for key in self.special) + ")")
        self.max_length = 512

    def close(self):
        self.vocabulary.records.close()
        self.vocabulary.trie = None

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



