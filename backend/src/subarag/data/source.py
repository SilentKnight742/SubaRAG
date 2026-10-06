"""Explicit Parquet files avoid the upstream card's stale JSONL loader configs."""

from __future__ import annotations

from collections.abc import Iterator, Sequence
from typing import Any

from .manifest import SourceFile

REPO_ID = "ai4bharat/MSMARCO-XI"
DEFAULT_REVISION = "bf5cdc1f26e581e519018e434db14edd1b77602b"
PREFIXES = {"hi": "hin", "mr": "mar"}


def source_files(languages: Sequence[str], splits: Sequence[str]) -> list[SourceFile]:
    # English is repeated in translated files; use Hindi as the English-only anchor.
    translated = [lang for lang in ("hi", "mr") if lang in languages] or ["hi"]
    return [SourceFile(path=f"{split}/{PREFIXES[lang]}{'val' if split == 'validation' else 'train'}.parquet",
                       language=lang, split=split)
            for split in ("validation", "train") if split in splits for lang in translated]


def stream_rows(source: SourceFile, revision: str) -> Iterator[dict[str, Any]]:
    try:
        from datasets import load_dataset
        from pyarrow.dataset import ParquetFragmentScanOptions
    except ImportError as exc:
        raise RuntimeError("Install the data extra in your cloud runtime: pip install './backend[data]'") from exc
    url = f"https://huggingface.co/datasets/{REPO_ID}/resolve/{revision}/{source.path}"
    # Source files have giant row groups. Small record batches alone do not stop
    # Arrow from eagerly buffering their column chunks before yielding a row.
    scan_options = ParquetFragmentScanOptions(
        pre_buffer=False, use_buffered_stream=True, buffer_size=64 * 1024,
    )
    print(f"Reading {source.path} (32-row batches, buffered reads)", flush=True)
    yield from load_dataset(
        "parquet", data_files={source.split: url}, split=source.split,
        streaming=True, batch_size=32, fragment_scan_options=scan_options,
        columns=["source_lang", "target_lang", "query_id", "passages",
                 "Eng_Query", "Eng_Answer", "query", "Answer"],
    )
