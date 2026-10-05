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
    except ImportError as exc:
        raise RuntimeError("Install the data extra in your cloud runtime: pip install './backend[data]'") from exc
    url = f"https://huggingface.co/datasets/{REPO_ID}/resolve/{revision}/{source.path}"
    # Built-in Parquet reader, no dataset repository code or entire-file download.
    yield from load_dataset("parquet", data_files={source.split: url},
                            split=source.split, streaming=True)
