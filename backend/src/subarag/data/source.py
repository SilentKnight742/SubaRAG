"""Explicit Parquet files avoid the upstream card's stale JSONL loader configs."""

from __future__ import annotations

from collections.abc import Iterator, Sequence
from typing import Any

from .manifest import SourceFile

REPO_ID = "ai4bharat/MSMARCO-XI"
DEFAULT_REVISION = "bf5cdc1f26e581e519018e434db14edd1b77602b"
PREFIXES = {"hi": "hin", "mr": "mar"}
READER_VERSION = "parquet-iter-batches-v1"


def source_files(languages: Sequence[str], splits: Sequence[str]) -> list[SourceFile]:
    # English is repeated in translated files; use Hindi as the English-only anchor.
    translated = [lang for lang in ("hi", "mr") if lang in languages] or ["hi"]
    return [SourceFile(path=f"{split}/{PREFIXES[lang]}{'val' if split == 'validation' else 'train'}.parquet",
                       language=lang, split=split)
            for split in ("validation", "train") if split in splits for lang in translated]


def stream_rows(source: SourceFile, revision: str) -> Iterator[dict[str, Any]]:
    print(f"Reading {source.path} ({READER_VERSION})", flush=True)
    try:
        import aiohttp
        import fsspec
        import pyarrow.parquet as pq
    except ImportError as exc:
        raise RuntimeError("Install the data extra in your cloud runtime: pip install './backend[data]'") from exc
    url = f"https://huggingface.co/datasets/{REPO_ID}/resolve/{revision}/{source.path}"
    # The dataset scanner fails on nested columns above Arrow's chunking limit,
    # even with a small batch size. Decode directly through ParquetFile instead.
    columns = ["source_lang", "target_lang", "query_id", "passages",
               "Eng_Query", "Eng_Answer", "query", "Answer"]
    with fsspec.open(
        url, "rb", block_size=256 * 1024, cache_type="blockcache",
        cache_options={"maxblocks": 8},
        client_kwargs={"trust_env": True, "timeout": aiohttp.ClientTimeout(total=60)},
    ) as handle:
        with pq.ParquetFile(handle, pre_buffer=False, buffer_size=64 * 1024) as parquet:
            for batch in parquet.iter_batches(
                batch_size=32, columns=columns, use_threads=False,
            ):
                yield from batch.to_pylist()
