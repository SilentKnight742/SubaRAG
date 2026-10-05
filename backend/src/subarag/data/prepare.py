"""Bounded-memory corpus preparation, with disk-backed passage/query deduplication."""

from __future__ import annotations

import hashlib
import json
import re
import sqlite3
import unicodedata
from collections.abc import Callable, Iterable, Sequence
from datetime import datetime, timezone
from itertools import islice
from pathlib import Path
from typing import Any

from .manifest import Counts, DataManifest, Shard, Source, SourceFile
from .source import DEFAULT_REVISION, REPO_ID, source_files, stream_rows

RowReader = Callable[[SourceFile, str], Iterable[dict[str, Any]]]
TARGET_LANGUAGES = {"hi": "hin_Deva", "mr": "mar_Deva"}


def normalize_text(value: Any) -> str:
    if not isinstance(value, str):
        raise ValueError("Expected text")
    text = " ".join(unicodedata.normalize("NFC", value).split())
    if not text:
        raise ValueError("Empty text")
    return text


def passage_id(language: str, text: str) -> str:
    content = f"subarag-passage-v1\0{language}\0{normalize_text(text)}"
    return "p_" + hashlib.sha256(content.encode("utf-8")).hexdigest()


def _query_id(row: dict[str, Any]) -> str:
    value = row.get("query_id")
    if isinstance(value, bool) or not isinstance(value, (str, int)) or not str(value).isdigit():
        raise ValueError("query_id must be a nonnegative integer")
    return str(int(value))


def _validate_row(row: dict[str, Any], source: SourceFile) -> tuple[str, list[str], list[str], list[int]]:
    query = _query_id(row)
    if row.get("source_lang") != "eng_Latn" or row.get("target_lang") != TARGET_LANGUAGES[source.language]:
        raise ValueError("Unexpected source or target language")
    passages = row.get("passages")
    if isinstance(passages, str):
        try:
            passages = json.loads(passages)
        except json.JSONDecodeError as exc:
            raise ValueError("passages must contain valid JSON") from exc
    if not isinstance(passages, dict):
        raise ValueError("passages must be an object")
    english, translated, selected = (passages.get(key) for key in
                                      ("English_passages", "Translated_passages", "is_selected"))
    if not all(isinstance(value, list) for value in (english, translated, selected)):
        raise ValueError("Passage fields must be lists")
    if not english or not len(english) == len(translated) == len(selected):
        raise ValueError("Passage and relevance arrays must have equal, nonzero lengths")
    if any(type(value) is not int or value not in (0, 1) for value in selected):
        raise ValueError("Relevance flags must be integers 0 or 1")
    return query, [normalize_text(text) for text in english], [normalize_text(text) for text in translated], selected


def _answer(value: Any) -> list[str]:
    # Upstream uses scalar Answer/Eng_Answer; lists are accepted for schema-compatible exports.
    if value is None or value == "":
        return []
    values = value if isinstance(value, list) else [value]
    return [normalize_text(answer) for answer in values]


class ShardWriter:
    def __init__(self, output: Path, max_rows: int):
        self.output, self.max_rows = output, max_rows
        self.active: dict[tuple[str, str | None, str | None], dict[str, Any]] = {}
        self.indices: dict[tuple[str, str | None, str | None], int] = {}
        self.shards: list[Shard] = []

    def write(self, kind: str, row: dict[str, Any], language: str | None = None, split: str | None = None) -> None:
        key = (kind, language, split)
        if key not in self.active:
            index = self.indices.get(key, 0)
            name = "-".join(part for part in (language or "source", split) if part)
            path = f"{kind}/{name}-{index:05d}.jsonl"
            destination = self.output / path
            destination.parent.mkdir(parents=True, exist_ok=True)
            self.active[key] = {"path": path, "file": destination.open("xb"), "hash": hashlib.sha256(), "rows": 0, "bytes": 0}
            self.indices[key] = index + 1
        current = self.active[key]
        line = (json.dumps(row, ensure_ascii=False, sort_keys=True, separators=(",", ":")) + "\n").encode("utf-8")
        current["file"].write(line)
        current["hash"].update(line)
        current["rows"] += 1
        current["bytes"] += len(line)
        if current["rows"] >= self.max_rows:
            self._finish(key)

    def _finish(self, key: tuple[str, str | None, str | None]) -> None:
        current = self.active.pop(key)
        current["file"].close()
        self.shards.append(Shard(path=current["path"], kind=key[0], language=key[1], split=key[2],
                                 rows=current["rows"], bytes=current["bytes"], sha256=current["hash"].hexdigest()))

    def close(self) -> None:
        for key in list(self.active):
            self._finish(key)


def prepare_corpus(
    output: Path,
    *,
    languages: Sequence[str] = ("en", "hi", "mr"),
    splits: Sequence[str] = ("train", "validation"),
    revision: str = DEFAULT_REVISION,
    rows_per_source: int | None = 1000,
    shard_rows: int = 50_000,
    reader: RowReader = stream_rows,
) -> DataManifest:
    """Prepare an immutable artifact. Failed runs have no completion manifest; use a fresh output directory."""
    if not re.fullmatch(r"[0-9a-f]{40}", revision):
        raise ValueError("Pin source revision to a full 40-character commit SHA")
    if not languages or not set(languages) <= {"en", "hi", "mr"} or len(set(languages)) != len(languages):
        raise ValueError("Choose distinct supported languages: en hi mr")
    if not splits or not set(splits) <= {"train", "validation"} or len(set(splits)) != len(splits):
        raise ValueError("Choose distinct splits: train validation")
    if rows_per_source is not None and (type(rows_per_source) is not int or rows_per_source < 1):
        raise ValueError("rows_per_source must be positive or None for an explicit full run")
    if type(shard_rows) is not int or shard_rows < 1:
        raise ValueError("shard_rows must be positive")
    output = Path(output)
    output.mkdir(parents=True, exist_ok=True)
    if any(output.iterdir()):
        raise ValueError("Output directory must be empty; existing artifacts are never overwritten")
    sources = source_files(languages, splits)
    writer = ShardWriter(output, shard_rows)
    counts = dict(source_rows=0, accepted_rows=0, rejected_rows=0, documents=0, evaluations=0,
                  duplicate_documents=0, duplicate_evaluations=0, held_out_train_evaluations=0)
    language_counts = {language: 0 for language in languages}
    work = output / "_work"
    work.mkdir()
    database = sqlite3.connect(work / "dedup.sqlite3")
    # SQLite stays on disk; its bounded cache is independent of corpus size.
    database.execute("PRAGMA cache_size = -8192")
    database.execute("PRAGMA temp_store = FILE")
    database.executescript("""
        CREATE TABLE documents (id TEXT PRIMARY KEY) WITHOUT ROWID;
        CREATE TABLE evaluations (language TEXT, query_id TEXT, PRIMARY KEY(language, query_id)) WITHOUT ROWID;
        CREATE TABLE validation_queries (query_id TEXT PRIMARY KEY) WITHOUT ROWID;
    """)
    try:
        for source in sources:
            stream = reader(source, revision)
            rows = stream if rows_per_source is None else islice(stream, rows_per_source)
            for row_index, row in enumerate(rows):
                counts["source_rows"] += 1
                location = {"file": source.path, "row": row_index, "split": source.split}
                try:
                    # Reserve validation IDs even when another field in the row is invalid.
                    query_id = _query_id(row)
                    if source.split == "validation":
                        database.execute("INSERT OR IGNORE INTO validation_queries VALUES (?)", (query_id,))
                    query_id, english, translated, selected = _validate_row(row, source)
                    prepared = []
                    for language, passages, query_field, answer_field in (
                        ("en", english, "Eng_Query", "Eng_Answer"),
                        (source.language, translated, "query", "Answer"),
                    ):
                        if language in languages:
                            prepared.append((language, passages, normalize_text(row.get(query_field)), _answer(row.get(answer_field))))
                except (ValueError, TypeError, AttributeError) as exc:
                    counts["rejected_rows"] += 1
                    writer.write("rejections", {"source": location, "reason": str(exc)})
                    continue
                counts["accepted_rows"] += 1
                held_out = source.split == "train" and database.execute(
                    "SELECT 1 FROM validation_queries WHERE query_id = ?", (query_id,)).fetchone() is not None
                for language, passages, query, answers in prepared:
                    ids = [passage_id(language, text) for text in passages]
                    for passage_index, (identifier, text) in enumerate(zip(ids, passages, strict=True)):
                        inserted = database.execute("INSERT OR IGNORE INTO documents VALUES (?)", (identifier,)).rowcount
                        if not inserted:
                            counts["duplicate_documents"] += 1
                            continue
                        writer.write("documents", {"id": identifier, "language": language, "text": text,
                                                    "source": {**location, "passage_index": passage_index}}, language)
                        counts["documents"] += 1
                        language_counts[language] += 1
                    if held_out:
                        counts["held_out_train_evaluations"] += 1
                        continue
                    inserted = database.execute("INSERT OR IGNORE INTO evaluations VALUES (?, ?)", (language, query_id)).rowcount
                    if not inserted:
                        counts["duplicate_evaluations"] += 1
                        continue
                    writer.write("evaluation", {
                        "id": f"q_{language}_{query_id}", "query_id": query_id, "language": language,
                        "split": source.split, "query": query, "answers": answers,
                        "candidate_passage_ids": list(dict.fromkeys(ids)),
                        "relevant_passage_ids": list(dict.fromkeys(identifier for identifier, relevance in zip(ids, selected, strict=True) if relevance)),
                        "source": location,
                    }, language, source.split)
                    counts["evaluations"] += 1
                if counts["source_rows"] % 1000 == 0:
                    database.commit()
            database.commit()
        writer.close()
        manifest = DataManifest(
            created_at=datetime.now(timezone.utc).isoformat(),
            source=Source(repo_id=REPO_ID, revision=revision, files=sources),
            languages=list(languages), splits=list(splits), sample_rows_per_source=rows_per_source,
            counts=Counts(**counts), document_counts_by_language=language_counts,
            shards=sorted(writer.shards, key=lambda shard: shard.path),
        )
        # Publish a complete manifest only after all shards are closed and validated.
        temporary_manifest = output / "manifest.json.tmp"
        temporary_manifest.write_text(manifest.model_dump_json(indent=2) + "\n", encoding="utf-8")
        temporary_manifest.replace(output / "manifest.json")
        return manifest
    finally:
        writer.close()
        database.close()
