"""Small, versioned artifact contract usable by the application status endpoint."""

from __future__ import annotations

import hashlib
import json
from pathlib import Path, PurePosixPath
from typing import Literal

from pydantic import BaseModel, ConfigDict, Field, model_validator

Language = Literal["en", "hi", "mr"]
Split = Literal["train", "validation"]


class Contract(BaseModel):
    model_config = ConfigDict(extra="forbid", strict=True)


class SourceFile(Contract):
    path: str
    language: Literal["hi", "mr"]
    split: Split


class Source(Contract):
    repo_id: Literal["ai4bharat/MSMARCO-XI"]
    revision: str = Field(pattern=r"^[0-9a-f]{40}$")
    files: list[SourceFile] = Field(min_length=1)


class Counts(Contract):
    source_rows: int = Field(ge=0)
    accepted_rows: int = Field(ge=0)
    rejected_rows: int = Field(ge=0)
    documents: int = Field(gt=0)
    evaluations: int = Field(ge=0)
    duplicate_documents: int = Field(ge=0)
    duplicate_evaluations: int = Field(ge=0)
    held_out_train_evaluations: int = Field(ge=0)


class Shard(Contract):
    path: str
    kind: Literal["documents", "evaluation", "rejections"]
    language: Language | None = None
    split: Split | None = None
    rows: int = Field(gt=0)
    bytes: int = Field(gt=0)
    sha256: str = Field(pattern=r"^[0-9a-f]{64}$")

    @model_validator(mode="after")
    def validate_path(self) -> Shard:
        path = PurePosixPath(self.path)
        if path.is_absolute() or ".." in path.parts or "\\" in self.path:
            raise ValueError("Shard paths must be relative paths inside the artifact")
        if not path.parts or self.path != path.as_posix() or path.suffix != ".jsonl":
            raise ValueError("Shard paths must be normalized JSONL paths")
        if self.kind == "documents" and (self.language is None or self.split is not None):
            raise ValueError("Documents require a language and have no query split")
        if self.kind == "evaluation" and (self.language is None or self.split is None):
            raise ValueError("Evaluation shards require a language and a split")
        if self.kind == "rejections" and (self.language is not None or self.split is not None):
            raise ValueError("Rejections contain source locations rather than a language split")
        return self


class Preparation(Contract):
    version: Literal["1"] = "1"
    normalization: Literal["unicode-nfc-whitespace-v1"] = "unicode-nfc-whitespace-v1"
    passage_id: Literal["sha256-language-normalized-text-v1"] = "sha256-language-normalized-text-v1"
    query_policy: Literal["validation-first-query-id"] = "validation-first-query-id"
    document_policy: Literal["all-candidate-passages-no-queries-or-answers"] = "all-candidate-passages-no-queries-or-answers"


class DataManifest(Contract):
    schema_version: Literal[1] = 1
    artifact_kind: Literal["prepared_corpus"] = "prepared_corpus"
    created_at: str
    source: Source
    languages: list[Language] = Field(min_length=1)
    splits: list[Split] = Field(min_length=1)
    sample_rows_per_source: int | None = Field(default=None, gt=0)
    preparation: Preparation = Field(default_factory=Preparation)
    counts: Counts
    document_counts_by_language: dict[Language, int]
    shards: list[Shard] = Field(min_length=1)

    @model_validator(mode="after")
    def consistent_counts(self) -> DataManifest:
        if len(set(self.languages)) != len(self.languages) or len(set(self.splits)) != len(self.splits):
            raise ValueError("Languages and splits must be unique")
        if len({shard.path for shard in self.shards}) != len(self.shards):
            raise ValueError("Shard paths must be unique")
        if self.counts.source_rows != self.counts.accepted_rows + self.counts.rejected_rows:
            raise ValueError("Source row counts are inconsistent")
        totals = {kind: sum(s.rows for s in self.shards if s.kind == kind)
                  for kind in ("documents", "evaluation", "rejections")}
        if totals != {"documents": self.counts.documents, "evaluation": self.counts.evaluations,
                      "rejections": self.counts.rejected_rows}:
            raise ValueError("Shard row counts are inconsistent")
        actual = {lang: sum(s.rows for s in self.shards if s.kind == "documents" and s.language == lang)
                  for lang in self.languages}
        if actual != self.document_counts_by_language or any(count <= 0 for count in actual.values()):
            raise ValueError("Every selected language must have a consistent positive document count")
        if any(s.language is not None and s.language not in self.languages for s in self.shards):
            raise ValueError("Unexpected shard language")
        if any(s.split is not None and s.split not in self.splits for s in self.shards):
            raise ValueError("Unexpected shard split")
        return self


def load_manifest(path: str | Path, *, verify_files: bool = False) -> DataManifest:
    """Validate metadata; optionally stat shards, without reading corpus contents."""
    path = Path(path)
    if path.stat().st_size > 1_048_576:
        raise ValueError("Manifest exceeds the 1 MiB metadata limit")
    manifest = DataManifest.model_validate(json.loads(path.read_text(encoding="utf-8")))
    if verify_files:
        root = path.resolve().parent
        for shard in manifest.shards:
            artifact_path = (root / shard.path).resolve()
            if not artifact_path.is_relative_to(root):
                raise ValueError("Shard resolves outside the artifact directory")
            if not artifact_path.is_file() or artifact_path.stat().st_size != shard.bytes:
                raise ValueError(f"Shard missing or size mismatch: {shard.path}")
    return manifest


def verify_manifest(path: str | Path) -> DataManifest:
    """Stream local shards to verify byte counts, line counts, and checksums."""
    manifest = load_manifest(path)
    root = Path(path).resolve().parent
    for shard in manifest.shards:
        artifact_path = (root / shard.path).resolve()
        if not artifact_path.is_relative_to(root):
            raise ValueError("Shard resolves outside the artifact directory")
        digest = hashlib.sha256()
        rows = size = 0
        with artifact_path.open("rb") as handle:
            for line in handle:
                rows += 1
                size += len(line)
                digest.update(line)
        if (rows, size, digest.hexdigest()) != (shard.rows, shard.bytes, shard.sha256):
            raise ValueError(f"Shard verification failed: {shard.path}")
    return manifest
