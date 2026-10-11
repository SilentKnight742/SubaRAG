"""Load prepared artifacts without ever treating evaluation rows as documents."""

from __future__ import annotations

import json
from collections.abc import Iterator
from pathlib import Path

from ..data.manifest import DataManifest, load_manifest
from .models import Document, EvaluationQuery


def _rows(root: Path, manifest: DataManifest, *, kind: str, language: str) -> Iterator[dict]:
    for shard in manifest.shards:
        if shard.kind != kind or shard.language != language:
            continue
        with (root / shard.path).open(encoding="utf-8") as handle:
            for line in handle:
                yield json.loads(line)


def load_language_artifact(
    manifest_path: str | Path,
    language: str,
    *,
    max_documents: int | None = None,
    max_queries: int | None = None,
    split: str = "validation",
) -> tuple[list[Document], list[EvaluationQuery]]:
    """Load one language and retain only queries with relevance inside the selected corpus."""
    manifest_path = Path(manifest_path).resolve()
    manifest = load_manifest(manifest_path, verify_files=True)
    if language not in manifest.languages:
        raise ValueError(f"Language {language!r} is not present in the artifact")
    if max_documents is not None and max_documents < 1:
        raise ValueError("max_documents must be positive")
    if max_queries is not None and max_queries < 1:
        raise ValueError("max_queries must be positive")

    root = manifest_path.parent
    documents: list[Document] = []
    for row in _rows(root, manifest, kind="documents", language=language):
        documents.append(Document(
            id=row["id"], language=row["language"], text=row["text"], source=row.get("source", {}),
        ))
        if max_documents is not None and len(documents) >= max_documents:
            break

    document_ids = {document.id for document in documents}
    queries: list[EvaluationQuery] = []
    for shard in manifest.shards:
        if shard.kind != "evaluation" or shard.language != language or shard.split != split:
            continue
        with (root / shard.path).open(encoding="utf-8") as handle:
            for line in handle:
                row = json.loads(line)
                relevant = frozenset(row["relevant_passage_ids"]) & document_ids
                if not relevant:
                    continue
                queries.append(EvaluationQuery(
                    id=row["id"], language=row["language"], query=row["query"],
                    relevant_passage_ids=relevant,
                    candidate_passage_ids=tuple(row.get("candidate_passage_ids", ())),
                ))
                if max_queries is not None and len(queries) >= max_queries:
                    return documents, queries
    return documents, queries
