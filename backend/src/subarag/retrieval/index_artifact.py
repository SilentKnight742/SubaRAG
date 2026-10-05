"""Portable, checksum-verified HNSW artifact loaded without corpus re-embedding."""

from __future__ import annotations

import hashlib
import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from .dense import HnswDenseIndex
from .models import Document


SCHEMA_VERSION = 1
MANIFEST_NAME = "retrieval-manifest.json"
DOCUMENTS_NAME = "documents.jsonl"
INDEX_NAME = "index.faiss"


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        while chunk := handle.read(1024 * 1024):
            digest.update(chunk)
    return digest.hexdigest()


@dataclass(frozen=True, slots=True)
class LoadedRetrievalArtifact:
    index: HnswDenseIndex
    corpus_version: str
    model_name: str
    retriever_version: str
    language: str


def build_hnsw_artifact(
    output: str | Path,
    *,
    documents: list[Document],
    embeddings,
    corpus_version: str,
    model_name: str,
    retriever_version: str,
    language: str,
) -> Path:
    """Write data files first and the manifest last so partial builds are invalid."""
    output = Path(output)
    if output.exists() and any(output.iterdir()):
        raise ValueError("Retrieval artifact output must be empty")
    output.mkdir(parents=True, exist_ok=True)

    index = HnswDenseIndex(documents, embeddings)
    documents_path = output / DOCUMENTS_NAME
    with documents_path.open("w", encoding="utf-8") as handle:
        for document in documents:
            handle.write(json.dumps({
                "id": document.id,
                "language": document.language,
                "text": document.text,
                "source": document.source,
            }, ensure_ascii=False, sort_keys=True) + "\n")

    index_path = output / INDEX_NAME
    index.save(index_path)
    manifest: dict[str, Any] = {
        "schema_version": SCHEMA_VERSION,
        "artifact_kind": "subarag_hnsw",
        "corpus_version": corpus_version,
        "retriever_version": retriever_version,
        "model_name": model_name,
        "language": language,
        "document_count": len(documents),
        "dimension": int(index.index.d),
        "files": {
            DOCUMENTS_NAME: {
                "bytes": documents_path.stat().st_size,
                "sha256": _sha256(documents_path),
            },
            INDEX_NAME: {
                "bytes": index_path.stat().st_size,
                "sha256": _sha256(index_path),
            },
        },
    }
    manifest_path = output / MANIFEST_NAME
    manifest_path.write_text(
        json.dumps(manifest, ensure_ascii=False, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    return manifest_path


def load_hnsw_artifact(
    manifest_path: str | Path,
    *,
    expected_model: str | None = None,
    ef_search: int = 64,
) -> LoadedRetrievalArtifact:
    manifest_path = Path(manifest_path).resolve()
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    if manifest.get("schema_version") != SCHEMA_VERSION:
        raise ValueError("Unsupported retrieval artifact schema")
    if manifest.get("artifact_kind") != "subarag_hnsw":
        raise ValueError("Not a SubaRAG HNSW artifact")
    if expected_model is not None and manifest.get("model_name") != expected_model:
        raise ValueError("Retrieval artifact model does not match the configured embedder")

    root = manifest_path.parent
    for name in (DOCUMENTS_NAME, INDEX_NAME):
        path = (root / name).resolve()
        if path.parent != root:
            raise ValueError("Retrieval artifact file escapes its directory")
        metadata = manifest.get("files", {}).get(name, {})
        if not path.is_file() or path.stat().st_size != metadata.get("bytes"):
            raise ValueError(f"Retrieval artifact file is missing or has the wrong size: {name}")
        if _sha256(path) != metadata.get("sha256"):
            raise ValueError(f"Retrieval artifact checksum failed: {name}")

    documents: list[Document] = []
    with (root / DOCUMENTS_NAME).open(encoding="utf-8") as handle:
        for line in handle:
            row = json.loads(line)
            documents.append(Document(
                id=row["id"],
                language=row["language"],
                text=row["text"],
                source=row.get("source", {}),
            ))
    if len(documents) != manifest.get("document_count"):
        raise ValueError("Retrieval artifact document count is invalid")

    index = HnswDenseIndex.load(documents, root / INDEX_NAME, ef_search=ef_search)
    if index.index.d != manifest.get("dimension"):
        raise ValueError("Retrieval artifact embedding dimension is invalid")
    return LoadedRetrievalArtifact(
        index=index,
        corpus_version=manifest["corpus_version"],
        model_name=manifest["model_name"],
        retriever_version=manifest["retriever_version"],
        language=manifest["language"],
    )
