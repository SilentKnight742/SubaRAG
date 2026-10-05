"""Build an immutable serving index from a validated prepared corpus."""

from __future__ import annotations

import argparse
import hashlib
from pathlib import Path

from ..data.manifest import load_manifest
from ..retrieval import (
    SentenceTransformerEmbedder,
    build_hnsw_artifact,
    load_language_artifact,
)


def parser() -> argparse.ArgumentParser:
    value = argparse.ArgumentParser(description="Build a SubaRAG HNSW serving artifact")
    value.add_argument("--manifest", type=Path, required=True)
    value.add_argument("--output", type=Path, required=True)
    value.add_argument("--language", choices=("en", "hi", "mr"), required=True)
    value.add_argument("--model", default="intfloat/multilingual-e5-small")
    value.add_argument("--batch-size", type=int, default=32)
    value.add_argument("--documents", type=int)
    return value


def main() -> None:
    args = parser().parse_args()
    prepared = load_manifest(args.manifest, verify_files=True)
    documents, _ = load_language_artifact(
        args.manifest,
        args.language,
        max_documents=args.documents,
        max_queries=1,
    )
    digest = hashlib.sha256()
    for document in documents:
        digest.update(document.id.encode())
        digest.update(b"\0")
    corpus_version = f"{prepared.source.revision[:12]}-{digest.hexdigest()[:12]}"
    embedder = SentenceTransformerEmbedder(args.model, batch_size=args.batch_size)
    embeddings = embedder.embed_documents([document.text for document in documents])
    manifest = build_hnsw_artifact(
        args.output,
        documents=documents,
        embeddings=embeddings,
        corpus_version=corpus_version,
        model_name=args.model,
        retriever_version="e5-small-hnsw-k8-v1",
        language=args.language,
    )
    print(manifest.resolve())


if __name__ == "__main__":
    main()
