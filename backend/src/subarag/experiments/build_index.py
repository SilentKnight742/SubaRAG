"""Build an immutable serving index from a validated prepared corpus."""

from __future__ import annotations

import argparse
import hashlib
import re
from pathlib import Path

from ..data.manifest import verify_manifest
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
    value.add_argument("--model-revision", help="Immutable 40-character model commit SHA")
    value.add_argument("--device", choices=("cpu", "cuda"), default="cpu")
    value.add_argument("--batch-size", type=int, default=32)
    value.add_argument("--documents", type=int)
    return value


def main() -> None:
    args = parser().parse_args()
    if args.batch_size < 1:
        raise ValueError("Batch size must be positive")
    if args.model_revision is not None and not re.fullmatch(r"[0-9a-f]{40}", args.model_revision):
        raise ValueError("Model revision must be a full commit SHA")
    print(f"Verifying prepared corpus for {args.language}…", flush=True)
    prepared = verify_manifest(args.manifest)
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
    if not documents:
        raise ValueError("The selected corpus contains no documents")
    print(f"Embedding {len(documents):,} {args.language} passages on {args.device}…", flush=True)
    embedder = SentenceTransformerEmbedder(
        args.model, batch_size=args.batch_size, device=args.device, revision=args.model_revision,
    )
    embeddings = embedder.embed_documents([document.text for document in documents])
    print("Writing HNSW index and checksums…", flush=True)
    manifest = build_hnsw_artifact(
        args.output,
        documents=documents,
        embeddings=embeddings,
        corpus_version=corpus_version,
        model_name=args.model,
        retriever_version="e5-small-hnsw-k8-v1" + (
            f"-{args.model_revision}" if args.model_revision else ""
        ),
        language=args.language,
        model_revision=args.model_revision,
    )
    print(manifest.resolve())


if __name__ == "__main__":
    main()
