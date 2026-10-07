"""Build the measured RAG serving path from verified immutable artifacts."""

from __future__ import annotations

from collections.abc import Callable, Mapping
from pathlib import Path

import httpx

from ..retrieval import (
    FixedDenseRetriever,
    LanguageRouterRetriever,
    SentenceTransformerEmbedder,
    load_hnsw_artifact,
)
from .engine import GroundedRagEngine
from .providers import GeminiAnswerGenerator


DEFAULT_EMBEDDING_MODEL = "intfloat/multilingual-e5-small"
SUPPORTED_LANGUAGES = ("en", "hi", "mr")


def build_grounded_rag_engine(
    *,
    artifact_manifests: Mapping[str, str | Path],
    gemini_api_key: str,
    generation_client: httpx.AsyncClient,
    gemini_model: str,
    embedding_model: str = DEFAULT_EMBEDDING_MODEL,
    embedder_factory: Callable[..., object] = SentenceTransformerEmbedder,
) -> GroundedRagEngine:
    """Load checksummed indexes and share one query embedder across languages."""
    if set(artifact_manifests) != set(SUPPORTED_LANGUAGES):
        raise ValueError("English, Hindi, and Marathi retrieval manifests are required")

    artifacts = {
        language: load_hnsw_artifact(path, expected_model=embedding_model)
        for language, path in artifact_manifests.items()
    }
    for language, artifact in artifacts.items():
        if artifact.language != language:
            raise ValueError(f"Retrieval artifact language does not match {language}")

    revisions = {artifact.model_revision for artifact in artifacts.values()}
    if len(revisions) != 1:
        raise ValueError("Retrieval artifacts must use the same embedding model revision")
    revision = revisions.pop()
    options = {"revision": revision} if revision is not None else {}
    embedder = embedder_factory(embedding_model, batch_size=1, device="cpu", **options)
    dimensions = {artifact.index.index.d for artifact in artifacts.values()}
    if dimensions != {embedder.dimension}:
        raise ValueError("Retrieval artifacts do not match the embedder dimension")

    routed = LanguageRouterRetriever({
        language: FixedDenseRetriever(
            embedder=embedder,
            dense_index=artifact.index,
            corpus_version=artifact.corpus_version,
            retriever_version=artifact.retriever_version,
            k=8,
        )
        for language, artifact in artifacts.items()
    })
    generator = GeminiAnswerGenerator(
        api_key=gemini_api_key,
        client=generation_client,
        model=gemini_model,
    )
    return GroundedRagEngine(retriever=routed, generator=generator)
