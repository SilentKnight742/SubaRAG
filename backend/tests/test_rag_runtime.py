from __future__ import annotations

import httpx
import numpy as np

from subarag.rag import build_grounded_rag_engine
from subarag.retrieval import Document, build_hnsw_artifact


class FixtureEmbedder:
    dimension = 2

    def __init__(self, model_name, *, batch_size, device):
        self.name = model_name
        assert batch_size == 1
        assert device == "cpu"

    def embed_queries(self, texts):
        return np.asarray([[1, 0] for _text in texts], dtype=np.float32)


def test_runtime_loads_all_verified_indexes_and_selected_generator(tmp_path):
    manifests = {}
    for language in ("en", "hi", "mr"):
        manifests[language] = build_hnsw_artifact(
            tmp_path / language,
            documents=[Document(f"{language}-doc", language, f"Evidence in {language}.")],
            embeddings=np.asarray([[1, 0]], dtype=np.float32),
            corpus_version=f"corpus-{language}",
            model_name="intfloat/multilingual-e5-small",
            retriever_version="e5-small-hnsw-k8-v1",
            language=language,
        )

    client = httpx.AsyncClient(transport=httpx.MockTransport(lambda _request: None))
    engine = build_grounded_rag_engine(
        artifact_manifests=manifests,
        gemini_api_key="test-key",
        generation_client=client,
        gemini_model="gemini-3.5-flash-lite",
        embedder_factory=FixtureEmbedder,
    )

    assert engine.status == "ready"
    assert engine.generator.model == "gemini-3.5-flash-lite"
    result = engine.retriever.retrieve("question", "hin-IN")
    assert result.hits[0].document.id == "hi-doc"
