"""Measured, replaceable retrieval components used by SubaRAG."""

from .artifact import load_language_artifact
from .cache import BoundedCache, embedding_cache_key, retrieval_cache_key
from .compression import extract_supporting_spans
from .controller import AdaptiveRetriever, FixedDenseRetriever, LanguageRouterRetriever, Retriever
from .dense import ExactDenseIndex, HnswDenseIndex
from .embedding import Embedder, SentenceTransformerEmbedder
from .index_artifact import (
    LoadedRetrievalArtifact,
    build_hnsw_artifact,
    load_hnsw_artifact,
)
from .lexical import Bm25SparseIndex, reciprocal_rank_fusion, unicode_tokens
from .models import Document, EvaluationQuery, RetrievalHit, RetrievalOutcome
from .metrics import latency_summary, ranking_metrics
from .policies import route_lexical, select_adaptive_k

__all__ = [
    "AdaptiveRetriever", "Bm25SparseIndex", "BoundedCache", "Document", "Embedder",
    "EvaluationQuery", "ExactDenseIndex", "HnswDenseIndex", "RetrievalHit",
    "RetrievalOutcome", "Retriever", "FixedDenseRetriever", "LanguageRouterRetriever",
    "LoadedRetrievalArtifact",
    "SentenceTransformerEmbedder", "embedding_cache_key",
    "build_hnsw_artifact", "extract_supporting_spans", "load_hnsw_artifact",
    "load_language_artifact", "reciprocal_rank_fusion",
    "retrieval_cache_key", "route_lexical", "select_adaptive_k", "unicode_tokens",
    "latency_summary", "ranking_metrics",
]
