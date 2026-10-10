"""Verified compact query runtime: disk documents and one resident HNSW index."""

from __future__ import annotations

import gc
import json
from pathlib import Path
import re
import sqlite3
from threading import RLock
from time import perf_counter

from .compact_tokenizer import CompactTokenizer
from .controller import LANGUAGE_ALIASES
from .models import Document, RetrievalHit, RetrievalOutcome, RetrievalTelemetry
from .onnx_embedding import OnnxQueryEmbedder, file_sha256

LANGUAGES = ("en", "hi", "mr")
MODEL = "intfloat/multilingual-e5-small"
FILES = {"encoder/" + name for name in ("model.onnx", "tokenizer.json", "embedding-manifest.json")} | {
    "tokenizer/" + name for name in ("float64-manifest.json", "vocabulary.marisa", "scores.bin", "pipeline.json")
} | {f"{lang}/{name}" for lang in LANGUAGES for name in ("index.faiss", "documents.sqlite")}


def verify_compact_bundle(path: Path):
    path = Path(path).resolve()
    if path.stat().st_size > 1024 * 1024:
        raise ValueError("Compact bundle manifest is too large")
    metadata = json.loads(path.read_text())
    if (metadata.get("schema_version") != 1 or metadata.get("artifact_kind") != "subarag_compact_bundle"
            or metadata.get("model_name") != MODEL or metadata.get("dimension") != 384
            or not re.fullmatch(r"[0-9a-f]{40}", metadata.get("model_revision", ""))
            or set(metadata.get("files", {})) != FILES or set(metadata.get("languages", {})) != set(LANGUAGES)):
        raise ValueError("Incompatible compact serving bundle")
    for name, item in metadata["files"].items():
        target = (path.parent / name).resolve()
        if (not target.is_relative_to(path.parent) or target.stat().st_size != item["bytes"]
                or file_sha256(target) != item["sha256"]):
            raise ValueError("Compact serving file checksum mismatch")
    for item in metadata["languages"].values():
        if (not isinstance(item.get("documents"), int) or item["documents"] < 1
                or not isinstance(item.get("corpus_version"), str)):
            raise ValueError("Invalid compact language metadata")
    return metadata


class CompactRetriever:
    """Serialize encoding/index changes; materialize only the returned passages.

    A lock remains held even if an awaiting HTTP request times out. No second
    worker can replace an index underneath an in-flight native search.
    """

    def __init__(self, manifest: str | Path, *, embedder_factory=OnnxQueryEmbedder):
        self.root = Path(manifest).resolve().parent
        self.metadata = verify_compact_bundle(Path(manifest))
        self._lock = RLock()
        self._index = None
        self._language = None
        self._closed = False
        self._connections = {}
        self.embedder = None
        self.tokenizer = CompactTokenizer(self.root / "tokenizer")
        try:
            self.embedder = embedder_factory(self.root / "encoder",
                expected_revision=self.metadata["model_revision"], tokenizer=self.tokenizer, requested_arena=True)
            for lang in LANGUAGES:
                connection = sqlite3.connect((self.root / lang / "documents.sqlite").as_uri() + "?mode=ro&immutable=1",
                                             uri=True, check_same_thread=False)
                self._connections[lang] = connection
                connection.execute("PRAGMA query_only=ON")
                connection.execute("PRAGMA cache_size=-512")
                connection.execute("PRAGMA mmap_size=0")
                count, low, high = connection.execute("SELECT count(*), min(position), max(position) FROM documents").fetchone()
                if (count, low, high) != (self.metadata["languages"][lang]["documents"], 0,
                                         self.metadata["languages"][lang]["documents"] - 1):
                    raise ValueError("Compact document positions differ from index metadata")
        except Exception:
            self.close()
            raise

    @property
    def resident_language(self):
        with self._lock:
            return self._language

    def retrieve(self, query: str, language: str | None = None) -> RetrievalOutcome:
        import faiss
        from .dense import normalized_float32
        with self._lock:
            if self._closed:
                raise RuntimeError("Compact retriever is closed")
            faiss.omp_set_num_threads(1)
            base = language.strip().lower().split("-", 1)[0].split("_", 1)[0] if language else ""
            selected = LANGUAGE_ALIASES.get(base, "en")
            started = perf_counter()
            vector = self.embedder.embed_queries([query])
            if self._language != selected:
                self._index = None  # Release before loading the next language.
                self._language = None
                gc.collect()
                index = faiss.read_index(str(self.root / selected / "index.faiss"))
                if (not hasattr(index, "hnsw") or index.d != 384 or index.metric_type != faiss.METRIC_INNER_PRODUCT
                        or index.ntotal != self.metadata["languages"][selected]["documents"]):
                    raise ValueError("Invalid compact HNSW index")
                index.hnsw.efSearch = 64
                self._index, self._language = index, selected
            scores, positions = self._index.search(normalized_float32(vector), min(8, self._index.ntotal))
            hits = []
            for rank, (position, score) in enumerate(zip(positions[0], scores[0], strict=True), 1):
                if position < 0:
                    continue
                row = self._connections[selected].execute("SELECT payload FROM documents WHERE position=?", (int(position),)).fetchone()
                if row is None:
                    raise ValueError("Indexed document is missing")
                payload = json.loads(row[0])
                if payload["language"] != selected:
                    raise ValueError("Document language differs from its index")
                hits.append(RetrievalHit(Document(payload["id"], selected, payload["text"], payload.get("source", {})),
                                         float(score), rank, "dense"))
            return RetrievalOutcome(tuple(hits), RetrievalTelemetry(
                route="dense", dense_ms=(perf_counter() - started) * 1000, lexical_ms=0, fusion_ms=0,
                selected_k=len(hits), candidate_k=8, selection_policy="fixed", lexical_triggered=False, lexical_reasons=(),
            ))

    def close(self):
        with self._lock:
            if self._closed:
                return
            self._closed = True
            self._index = None
            self._language = None
            for connection in self._connections.values():
                connection.close()
            self._connections.clear()
            if self.embedder is not None:
                self.embedder.close()
            self.tokenizer.close()
