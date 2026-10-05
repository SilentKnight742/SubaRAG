"""Bounded, version-aware caches; answer caching intentionally does not live here."""

from __future__ import annotations

from collections import OrderedDict
from collections.abc import Hashable
from threading import Lock
from typing import Generic, TypeVar


Value = TypeVar("Value")


class BoundedCache(Generic[Value]):
    def __init__(self, max_entries: int = 1024) -> None:
        if max_entries < 1:
            raise ValueError("max_entries must be positive")
        self.max_entries = max_entries
        self._values: OrderedDict[Hashable, Value] = OrderedDict()
        self._lock = Lock()

    def get(self, key: Hashable) -> Value | None:
        with self._lock:
            value = self._values.get(key)
            if value is not None:
                self._values.move_to_end(key)
            return value

    def put(self, key: Hashable, value: Value) -> None:
        with self._lock:
            self._values[key] = value
            self._values.move_to_end(key)
            while len(self._values) > self.max_entries:
                self._values.popitem(last=False)

    def __len__(self) -> int:
        with self._lock:
            return len(self._values)


def normalize_query(query: str) -> str:
    return " ".join(query.casefold().split())


def retrieval_cache_key(
    query: str, language: str | None, *, corpus_version: str, retriever_version: str,
) -> tuple[str, str, str, str]:
    return normalize_query(query), language or "auto", corpus_version, retriever_version


def embedding_cache_key(query: str, *, model_version: str) -> tuple[str, str]:
    return normalize_query(query), model_version
