from __future__ import annotations

import asyncio
import hashlib
import json

import httpx

from subarag.retrieval.compact import FILES
from subarag.retrieval.compact_remote import fetch_compact_bundle


def test_fetch_compact_bundle_verifies_and_reuses_atomic_cache(tmp_path, monkeypatch):
    objects = {name: ("fixture:" + name).encode() for name in FILES}
    metadata = {
        "schema_version": 1,
        "artifact_kind": "subarag_compact_bundle",
        "model_name": "intfloat/multilingual-e5-small",
        "model_revision": "6" * 40,
        "dimension": 384,
        "languages": {
            lang: {"documents": 1, "corpus_version": "fixture-v1"}
            for lang in ("en", "hi", "mr")
        },
        "provisional": True,
        "quality_note": "Fixture only.",
        "source_preprocessing_report_sha256": "7" * 64,
        "files": {
            name: {"bytes": len(value), "sha256": hashlib.sha256(value).hexdigest()}
            for name, value in sorted(objects.items())
        },
    }
    manifest = (json.dumps(metadata, sort_keys=True) + "\n").encode()
    fingerprint = hashlib.sha256(manifest).hexdigest()
    objects["compact-bundle.json"] = manifest
    calls = []

    def handler(request: httpx.Request) -> httpx.Response:
        assert request.headers["Authorization"] == "Bearer test-token"
        name = request.url.path.rsplit("/bundle/", 1)[1]
        calls.append(name)
        return httpx.Response(200, content=objects[name])

    original_client = httpx.AsyncClient
    transport = httpx.MockTransport(handler)

    def client(**kwargs):
        return original_client(transport=transport, headers=kwargs["headers"])

    monkeypatch.setattr("subarag.retrieval.compact_remote.httpx.AsyncClient", client)
    options = dict(
        repo_id="silentkirmada/SubaRAG-data",
        revision="3" * 40,
        path="serving/compact/" + fingerprint,
        token="test-token",
        cache_dir=tmp_path,
    )
    result = asyncio.run(fetch_compact_bundle(**options))
    assert result.name == "compact-bundle.json"
    assert json.loads(result.read_text()) == metadata
    assert set(calls) == set(objects)
    before = list(calls)
    assert asyncio.run(fetch_compact_bundle(**options)) == result
    assert calls == before
