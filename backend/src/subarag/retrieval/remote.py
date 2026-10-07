"""Stream a pinned retrieval bundle to disk before any native index is loaded."""

from __future__ import annotations

import asyncio
import hashlib
import json
import logging
import re
from pathlib import Path
from tempfile import TemporaryDirectory
from typing import Annotated, Literal

import httpx
from pydantic import BaseModel, ConfigDict, Field, model_validator

from ..data.remote import MAX_MANIFEST_BYTES, _download


LANGUAGES = ("en", "hi", "mr")
FILENAMES = ("retrieval-manifest.json", "documents.jsonl", "index.faiss")
MAX_BUNDLE_BYTES = 256 * 1024 * 1024
SHA256 = Annotated[str, Field(pattern=r"^[0-9a-f]{64}$")]
REVISION = Annotated[str, Field(pattern=r"^[0-9a-f]{40}$")]
REPOSITORY = Annotated[str, Field(pattern=r"^[A-Za-z0-9_-]+/[A-Za-z0-9_.-]+$")]
logger = logging.getLogger(__name__)


class Contract(BaseModel):
    model_config = ConfigDict(extra="forbid", strict=True)


class FileMetadata(Contract):
    bytes: int = Field(gt=0, le=MAX_BUNDLE_BYTES)
    sha256: SHA256


class LanguageMetadata(Contract):
    documents: int = Field(gt=0)
    dimension: int = Field(gt=0)
    corpus_version: str = Field(min_length=1)


class RetrievalBundle(Contract):
    schema_version: Literal[1]
    artifact_kind: Literal["subarag_retrieval_bundle"]
    prepared_repo: REPOSITORY
    prepared_revision: REVISION
    prepared_manifest_sha256: SHA256
    source_archive_sha256: SHA256
    model_name: Literal["intfloat/multilingual-e5-small"]
    model_revision: REVISION
    build_device: Literal["cpu", "cuda"]
    packages: dict[str, str]
    languages: dict[str, LanguageMetadata]
    files: dict[str, FileMetadata]
    total_bytes: int = Field(gt=0, le=MAX_BUNDLE_BYTES)

    @model_validator(mode="after")
    def complete_bundle(self) -> "RetrievalBundle":
        if set(self.languages) != set(LANGUAGES):
            raise ValueError("A retrieval bundle requires English, Hindi, and Marathi")
        expected = {f"{language}/{name}" for language in LANGUAGES for name in FILENAMES}
        if set(self.files) != expected:
            raise ValueError("A retrieval bundle must list exactly the nine index files")
        if sum(item.bytes for item in self.files.values()) != self.total_bytes:
            raise ValueError("Retrieval bundle size is inconsistent")
        if any(item.dimension != 384 for item in self.languages.values()):
            raise ValueError("E5-small retrieval requires 384 dimensions")
        if any(self.files[f"{lang}/retrieval-manifest.json"].bytes > MAX_MANIFEST_BYTES
               for lang in LANGUAGES):
            raise ValueError("Retrieval manifest exceeds its metadata limit")
        return self


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        while chunk := handle.read(64 * 1024):
            digest.update(chunk)
    return digest.hexdigest()


def _read_bundle(
    path: Path, expected_hash: str, *, prepared_manifest: Path | None,
    prepared_repo: str | None, prepared_revision: str | None,
) -> RetrievalBundle:
    if path.stat().st_size > MAX_MANIFEST_BYTES or _sha256(path) != expected_hash:
        raise ValueError("Retrieval bundle fingerprint mismatch")
    bundle = RetrievalBundle.model_validate_json(path.read_bytes())
    if prepared_repo is not None and bundle.prepared_repo != prepared_repo:
        raise ValueError("Retrieval bundle belongs to a different prepared repository")
    if prepared_revision is not None and bundle.prepared_revision != prepared_revision:
        raise ValueError("Retrieval bundle belongs to a different prepared revision")
    if prepared_manifest is not None and _sha256(prepared_manifest) != bundle.prepared_manifest_sha256:
        raise ValueError("Retrieval bundle does not match the configured corpus manifest")
    return bundle


def _verify_files(root: Path, bundle: RetrievalBundle) -> dict[str, Path]:
    for relative, metadata in bundle.files.items():
        path = (root / relative).resolve()
        if not path.is_relative_to(root.resolve()):
            raise ValueError("Retrieval file escapes its cache directory")
        if path.stat().st_size != metadata.bytes or _sha256(path) != metadata.sha256:
            raise ValueError("Retrieval file failed checksum or size verification")
    for language in LANGUAGES:
        manifest = json.loads((root / language / FILENAMES[0]).read_bytes())
        if not isinstance(manifest, dict):
            raise ValueError("Retrieval manifest must be an object")
        metadata = bundle.languages[language]
        if any(manifest.get(key) != value for key, value in {
            "schema_version": 1, "artifact_kind": "subarag_hnsw", "language": language,
            "model_name": bundle.model_name, "model_revision": bundle.model_revision,
            "document_count": metadata.documents, "dimension": metadata.dimension,
            "corpus_version": metadata.corpus_version,
            "files": {name: bundle.files[f"{language}/{name}"].model_dump()
                      for name in FILENAMES[1:]},
        }.items()):
            raise ValueError("Retrieval manifest disagrees with its bundle")
    return {language: root / language / FILENAMES[0] for language in LANGUAGES}


async def fetch_retrieval_bundle(
    *, repo_id: str, revision: str, path: str, token: str, cache_dir: Path,
    prepared_manifest: Path | None = None, prepared_repo: str | None = None,
    prepared_revision: str | None = None,
) -> dict[str, Path]:
    """Accept a cache only after checking its content-addressed report and all files.

    This function requires no PyTorch or FAISS. Native deserialization and model
    loading happen later, in the configured RAG engine, only on verified files.
    """
    if not re.fullmatch(r"[A-Za-z0-9_-]+/[A-Za-z0-9_.-]+", repo_id):
        raise ValueError("Invalid retrieval repository")
    if not re.fullmatch(r"[0-9a-f]{40}", revision):
        raise ValueError("Retrieval revision must be a full commit SHA")
    if not re.fullmatch(r"retrieval/[0-9a-f]{64}", path):
        raise ValueError("Retrieval path must be content-addressed")
    expected_hash = path.split("/")[1]
    identity = f"{repo_id}\n{revision}\n{path}"
    cache_dir = Path(cache_dir).resolve()
    destination = cache_dir / hashlib.sha256(identity.encode()).hexdigest()
    provenance = dict(prepared_manifest=prepared_manifest, prepared_repo=prepared_repo,
                      prepared_revision=prepared_revision)
    try:
        if destination.exists():
            bundle = _read_bundle(destination / "bundle.json", expected_hash, **provenance)
            return _verify_files(destination, bundle)
        cache_dir.mkdir(parents=True, exist_ok=True)
        base = f"https://huggingface.co/datasets/{repo_id}/resolve/{revision}/{path}/"
        async with asyncio.timeout(300):
            async with httpx.AsyncClient(
                headers={"Authorization": f"Bearer {token}"},
                follow_redirects=True, max_redirects=5, timeout=30,
            ) as client:
                with TemporaryDirectory(prefix=".download-", dir=cache_dir) as staging:
                    root = Path(staging)
                    report = root / "bundle.json"
                    await _download(client, base + "bundle.json", report, limit=MAX_MANIFEST_BYTES)
                    bundle = _read_bundle(report, expected_hash, **provenance)
                    for relative, metadata in bundle.files.items():
                        await _download(client, base + relative, root / relative,
                                        limit=metadata.bytes, expected_hash=metadata.sha256)
                    _verify_files(root, bundle)
                    root.rename(destination)
        logger.info("Retrieval bundle verified: %s at %s", repo_id, revision)
        return {language: destination / language / FILENAMES[0] for language in LANGUAGES}
    except (httpx.HTTPError, OSError, ValueError, TimeoutError):
        # Neither signed redirect URLs nor authorization headers belong in logs.
        raise RuntimeError(
            "Retrieval bundle verification failed. Check HF_TOKEN read access, the "
            "pinned repository/revision/path, corpus provenance, disk space, and checksums."
        ) from None
