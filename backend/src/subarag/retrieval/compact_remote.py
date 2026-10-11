"""Fetch one immutable compact bundle before loading native model/index files."""

from __future__ import annotations

import asyncio
import hashlib
import logging
from pathlib import Path
import re
from tempfile import TemporaryDirectory

import httpx

from ..data.remote import _download
from .compact import (
    MAX_MANIFEST_BYTES,
    load_compact_bundle_manifest,
    verify_compact_bundle,
)
from .onnx_embedding import file_sha256


logger = logging.getLogger(__name__)


async def fetch_compact_bundle(
    *, repo_id: str, revision: str, path: str, token: str, cache_dir: Path,
) -> Path:
    """Return a local verified manifest for a pinned, content-addressed bundle."""
    if not re.fullmatch(r"[A-Za-z0-9_-]+/[A-Za-z0-9_.-]+", repo_id):
        raise ValueError("Invalid compact repository")
    if not re.fullmatch(r"[0-9a-f]{40}", revision):
        raise ValueError("Compact revision must be a full commit SHA")
    if not re.fullmatch(r"serving/compact/[0-9a-f]{64}", path):
        raise ValueError("Compact path must be content-addressed")

    expected_manifest_hash = path.rsplit("/", 1)[1]
    identity = f"{repo_id}\n{revision}\n{path}"
    cache_dir = Path(cache_dir).resolve()
    destination = cache_dir / hashlib.sha256(identity.encode()).hexdigest()
    manifest = destination / "compact-bundle.json"
    try:
        if destination.exists():
            if file_sha256(manifest) != expected_manifest_hash:
                raise ValueError("Cached compact manifest fingerprint mismatch")
            verify_compact_bundle(manifest)
            return manifest

        cache_dir.mkdir(parents=True, exist_ok=True)
        base = f"https://huggingface.co/datasets/{repo_id}/resolve/{revision}/{path}/bundle/"
        async with asyncio.timeout(900):
            async with httpx.AsyncClient(
                headers={"Authorization": f"Bearer {token}"},
                follow_redirects=True,
                max_redirects=5,
                timeout=60,
            ) as client:
                with TemporaryDirectory(prefix=".compact-download-", dir=cache_dir) as staging:
                    root = Path(staging)
                    pending_manifest = root / "compact-bundle.json"
                    await _download(client, base + "compact-bundle.json", pending_manifest,
                                    limit=MAX_MANIFEST_BYTES)
                    if file_sha256(pending_manifest) != expected_manifest_hash:
                        raise ValueError("Compact manifest fingerprint mismatch")
                    metadata = load_compact_bundle_manifest(pending_manifest)
                    for relative, item in metadata["files"].items():
                        await _download(client, base + relative, root / relative,
                                        limit=item["bytes"], expected_hash=item["sha256"])
                    verify_compact_bundle(pending_manifest)
                    root.rename(destination)
        logger.info("Compact bundle verified: %s at %s", repo_id, revision)
        return manifest
    except (httpx.HTTPError, OSError, ValueError, TimeoutError):
        raise RuntimeError(
            "Compact bundle verification failed. Check HF_TOKEN read access, the pinned "
            "repository/revision/path, disk space, and checksums."
        ) from None
