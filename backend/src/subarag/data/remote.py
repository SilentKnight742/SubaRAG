"""Fetch an immutable prepared corpus, never the upstream training dataset."""

import asyncio
import hashlib
import logging
import re
from pathlib import Path
from tempfile import TemporaryDirectory
from urllib.parse import quote

import httpx

from .manifest import load_manifest, verify_manifest


MAX_MANIFEST_BYTES = 1024 * 1024
MAX_CORPUS_BYTES = 256 * 1024 * 1024
logger = logging.getLogger(__name__)


async def _download(
    client: httpx.AsyncClient, url: str, destination: Path, *,
    limit: int, expected_hash: str | None = None,
) -> None:
    destination.parent.mkdir(parents=True, exist_ok=True)
    digest = hashlib.sha256()
    size = 0
    async with client.stream("GET", url) as response:
        response.raise_for_status()
        with destination.open("xb") as output:
            async for chunk in response.aiter_bytes(chunk_size=64 * 1024):
                size += len(chunk)
                if size > limit:
                    raise ValueError("Prepared artifact exceeds its allowed size")
                output.write(chunk)
                digest.update(chunk)
    if expected_hash is not None and (size != limit or digest.hexdigest() != expected_hash):
        raise ValueError("Prepared artifact checksum or size does not match its manifest")


async def fetch_prepared_corpus(
    *, repo_id: str, revision: str, token: str, cache_dir: Path,
) -> Path:
    """Publish a cache directory only once every listed shard has been verified."""
    if not re.fullmatch(r"[A-Za-z0-9_-]+/[A-Za-z0-9_.-]+", repo_id):
        raise ValueError("Invalid dataset repository name")
    if not re.fullmatch(r"[0-9a-f]{40}", revision):
        raise ValueError("Dataset revision must be a full commit SHA")
    key = hashlib.sha256(repo_id.encode()).hexdigest()[:16] + "-" + revision
    cache_dir = Path(cache_dir).resolve()
    destination = cache_dir / key
    manifest_path = destination / "manifest.json"
    if destination.exists():
        verify_manifest(manifest_path)
        return manifest_path

    cache_dir.mkdir(parents=True, exist_ok=True)
    base = f"https://huggingface.co/datasets/{repo_id}/resolve/{revision}/"
    try:
        async with asyncio.timeout(180):
            # HTTPX strips Authorization when following cross-origin redirects.
            async with httpx.AsyncClient(
                headers={"Authorization": f"Bearer {token}"},
                follow_redirects=True, max_redirects=5, timeout=30,
            ) as client:
                with TemporaryDirectory(prefix=".download-", dir=cache_dir) as staging:
                    root = Path(staging)
                    pending_manifest = root / "manifest.json"
                    await _download(client, base + "manifest.json", pending_manifest,
                                    limit=MAX_MANIFEST_BYTES)
                    manifest = load_manifest(pending_manifest)
                    if sum(shard.bytes for shard in manifest.shards) > MAX_CORPUS_BYTES:
                        raise ValueError("Prepared corpus exceeds the 256 MiB preview limit")
                    for shard in manifest.shards:
                        await _download(
                            client, base + quote(shard.path, safe="/"), root / shard.path,
                            limit=shard.bytes, expected_hash=shard.sha256,
                        )
                    verify_manifest(pending_manifest)
                    root.rename(destination)
        logger.info("Prepared corpus verified: %s at %s", repo_id, revision)
        return manifest_path
    except (httpx.HTTPError, OSError, ValueError, TimeoutError):
        # Never include signed download URLs, headers, or provider error bodies.
        raise RuntimeError(
            "Prepared corpus download failed. Check HF_TOKEN read access, the pinned "
            "dataset revision, available disk space, and artifact checksums."
        ) from None
