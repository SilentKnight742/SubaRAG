"""Portable corpus preparation; independent of retrieval and model dependencies."""

from .manifest import DataManifest, load_manifest

__all__ = ["DataManifest", "load_manifest"]
