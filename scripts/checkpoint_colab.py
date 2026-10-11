"""Create and verify a private, downloadable checkpoint of a rebuilt Colab run.

No downloads, credentials, model execution, cloud writes, or changes to inputs.
The archive contains dataset text: keep it private, not in the public Git repo.
"""

from __future__ import annotations

import argparse
import hashlib
import importlib.metadata
import io
import json
import os
from pathlib import Path
import shutil
import sys
import tarfile
import tempfile

HELPERS = {
    "check_light_tokenizer.py": "b8b66f5b22ddd80be600400324b0df8b9002de4eb523f1900f76c9cb66b846f8",
    "check_float64_tokenizer.py": "4d4e39bf27366b93e05963d0e9debca53901afb19b83d981428197f9de8424a1",
    "check_disk_retrieval.py": "5ebc4544ae6b4ad5b692c00ce4b6c0f15901f14c395899c6cb7c39f7f0b6d63d",
}
WORK_SUFFIXES = {".json", ".jsonl", ".faiss", ".onnx", ".model", ".marisa", ".bin", ".sqlite"}


def digest_stream(handle):
    result = hashlib.sha256()
    while chunk := handle.read(1024 * 1024):
        result.update(chunk)
    return result.hexdigest()


def digest(path):
    with path.open("rb") as handle:
        return digest_stream(handle)


def collect(work, backend, helper_dir):
    required = ("inputs.json", "reference.json", "candidate.json", "compact-report.json",
                "e5-fp32.onnx", "compact-model/model.onnx", "compact-model/tokenizer.json",
                "compact-model/embedding-manifest.json")
    if not all((work / name).is_file() for name in required):
        raise ValueError("The rebuilt compact run is incomplete; keep its files and report this error.")
    reports = list(work.glob("disk-retrieval-*/disk-retrieval-report.json"))
    if not reports:
        raise ValueError("Run the disk-retrieval check before saving this checkpoint.")
    latest = json.loads(max(reports, key=lambda p: p.stat().st_mtime).read_text())
    if latest["script_sha256"] != HELPERS["check_disk_retrieval.py"] or not latest["original_files_unchanged"]:
        raise ValueError("Unexpected disk-retrieval report provenance")
    for name, checksum in latest["source_sha256"].items():
        source = Path(name).resolve()
        if not source.is_relative_to(work) or digest(source) != checksum:
            raise ValueError("A verified experiment input is missing or changed")

    sources = {}
    for path in sorted(work.rglob("*")):
        if path.is_symlink():
            raise ValueError(f"Checkpoint refuses symlinks: {path.name}")
        if path.is_dir():
            continue
        if (not path.is_file() or path.name.startswith(".") or path.suffix not in WORK_SUFFIXES
                or any(part.startswith(".") for part in path.relative_to(work).parts)):
            raise ValueError(f"Unexpected experiment file; inspect before archiving: {path.name}")
        sources["work/" + path.relative_to(work).as_posix()] = path

    # Only package source and its installation metadata, never .env or caches.
    package = backend / "src/subarag"
    if not (backend / "pyproject.toml").is_file() or not (package / "experiments/compact_embedding.py").is_file():
        raise ValueError("Cannot locate the notebook's extracted backend source")
    sources["backend/pyproject.toml"] = backend / "pyproject.toml"
    for path in sorted(package.rglob("*")):
        if path.is_symlink():
            raise ValueError("Checkpoint refuses source symlinks")
        if path.is_file() and path.suffix == ".py":
            sources["backend/" + path.relative_to(backend).as_posix()] = path
    for name, checksum in HELPERS.items():
        matches = [p for p in helper_dir.glob(Path(name).stem + "*.py")
                   if not p.is_symlink() and digest(p) == checksum]
        if not matches:
            raise ValueError(f"Keep the previously uploaded {name} in the same folder as this script.")
        sources["helpers/" + name] = sorted(matches)[0]
    return sources


def verify(archive, manifest):
    expected = manifest["files"]
    seen = set()
    with tarfile.open(archive, "r|gz") as saved:
        for member in saved:
            if not member.isfile() or member.name in seen:
                raise ValueError("Invalid or duplicate checkpoint member")
            seen.add(member.name)
            with saved.extractfile(member) as handle:
                if member.name == "checkpoint-manifest.json":
                    if json.load(handle) != manifest:
                        raise ValueError("Checkpoint manifest differs")
                elif (member.name not in expected or member.size != expected[member.name]["bytes"]
                      or digest_stream(handle) != expected[member.name]["sha256"]):
                    raise ValueError("Checkpoint file verification failed")
    if seen != set(expected) | {"checkpoint-manifest.json"}:
        raise ValueError("Checkpoint is incomplete")


def create(work, backend, helper_dir):
    work, backend = work.resolve(), backend.resolve()
    sources = collect(work, backend, helper_dir)
    entries = {name: {"bytes": path.stat().st_size, "sha256": digest(path)}
               for name, path in sources.items()}
    size = sum(item["bytes"] for item in entries.values())
    if shutil.disk_usage(work.parent).free < size + 64 * 1024 * 1024:
        raise ValueError("Not enough free space for a safe checkpoint; do not delete experiment files.")
    manifest = {
        "schema_version": 1, "kind": "subarag-private-colab-checkpoint",
        "original_work": str(work), "original_backend": str(backend),
        "python": sys.version, "files": entries,
        "packages": sorted({(d.metadata["Name"], d.version) for d in importlib.metadata.distributions()
                            if d.metadata.get("Name") and d.version}),
        "restore_note": "Restore work and backend to their original paths to preserve report references; reinstall dependencies separately. No Python environment or public model cache is included.",
    }
    # JSON normalizes package tuples for the read-back comparison.
    manifest = json.loads(json.dumps(manifest))
    output = Path(tempfile.mkdtemp(prefix="subarag-checkpoint-", dir=work.parent))
    archive = output / "SubaRAG-private-checkpoint.tar.gz"
    print(f"Saving {len(entries)} files ({size / 1024**2:.1f} MiB before compression)…", flush=True)
    with tarfile.open(archive, "w:gz", compresslevel=1) as saved:
        for name, path in sources.items():
            saved.add(path, arcname=name, recursive=False)
        payload = (json.dumps(manifest, indent=2) + "\n").encode()
        info = tarfile.TarInfo("checkpoint-manifest.json")
        info.size, info.mode = len(payload), 0o600
        saved.addfile(info, io.BytesIO(payload))
    print("Verifying every archived file…", flush=True)
    verify(archive, manifest)
    if any(digest(path) != entries[name]["sha256"] for name, path in sources.items()):
        raise ValueError("Inputs changed while archiving; keep the runtime and report this error.")
    report = {"schema_version": 1, "archive": str(archive), "archive_bytes": archive.stat().st_size,
              "archive_sha256": digest(archive), "file_count": len(entries),
              "all_archived_files_verified": True, "original_files_unchanged": True,
              "original_work": str(work), "original_backend": str(backend),
              "privacy": "Contains private dataset text; keep offline/private. No environment variables, tokens, .env files, or account caches were collected."}
    report_path = output / "checkpoint-report.json"
    report_path.write_text(json.dumps(report, indent=2) + "\n")
    print(f"Verified archive: {archive} ({archive.stat().st_size / 1024**2:.1f} MiB)", flush=True)
    print(f"REPORT: {report_path}", flush=True)
    return report_path


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--work", type=Path, required=True)
    args = parser.parse_args()
    roots = [Path(item).resolve() for item in os.environ.get("PYTHONPATH", "").split(os.pathsep) if item]
    backends = [p.parent for p in roots if (p / "subarag/experiments/compact_embedding.py").is_file()]
    if len(backends) != 1:
        parser.error("Run with the notebook's existing child_env so the exact source is preserved.")
    create(args.work, backends[0], Path(__file__).resolve().parent)
