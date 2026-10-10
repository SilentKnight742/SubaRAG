"""Restore the verified private SubaRAG checkpoint in a fresh Colab runtime.

Standard-library file restoration, without model execution or overwrite.
Optional --install fetches runtime libraries only, after file verification.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
from pathlib import Path, PurePosixPath
import re
import shutil
import subprocess
import sys
import tarfile
import tempfile

ARCHIVE_SHA = "4e014739505d92bf6213fa099d26917e0aff0cca08ef97d74f700e8fa39afad7"
ARCHIVE_BYTES = 537621418
HELPERS = {"check_light_tokenizer.py", "check_float64_tokenizer.py", "check_disk_retrieval.py"}
PACKAGES = ("av", "fastapi", "httpx", "pydantic", "pydantic-settings", "python-multipart",
            "uvicorn", "numpy", "faiss-cpu", "onnx", "onnxruntime", "tokenizers",
            "marisa-trie", "protobuf", "ml-dtypes", "flatbuffers", "sympy", "packaging")


def digest(path):
    checksum = hashlib.sha256()
    with path.open("rb") as handle:
        while chunk := handle.read(1024 * 1024):
            checksum.update(chunk)
    return checksum.hexdigest()


def safe_parts(name):
    path = PurePosixPath(name)
    if path.is_absolute() or path.as_posix() != name or ".." in path.parts or "\\" in name:
        raise ValueError("Unsafe checkpoint member path")
    if not path.parts or any(part.startswith(".") for part in path.parts):
        raise ValueError("Unexpected hidden checkpoint member")
    return path.parts


def original_path(value, content, kind):
    path = PurePosixPath(value)
    if path.as_posix() != value or not path.is_absolute() or ".." in path.parts:
        raise ValueError("Invalid original path")
    parts = path.relative_to("/content").parts
    if kind == "work":
        valid = len(parts) == 1 and re.fullmatch(r"subarag-compact-[a-zA-Z0-9_-]+", parts[0])
    else:
        valid = (len(parts) == 3 and re.fullmatch(r"subarag-compact-source-[a-zA-Z0-9_-]+", parts[0])
                 and parts[1:] == ("SubaRAG", "backend"))
    if not valid:
        raise ValueError("Unexpected checkpoint destination")
    return content.joinpath(*parts)


def destination(name, roots):
    parts = safe_parts(name)
    if len(parts) < 2 or parts[0] not in roots:
        raise ValueError("Unknown checkpoint member group")
    if parts[0] == "helpers" and (len(parts) != 2 or parts[1] not in HELPERS):
        raise ValueError("Unexpected helper script")
    return roots[parts[0]].joinpath(*parts[1:])


def check_existing(target, content, expected):
    for path in (target, *target.parents):
        if path == content:
            break
        if path.is_symlink():
            raise ValueError(f"Refusing a symlink at {path}")
        if path != target and path.exists() and not path.is_dir():
            raise ValueError(f"Destination parent is not a directory: {path}")
    if target.exists():
        if (not target.is_file() or target.stat().st_size != expected["bytes"]
                or digest(target) != expected["sha256"]):
            raise ValueError(f"Existing file differs; nothing will be overwritten: {target}")
        return True
    return False


def restore(archive, content, *, expected_sha=ARCHIVE_SHA, expected_bytes=ARCHIVE_BYTES):
    archive, content = archive.resolve(), content.resolve()
    if not content.is_dir():
        raise ValueError("This helper must run in Colab, where /content exists")
    print("Checking the uploaded archive checksum…", flush=True)
    if archive.stat().st_size != expected_bytes or digest(archive) != expected_sha:
        raise ValueError("Wrong or incomplete archive. Upload the verified private checkpoint; do not rebuild.")
    with tarfile.open(archive, "r:gz") as saved:
        members = saved.getmembers()
        names = [member.name for member in members]
        if len(names) != len(set(names)) or any(not m.isfile() or m.sparse for m in members):
            raise ValueError("Checkpoint contains duplicate, linked, sparse, or non-file members")
        for name in names:
            safe_parts(name)
        marker = saved.getmember("checkpoint-manifest.json")
        if marker.size > 2 * 1024 * 1024:
            raise ValueError("Unexpectedly large checkpoint manifest")
        with saved.extractfile(marker) as handle:
            manifest = json.load(handle)
    if manifest.get("schema_version") != 1 or manifest.get("kind") != "subarag-private-colab-checkpoint":
        raise ValueError("Unsupported checkpoint")
    files = manifest["files"]
    if set(names) != set(files) | {"checkpoint-manifest.json"}:
        raise ValueError("Checkpoint inventory differs from manifest")
    roots = {"work": original_path(manifest["original_work"], content, "work"),
             "backend": original_path(manifest["original_backend"], content, "backend"), "helpers": content}
    for item in files.values():
        if (not isinstance(item["bytes"], int) or item["bytes"] < 0
                or not re.fullmatch(r"[0-9a-f]{64}", item["sha256"])):
            raise ValueError("Invalid manifest checksum/size")
    targets = {name: destination(name, roots) for name in files}
    if len(set(targets.values())) != len(targets):
        raise ValueError("Checkpoint destinations overlap")
    # Preflight ALL destinations before creating any restored files.
    existing = {name: check_existing(path, content, files[name]) for name, path in targets.items()}
    required = sum(item["bytes"] for item in files.values()) + 64 * 1024 * 1024
    if shutil.disk_usage(content).free < required:
        raise ValueError("Not enough free space for verified staging. Keep the archive and report this error.")
    packages = {}
    for name, version in manifest["packages"]:
        packages.setdefault(re.sub(r"[-_.]+", "-", name.lower()), set()).add(version)
    pins, duplicate_choices = [], {}
    for name in PACKAGES:
        choices = packages.get(name, set())
        if len(choices) > 1:
            # The saved Colab image contains stale duplicate distribution
            # metadata. NumPy 2 is required by the archived project. These are
            # explicit, recorded choices, not a claim about the old active import.
            known = {"numpy": ({"1.26.4", "2.1.3"}, "2.1.3"),
                     "packaging": ({"24.0", "26.3"}, "26.3")}
            if name not in known or choices != known[name][0]:
                raise ValueError(f"Ambiguous recorded versions for {name}; stop and report this error")
            version = known[name][1]
            duplicate_choices[name] = {"recorded": sorted(choices), "selected": version}
        else:
            version = next(iter(choices), None)
        if not version or not re.fullmatch(r"[0-9][A-Za-z0-9.!+_-]*", version):
            raise ValueError(f"Missing or invalid recorded version: {name}")
        pins.append(f"{name}=={version}")

    print(f"Verifying and staging {len(files)} files…", flush=True)
    with tempfile.TemporaryDirectory(prefix="subarag-restore-stage-", dir=content) as temporary:
        stage = Path(temporary)
        with tarfile.open(archive, "r|gz") as saved:
            for member in saved:
                if member.name == "checkpoint-manifest.json":
                    continue
                expected = files[member.name]
                if member.size != expected["bytes"]:
                    raise ValueError("Archived file size differs from manifest")
                target = stage / member.name
                target.parent.mkdir(parents=True, exist_ok=True)
                checksum = hashlib.sha256()
                with saved.extractfile(member) as source, target.open("xb") as output:
                    while chunk := source.read(1024 * 1024):
                        checksum.update(chunk)
                        output.write(chunk)
                if target.stat().st_size != expected["bytes"] or checksum.hexdigest() != expected["sha256"]:
                    raise ValueError("Archived file checksum failed; original destinations remain untouched")
                target.chmod(0o600)
        # Same-filesystem hardlinks publish verified files without another model
        # copy and fail atomically if a destination appears concurrently.
        for name, target in targets.items():
            if existing[name]:
                continue
            check_existing(target, content, files[name])
            target.parent.mkdir(parents=True, exist_ok=True)
            os.link(stage / name, target)
    print("Verifying restored files at their original paths…", flush=True)
    for name, target in targets.items():
        if not check_existing(target, content, files[name]):
            raise ValueError("Restored file is missing")
    output = Path(tempfile.mkdtemp(prefix="subarag-restore-", dir=content))
    requirements = output / "requirements.txt"
    requirements.write_text("\n".join(pins) + "\n")
    report = {"schema_version": 1, "archive_sha256": expected_sha, "restored_files_verified": True,
              "file_count": len(files), "reused_identical_files": sum(existing.values()),
              "work": str(roots["work"]), "backend": str(roots["backend"]), "helper_directory": str(content),
              "requirements": str(requirements), "recorded_python": manifest["python"],
              "current_python": sys.version, "dependencies_installed": False,
              "duplicate_distribution_choices": duplicate_choices,
              "note": "Files restored through the disk-retrieval experiment. Later experiment variants are not in this backup. No model execution, rebuild, download, or cloud upload occurred."}
    report_path = output / "restore-report.json"
    report_path.write_text(json.dumps(report, indent=2) + "\n")
    print("Files restored. Install the recorded runtime dependencies using the notebook cell next.", flush=True)
    print("REPORT:", report_path, flush=True)
    return report_path


def install_dependencies(report_path):
    report = json.loads(report_path.read_text())
    backend = Path(report["backend"])
    print("Installing required libraries at selected recorded versions (no model download)…", flush=True)
    subprocess.run([sys.executable, "-m", "pip", "install", f"{backend}[compact]",
                    "-r", report["requirements"]], check=True)
    env = dict(os.environ, PYTHONPATH=str(backend / "src"), OMP_NUM_THREADS="1",
               OPENBLAS_NUM_THREADS="1", TOKENIZERS_PARALLELISM="false",
               HF_HUB_OFFLINE="1", TRANSFORMERS_OFFLINE="1")
    env.pop("HF_TOKEN", None)
    # A new interpreter avoids stale notebook imports after package installation.
    probe = """
import importlib.metadata as metadata, json, sys
import numpy, faiss, onnx, onnxruntime, tokenizers, marisa_trie
import subarag.api.app
expected = json.loads(sys.argv[1])
versions = {name: metadata.version(name) for name in expected}
if versions != expected or numpy.__version__ != expected['numpy']:
    raise RuntimeError('Installed/imported versions differ from restoration requirements')
print(json.dumps(versions))
"""
    expected = dict(line.split("==", 1) for line in Path(report["requirements"]).read_text().splitlines())
    checked = subprocess.run([sys.executable, "-c", probe, json.dumps(expected)], env=env,
                             capture_output=True, text=True)
    if checked.returncode:
        print(checked.stdout, flush=True)
        print(checked.stderr, file=sys.stderr, flush=True)
        raise RuntimeError("Library import check failed. Keep the restored files and share the error above.")
    report.update(dependencies_installed=True, imports_checked=True,
                  installed_versions=json.loads(checked.stdout),
                  runtime_note="Required libraries checked in a fresh process; the whole Colab environment is not cloned. Numerical parity still needs a resumed control check.")
    report_path.write_text(json.dumps(report, indent=2) + "\n")
    print("Restore complete: files verified, dependencies installed, imports checked. No model was executed.", flush=True)
    print("REPORT:", report_path, flush=True)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--archive", type=Path, required=True)
    parser.add_argument("--install", action="store_true", help="Install recorded runtime libraries after file verification")
    args = parser.parse_args()
    report_path = restore(args.archive, Path("/content"))
    if args.install:
        install_dependencies(report_path)
