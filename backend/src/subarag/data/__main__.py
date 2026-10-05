"""Run with `python -m subarag.data` in a remote CPU runtime."""

import argparse
import json
from pathlib import Path

from .manifest import verify_manifest
from .prepare import prepare_corpus
from .source import DEFAULT_REVISION


def main() -> None:
    parser = argparse.ArgumentParser(description="Prepare SubaRAG corpus artifacts; no embeddings or retrieval.")
    commands = parser.add_subparsers(dest="command", required=True)
    prepare = commands.add_parser("prepare")
    prepare.add_argument("--output", type=Path, required=True)
    prepare.add_argument("--languages", nargs="+", choices=("en", "hi", "mr"), default=["en", "hi", "mr"])
    prepare.add_argument("--splits", nargs="+", choices=("train", "validation"), default=["train", "validation"])
    prepare.add_argument("--revision", default=DEFAULT_REVISION)
    scale = prepare.add_mutually_exclusive_group()
    scale.add_argument("--rows-per-source", type=int, default=1000)
    scale.add_argument("--full", action="store_true", help="Process all selected rows; run only on provisioned remote compute")
    prepare.add_argument("--shard-rows", type=int, default=50_000)
    verify = commands.add_parser("verify")
    verify.add_argument("manifest", type=Path)
    args = parser.parse_args()
    try:
        if args.command == "verify":
            manifest = verify_manifest(args.manifest)
        else:
            manifest = prepare_corpus(args.output, languages=args.languages, splits=args.splits,
                                      revision=args.revision, rows_per_source=None if args.full else args.rows_per_source,
                                      shard_rows=args.shard_rows)
    except (OSError, ValueError, RuntimeError) as exc:
        parser.exit(1, f"Data preparation failed: {exc}\n")
    print(json.dumps({"artifact_kind": manifest.artifact_kind, "counts": manifest.counts.model_dump(),
                      "sample_rows_per_source": manifest.sample_rows_per_source}, indent=2))


if __name__ == "__main__":
    main()
