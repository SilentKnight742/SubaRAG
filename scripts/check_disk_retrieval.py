"""Offline Colab screen: disk-backed documents and bounded index residency.

Uses the verified float64-tokenizer run and existing HNSW files. No index
rebuild, model download, upload, deployment, or modifications to original files.
"""

from __future__ import annotations

import argparse
from collections.abc import Sequence
import gc
import hashlib
import importlib.util
import json
import os
from pathlib import Path
import sqlite3
import subprocess
import sys
import tempfile
from time import perf_counter
from unittest.mock import patch

TOKENIZER_SCRIPT_SHA = "4d4e39bf27366b93e05963d0e9debca53901afb19b83d981428197f9de8424a1"
LANGUAGES = ("en", "hi", "mr")


def read(path):
    return json.loads(Path(path).read_text(encoding="utf-8"))


def write(path, value):
    Path(path).write_text(json.dumps(value, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")


def digest(path):
    value = hashlib.sha256()
    with Path(path).open("rb") as handle:
        while chunk := handle.read(1024 * 1024):
            value.update(chunk)
    return value.hexdigest()


def memory():
    values = {line.split(":")[0]: int(line.split()[1]) / 1024
              for line in Path("/proc/self/status").read_text().splitlines()
              if line.startswith(("VmRSS:", "VmHWM:"))}
    return {"rss_mib": values["VmRSS"], "peak_mib": values["VmHWM"]}


def tokenizer_helper():
    for path in Path(__file__).resolve().parent.glob("check_float64_tokenizer*.py"):
        if digest(path) == TOKENIZER_SCRIPT_SHA:
            spec = importlib.util.spec_from_file_location("verified_float64_helper", path)
            module = importlib.util.module_from_spec(spec)
            spec.loader.exec_module(module)
            return module
    raise ValueError("Keep the previous check_float64_tokenizer.py upload in the same Colab folder.")


def prepare(work, output):
    inputs = read(work / "inputs.json")
    embedding = read(work / "compact-model/embedding-manifest.json")
    stores = {}
    for language in LANGUAGES:
        manifest_path = Path(inputs["indexes"][language]).resolve()
        metadata = read(manifest_path)
        if (metadata.get("schema_version") != 1 or metadata.get("artifact_kind") != "subarag_hnsw"
                or metadata.get("language") != language or metadata.get("dimension") != 384
                or metadata.get("model_name") != embedding["model_name"]
                or metadata.get("model_revision") != embedding["model_revision"]
                or set(metadata["files"]) != {"documents.jsonl", "index.faiss"}):
            raise ValueError("Incompatible retrieval artifact")
        for name, item in metadata["files"].items():
            path = (manifest_path.parent / name).resolve()
            if path.parent != manifest_path.parent or path.stat().st_size != item["bytes"] or digest(path) != item["sha256"]:
                raise ValueError("Retrieval artifact checksum mismatch")
        database = output / f"{language}.sqlite"
        if database.exists():
            raise ValueError("Use a new diagnostic output folder; do not overwrite a store")
        connection = sqlite3.connect(database)
        try:
            connection.execute("PRAGMA cache_size=-512")
            connection.execute("CREATE TABLE documents (position INTEGER PRIMARY KEY, id TEXT NOT NULL UNIQUE, payload BLOB NOT NULL)")
            count = 0
            with (manifest_path.parent / "documents.jsonl").open("rb") as handle:
                for position, payload in enumerate(handle):
                    row = json.loads(payload)
                    if (row["language"] != language or not isinstance(row["id"], str)
                            or not isinstance(row["text"], str) or not isinstance(row.get("source", {}), dict)):
                        raise ValueError("Invalid document record")
                    connection.execute("INSERT INTO documents VALUES (?, ?, ?)", (position, row["id"], payload))
                    count += 1
            if count != metadata["document_count"]:
                raise ValueError("Document count differs from index manifest")
            connection.commit()
            if connection.execute("PRAGMA integrity_check").fetchone()[0] != "ok":
                raise ValueError("Document store integrity check failed")
            # Compare every saved byte payload and position, not just IDs/counts.
            cursor = connection.execute("SELECT position, payload FROM documents ORDER BY position")
            with (manifest_path.parent / "documents.jsonl").open("rb") as handle:
                for position, payload in enumerate(handle):
                    if cursor.fetchone() != (position, payload):
                        raise ValueError("Document payload/order changed during conversion")
            if cursor.fetchone() is not None:
                raise ValueError("Extra document rows after conversion")
        finally:
            connection.close()
        stores[language] = {"manifest_path": str(manifest_path), "manifest_sha256": digest(manifest_path),
                            "database": database.name, "database_sha256": digest(database),
                            "count": count, "payloads_verified_byte_for_byte": True}
        print(f"{language}: stored and verified {count} documents; index unchanged.", flush=True)
    write(output / "stores.json", stores)


class DiskDocuments(Sequence):
    def __init__(self, path, count):
        self.count = count
        self.connection = sqlite3.connect(path.resolve().as_uri() + "?mode=ro&immutable=1", uri=True)
        self.connection.execute("PRAGMA cache_size=-512")
        self.connection.execute("PRAGMA mmap_size=0")

    def __len__(self):
        return self.count

    @staticmethod
    def decode(payload):
        from subarag.retrieval.models import Document
        row = json.loads(payload)
        return Document(row["id"], row["language"], row["text"], row.get("source", {}))

    def __getitem__(self, position):
        if isinstance(position, slice):
            return [self[index] for index in range(*position.indices(self.count))]
        if position < 0:
            position += self.count
        if not 0 <= position < self.count:
            raise IndexError(position)
        row = self.connection.execute("SELECT payload FROM documents WHERE position=?", (position,)).fetchone()
        if row is None:
            raise ValueError("Missing document position")
        return self.decode(row[0])

    def __iter__(self):
        # The existing evaluation scans IDs once per language. Stream records;
        # never materialize the whole corpus as a Python document collection.
        for (payload,) in self.connection.execute("SELECT payload FROM documents ORDER BY position"):
            yield self.decode(payload)

    def close(self):
        self.connection.close()


class IndexPool:
    """Single-request diagnostic pool; not a concurrent serving implementation."""

    def __init__(self, metadata, single):
        self.metadata, self.single = metadata, single
        self.loaded, self.events = {}, []

    def get(self, language):
        if language not in self.loaded:
            import faiss
            started = perf_counter()
            if self.single:
                self.loaded.clear()  # Release the previous index BEFORE reading another.
                gc.collect()
            before = memory()
            path, manifest = self.metadata[language]
            index = faiss.read_index(str(path.parent / "index.faiss"))
            if (not hasattr(index, "hnsw") or index.ntotal != manifest["document_count"]
                    or index.d != manifest["dimension"] or index.metric_type != faiss.METRIC_INNER_PRODUCT):
                raise ValueError("Unexpected HNSW shape or metric")
            index.hnsw.efSearch = 64
            self.loaded[language] = index
            self.events.append({"language": language, "load_ms_including_release": (perf_counter() - started) * 1000,
                                "resident_index_count": len(self.loaded), "before_load": before, "after_load": memory()})
        return self.loaded[language]


class DiskIndex:
    def __init__(self, language, documents, pool):
        self.language, self.documents, self.pool = language, documents, pool

    def search(self, embedding, k):
        import numpy as np
        from subarag.retrieval.dense import normalized_float32
        from subarag.retrieval.models import RetrievalHit
        if k < 1:
            raise ValueError("k must be positive")
        query = normalized_float32(np.asarray(embedding).reshape(1, -1))
        scores, positions = self.pool.get(self.language).search(query, min(k, len(self.documents)))
        return [RetrievalHit(self.documents[int(position)], float(score), rank, "dense")
                for rank, (position, score) in enumerate(zip(positions[0], scores[0], strict=True), 1) if position >= 0]


def measure(work, output, mode):
    f64 = tokenizer_helper()
    helper = f64.load_helper()
    tokenizer_dir = Path(read(output / "screen-inputs.json")["tokenizer_report"]).parent
    stores = read(output / "stores.json")
    metadata, documents = {}, {}
    # Read/check files before deserialization. Hashing uses bounded chunks.
    for language, item in stores.items():
        path = Path(item["manifest_path"])
        if digest(path) != item["manifest_sha256"] or digest(output / item["database"]) != item["database_sha256"]:
            raise ValueError("Document/index metadata changed")
        manifest = read(path)
        if digest(path.parent / "index.faiss") != manifest["files"]["index.faiss"]["sha256"]:
            raise ValueError("Index checksum mismatch")
        metadata[language] = (path, manifest)
        documents[language] = DiskDocuments(output / item["database"], item["count"])
    pool = IndexPool(metadata, single=mode == "one-index")
    handles = {lang: DiskIndex(lang, documents[lang], pool) for lang in LANGUAGES}
    from subarag import retrieval
    from subarag.retrieval.index_artifact import LoadedRetrievalArtifact
    from subarag.experiments import compact_embedding as experiment
    import faiss
    faiss.omp_set_num_threads(1)

    def artifact(path, *, expected_model):
        path = Path(path).resolve()
        language = next(lang for lang, (source, _) in metadata.items() if source == path)
        manifest = metadata[language][1]
        if expected_model != manifest["model_name"]:
            raise ValueError("Wrong query model")
        if mode == "all-indexes":
            pool.get(language)
        return LoadedRetrievalArtifact(handles[language], manifest["corpus_version"], manifest["model_name"],
                                       manifest["retriever_version"], language, manifest["model_revision"])

    actual_embedder, embedder_used, result = experiment.OnnxQueryEmbedder, [], {}
    candidate_tokenizer = f64.tokenizer_class(helper)
    def capture_embedder(*args, **kwargs):
        instance = actual_embedder(*args, **kwargs)
        embedder_used.append(instance)
        return instance

    try:
        with patch.object(helper, "LightTokenizer", lambda _output: candidate_tokenizer(tokenizer_dir)), \
                patch.object(helper, "write", lambda _path, value: result.update(value)), \
                patch.object(retrieval, "load_hnsw_artifact", artifact), \
                patch.object(experiment, "OnnxQueryEmbedder", capture_embedder):
            helper.measure(work, output)
        # Extra interleaved requests expose language-switch cost. Time encoding,
        # index release/load, search, and payload reads together, not just FAISS.
        cases = read(work / "inputs.json")["cases"]
        first = {lang: next(case for case in cases if case["language"] == lang) for lang in LANGUAGES}
        mixed_ms, mixed_matches = [], True
        original = read(work / "candidate.json")
        for _ in range(3):
            for language in LANGUAGES:
                case = first[language]
                started = perf_counter()
                vector = embedder_used[0].embed_queries([case["query"]])[0]
                hits = handles[language].search(vector, 8)
                mixed_ms.append((perf_counter() - started) * 1000)
                mixed_matches &= [hit.document.id for hit in hits] == original["rankings"][language + ":" + case["id"]]
        result.update(runtime=mode, peak_rss_mib=memory()["peak_mib"],
                      load_order="verified disk stores, float64 tokenizer, query model, all indexes at startup or one lazily on request",
                      includes="web imports/startup, query encoder, disk documents, bounded indexes, short/512-token and interleaved-language probes",
                      mixed_language_ms=retrieval.latency_summary(mixed_ms), mixed_language_rankings_identical=mixed_matches,
                      index_loads=pool.events, max_resident_index_count=max(event["resident_index_count"] for event in pool.events),
                      concurrency="Sequential requests only; no concurrent-serving claim. OS file caches were not cleared.")
        if any(name in sys.modules for name in ("torch", "transformers", "sentence_transformers", "sentencepiece")):
            raise RuntimeError("Unexpected heavyweight runtime import")
        write(output / f"{mode}.json", result)
        print(mode, "final peak MiB:", round(result["peak_rss_mib"], 1), "switching p95 ms:", round(result["mixed_language_ms"]["p95"], 1), flush=True)
    finally:
        pool.loaded.clear()
        for store in documents.values():
            store.close()


def run(work):
    if not (work / "inputs.json").is_file():
        raise ValueError("The original Colab WORK files are missing. Stop and report the runtime reset; do not rebuild automatically.")
    tokenizer_helper()
    candidates = []
    for path in work.glob("float64-tokenizer-*/float64-tokenizer-report.json"):
        value = read(path)
        if (value.get("script_sha256") == TOKENIZER_SCRIPT_SHA and value["tokenizer_parity"]["exact_match"]
                and (value.get("full_pipeline") or {}).get("identical_vectors_to_original_uint8")
                and value["full_pipeline"]["identical_rankings_to_original_uint8"]):
            candidates.append(path)
    if not candidates:
        raise ValueError("Keep the completed passing float64-tokenizer run in the original WORK directory.")
    prior = max(candidates, key=lambda path: path.stat().st_mtime)
    baseline = read(prior)
    originals = {work / name: checksum for name, checksum in baseline["source_sha256"].items()}
    inputs = read(work / "inputs.json")
    for name in inputs["indexes"].values():
        path = Path(name).resolve()
        for source in (path, path.parent / "documents.jsonl", path.parent / "index.faiss"):
            originals[source] = digest(source)
    originals[prior] = digest(prior)
    manifest_path = prior.parent / "float64-manifest.json"
    originals[manifest_path] = digest(manifest_path)
    for name, checksum in read(manifest_path)["files"].items():
        if name not in ("vocabulary.marisa", "scores.bin", "pipeline.json"):
            raise ValueError("Unexpected tokenizer file")
        originals[prior.parent / name] = checksum
    if any(digest(path) != checksum for path, checksum in originals.items()):
        raise ValueError("An original input no longer matches the verified baseline")
    output = Path(tempfile.mkdtemp(prefix="disk-retrieval-", dir=work))
    write(output / "screen-inputs.json", {"tokenizer_report": str(prior)})
    print("Output:", output, flush=True)
    env = dict(os.environ, HF_HUB_OFFLINE="1", TRANSFORMERS_OFFLINE="1", TOKENIZERS_PARALLELISM="false",
               OMP_NUM_THREADS="1", OPENBLAS_NUM_THREADS="1")
    env.pop("HF_TOKEN", None)
    for mode in ("prepare", "all-indexes", "one-index"):
        subprocess.run([sys.executable, "-u", str(Path(__file__).resolve()), "--work", str(work),
                        "--output", str(output), "--child", mode], env=env, check=True)
    original, reference = read(work / "candidate.json"), read(work / "reference.json")
    variants = {}
    for mode in ("all-indexes", "one-index"):
        value = read(output / f"{mode}.json")
        if value["inputs_sha256"] != original["inputs_sha256"] or value["model_revision"] != original["model_revision"]:
            raise ValueError("Comparison provenance differs")
        value["identical_vectors_to_original_uint8"] = value.pop("vectors") == original["vectors"]
        value["identical_rankings_to_original_uint8"] = value.pop("rankings") == original["rankings"]
        value["within_400_mib"] = value["peak_rss_mib"] <= 400
        value["within_500_ms_grouped_p95"] = {lang: value["retrieval_ms"][lang]["p95"] <= 500 for lang in LANGUAGES}
        value["within_500_ms_switching_p95"] = value["mixed_language_ms"]["p95"] <= 500
        value["within_original_quality_gate"] = {
            lang: reference["metrics"][lang]["ndcg@8"] > 0 and value["metrics"][lang]["ndcg@8"] >=
            0.98 * reference["metrics"][lang]["ndcg@8"] for lang in LANGUAGES}
        variants[mode] = value
    if any(digest(path) != checksum for path, checksum in originals.items()):
        raise RuntimeError("An original artifact changed during the screen")
    write(output / "disk-retrieval-report.json", {
        "schema_version": 1, "diagnostic_only": True, "script_sha256": digest(__file__),
        "original_files_unchanged": True, "source_sha256": {str(path): value for path, value in originals.items()},
        "stores": read(output / "stores.json"), "variants": variants, "previous_full_pipeline": baseline["full_pipeline"],
        "limitation": "Colab process RSS, not a container guarantee. File/page caches and concurrent requests need separate validation. Index files and query model are unchanged; the existing Marathi quantization loss remains. No upload or deployment.",
    })
    print("REPORT:", output / "disk-retrieval-report.json", flush=True)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--work", type=Path, required=True)
    parser.add_argument("--output", type=Path)
    parser.add_argument("--child", choices=("prepare", "all-indexes", "one-index"))
    args = parser.parse_args()
    if args.child:
        if args.output is None:
            parser.error("Child stages require --output")
        if args.child == "prepare":
            prepare(args.work, args.output)
        else:
            measure(args.work, args.output, args.child)
    else:
        run(args.work.resolve())
