"""Experimental E5 query encoder: tokenizers + ONNX Runtime, without PyTorch."""

from __future__ import annotations

import hashlib
import json
from pathlib import Path
from collections.abc import Sequence


def file_sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        while chunk := handle.read(1024 * 1024):
            digest.update(chunk)
    return digest.hexdigest()


class OnnxQueryEmbedder:
    """One query at a time; the exported graph owns mean pooling and L2 normalization.

    Not selected by application settings until the remote memory/quality gate passes.
    """

    def __init__(self, directory: str | Path, *, expected_revision: str) -> None:
        import numpy as np
        import onnxruntime as ort
        from tokenizers import Tokenizer

        root = Path(directory).resolve()
        metadata = json.loads((root / "embedding-manifest.json").read_text())
        if (metadata.get("schema_version") != 1
                or metadata.get("model_name") != "intfloat/multilingual-e5-small"
                or metadata.get("model_revision") != expected_revision
                or metadata.get("dimension") != 384
                or metadata.get("max_length") != 512
                or metadata.get("pooling") != "masked-mean-l2"
                or metadata.get("query_prefix") != "query: "
                or set(metadata.get("files", {})) != {"model.onnx", "tokenizer.json"}):
            raise ValueError("Incompatible compact embedding artifact")
        for name, item in metadata["files"].items():
            path = (root / name).resolve()
            if path.parent != root or path.stat().st_size != item["bytes"] or file_sha256(path) != item["sha256"]:
                raise ValueError("Compact embedding artifact checksum mismatch")
        options = ort.SessionOptions()
        options.intra_op_num_threads = 1
        options.inter_op_num_threads = 1
        options.enable_cpu_mem_arena = False
        options.enable_mem_pattern = False
        self.session = ort.InferenceSession(
            str(root / "model.onnx"), sess_options=options, providers=["CPUExecutionProvider"],
        )
        if {item.name for item in self.session.get_inputs()} != {"input_ids", "attention_mask"}:
            raise ValueError("Unexpected ONNX model inputs")
        self.tokenizer = Tokenizer.from_file(str(root / "tokenizer.json"))
        self.tokenizer.no_padding()
        self.tokenizer.enable_truncation(max_length=metadata["max_length"])
        self.name = metadata["model_name"]
        self.dimension = metadata["dimension"]
        self._np = np

    def embed_queries(self, texts: Sequence[str]):
        np = self._np
        vectors = []
        for text in texts:
            encoded = self.tokenizer.encode("query: " + text)
            vector = self.session.run(["sentence_embedding"], {
                "input_ids": np.asarray([encoded.ids], dtype=np.int64),
                "attention_mask": np.asarray([encoded.attention_mask], dtype=np.int64),
            })[0]
            if vector.shape != (1, self.dimension) or not np.isfinite(vector).all():
                raise ValueError("Invalid compact query embedding")
            norm = np.linalg.norm(vector, axis=1, keepdims=True)
            if np.any(norm == 0):
                raise ValueError("Zero-length compact query embedding")
            vectors.append((vector / norm)[0])
        return np.asarray(vectors, dtype=np.float32).reshape(-1, self.dimension)
