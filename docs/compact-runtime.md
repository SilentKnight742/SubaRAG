# Compact query-embedding screen (not yet selected for serving)

The existing PyTorch E5 runtime has not passed the free Render memory budget.
This experiment keeps the same pinned E5 model and existing passage indexes,
and changes only query encoding to ONNX Runtime with 8-bit matrix weights and
embedding lookups. It is not enabled by the app, Dockerfile, or Render Blueprint.

## Run in Colab

Upload `notebooks/check_compact_runtime.ipynb` to a fresh free CPU runtime. When prompted,
upload the committed source ZIP `SubaRAG-compact-source.zip`, then enter the
existing dataset-scoped **read** token. Do not enter a write token, Gemini key, or
ElevenLabs key. The notebook reads the pinned prepared corpus and retrieval
bundle from `silentkirmada/SubaRAG-data`; it never touches the upstream Parquet.

It downloads roughly 214 MiB of prepared corpus/index files plus the public E5
model and dependencies to Colab. No model work runs on the developer VM. It does
not reuse another notebook's Python imports. Each export/measurement stage runs
in a fresh child process, so conversion's memory does not pollute inference RSS.
If the runtime disconnects, the already uploaded corpus/indexes remain safe;
only this experiment's temporary outputs would need rebuilding.

Run cells in order and share the small `compact-report.json`. Keep the runtime
open; do not upload the experimental model yet. An export error or failed gate
is a result to inspect, not permission to weaken a gate or upgrade hosting.

## What is measured

- The exact E5 model commit used for passage indexing:
  `614241f622f53c4eeff9890bdc4f31cfecc418b3`.
- First verify FP32 ONNX parity with Sentence Transformers on three multilingual
  probes (minimum cosine 0.9999), including uneven padded sequence lengths.
- Quantize both constant matrix weights and embedding lookups. Quantizing only
  transformer matrix weights leaves the large vocabulary embedding table intact.
- Freeze 64 validation queries per language, chosen deterministically by query-ID
  hash. Compare top-8 nDCG, recall, MRR, and query embedding + search latency against
  the existing unquantized encoder using the same three full indexes.
- Record Linux `VmHWM` (current process peak RSS), including native allocations, all three indexes,
  document objects, tokenizer, model, API imports/client startup, and repeated
  short/512-token probes. Candidate processes reject accidental PyTorch imports.
  `getrusage` is also reported for diagnosis but is not the gate: it can retain a
  launcher's pre-exec high-water mark and overstate candidate memory.
- Load the query model before the indexes to avoid overlapping temporary model
  construction buffers with corpus memory. Both measurements use this order;
  a promoted serving runtime must adopt it too (the current PyTorch serving
  builder still loads indexes first).

Predeclared **screen**, not deployment, gates:

| Check | Required |
| --- | --- |
| Peak process RSS | At most 400 MiB, leaving room below a 512 MiB container |
| Per-language nDCG@8 | At least 98% of that language's nonzero FP32 baseline |
| Per-language warm retrieval p95 | At most 500 ms on the measured Colab CPU |
| Heavy runtime imports | None in the compact process |

The report preserves per-language failures. Cosine similarity is reported but
does not substitute for retrieval-quality measurement. This is a 192-query
retrieval screen, **not** answer correctness, generation latency, or an evaluation
of how accuracy scales to a larger corpus.

Colab CPU speed is not Render CPU speed. Process RSS excludes container page
cache and sibling processes; this run also excludes concurrent transcription,
live Gemini requests, response buffering, and worst-case service concurrency.
Even a pass requires a memory-limited serving-container check and end-to-end
answer verification before deployment. If it fails, keep the working preview
unchanged and investigate another free serving arrangement.

The approach follows the official [ONNX Runtime quantization API](https://onnxruntime.ai/docs/performance/model-optimizations/quantization.html)
and its [Gather quantizer](https://github.com/microsoft/onnxruntime/blob/main/onnxruntime/python/tools/quantization/operators/gather.py).
