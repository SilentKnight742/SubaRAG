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

## First Colab result: rejected

The returned report matches source ZIP SHA-256
`7c603a83f17d9465d46d06c803a48dcb6868434636fa0da3938c1cd6d1205e16`.
The screen used identical frozen inputs for both runtimes (192 queries).

| Measure | FP32 reference | Quantized candidate |
| --- | ---: | ---: |
| Peak RSS | 1322.4 MiB | 874.9 MiB |
| English nDCG@8 | 0.75490 | 0.75786 |
| Hindi nDCG@8 | 0.54228 | 0.53454 |
| Marathi nDCG@8 | 0.51059 | 0.48810 |
| English retrieval p95 | 35.88 ms | 15.95 ms |
| Hindi retrieval p95 | 38.64 ms | 17.63 ms |
| Marathi retrieval p95 | 47.01 ms | 18.08 ms |

The candidate was faster but failed the 400 MiB memory gate and the Marathi
quality gate (4.4% relative nDCG decrease, versus the permitted 2%). It was **not**
uploaded or enabled. The report alone does not identify which stage caused peak
memory or prove which quantized operator caused the quality change.

## Follow-up without rebuilding or downloading

Upload only `scripts/diagnose_compact_runtime.py` to the **same Colab runtime**
that ran the compact notebook. Keep its original `WORK` variable. Run:

```python
from google.colab import files
from pathlib import Path
import subprocess, sys

uploaded = files.upload()  # select diagnose_compact_runtime.py only
if len(uploaded) != 1:
    raise ValueError("Upload just the diagnostic script.")
script = next(iter(uploaded))  # accepts Colab-renamed filenames
subprocess.run([sys.executable, "-u", script, "--work", str(WORK)],
               env=child_env, check=True)
report = max(Path(WORK).glob("memory-diagnostic-*/memory-diagnostic.json"),
             key=lambda path: path.stat().st_mtime)
files.download(str(report))
```

The script launches isolated processes for three measurements:

1. The current candidate, recording RSS/high-water marks after imports, ONNX
   session creation, tokenizer parsing, document loading, native index loading,
   and short/long query execution.
2. The same candidate with the tokenizer loaded before the ONNX session. This
   tests transient allocation overlap without changing numerical precision.
3. Matrix-only quantization built from the existing `e5-fp32.onnx`, retaining the
   full-precision embedding table. Matrix quantization parameters stay unchanged
   to isolate the effect of embedding-table quantization on Marathi quality.
   This larger model is diagnostic, not a proposed free-tier deployment.

All three reuse the exact frozen queries and indexes, with no repeat export or
model download. The new model and reports go in a unique diagnostic directory;
original `candidate.json`, `reference.json`, model files, and indexes are untouched.
Child processes use offline mode and no HF token. Share `memory-diagnostic.json`
and keep the runtime open. If that runtime has been lost, stop and report that
instead of restarting the entire build automatically.

Stage snapshots cannot be summed into exact component sizes: allocator reuse,
temporary native buffers, and retained memory affect RSS. We use them to locate
large increases, not to claim precise ownership or container safety. The new
report keeps the original failed result and unchanged quality thresholds.

## Memory follow-up result: still rejected

The returned diagnostic matches script SHA-256
`64eb0a0ecf815175470556c2844613f978e11fc193264d4e87892fc4f33e89f2`.

| Variant | Peak RSS | Marathi nDCG change vs reference |
| --- | ---: | ---: |
| Original compact runtime | 876.6 MiB | -4.4% |
| Load tokenizer first | 778.2 MiB | -4.4% |
| Keep embedding table FP32 | 1127.9 MiB | -6.8% |

In the original load order, RSS rose approximately 268 MiB when loading the
tokenizer and another 188 MiB while loading documents/indexes. Long queries
increased peak memory further. These are stage deltas, not exact owned sizes.
Load order alone does not meet the memory gate. Restoring embedding-table
precision did not recover Marathi quality, so that quantization step cannot
be assumed to be the sole cause. The live preview remains unchanged.

## Full-precision and tokenizer comparison

Upload only `scripts/check_onnx_parity.py` into the **same** Colab runtime. This
uses the retained `e5-fp32.onnx`, saved reference vectors, frozen queries, and
existing indexes; no export, model download, credentials, or index rebuild.

```python
from google.colab import files
from pathlib import Path
import subprocess, sys

uploaded = files.upload()  # select check_onnx_parity.py only
if len(uploaded) != 1:
    raise ValueError("Upload just the comparison script.")
subprocess.run([sys.executable, "-u", next(iter(uploaded)), "--work", str(WORK)],
               env=child_env, check=True)
report = max(Path(WORK).glob("onnx-parity-*/onnx-parity-report.json"),
             key=lambda path: path.stat().st_mtime)
files.download(str(report))
```

The script compares FP32 ONNX embeddings and top-8 retrieval against the saved
Sentence Transformers reference on all 192 frozen queries, alongside the
original quantized candidate. Per-language cosine similarity, vector errors,
ranking agreement, and nDCG help distinguish pre-quantization differences
from quantization effects. High cosine alone does not guarantee equal rankings.

Two isolated tokenizer processes check exact token IDs and attention masks
between the saved raw tokenizer and cached pinned Hugging Face AutoTokenizer,
including 11 additional whitespace, Unicode, special-token, and long-input
cases. This checks that boundary, not every Sentence Transformers preprocessing
detail or universal equivalence. Tokenizer vocabulary and processing metadata
are recorded to guide a later lighter implementation; no replacement is made
before its own exact-token comparison. Missing cached files stop the check
instead of downloading anything. Heavy tokenizer-wrapper imports never enter
the separate inference measurement process.

New artifacts go into a unique `onnx-parity-*` folder. The FP32 graph is hardlinked
there when supported (copied otherwise); no writes are made to the graph.
Original input/report/tokenizer/model hashes are checked before and after.
Share `onnx-parity-report.json` and keep Colab open. This is diagnostic work,
not a new serving candidate or permission to upload/deploy.

## Full-precision comparison result: quality preserved

The returned report matches script SHA-256
`c9c9d79bf20c5b636a4ae65afc14bd9ea84157b23e0044cd87f4d56ea50a3cc0`
and the original frozen input fingerprint. FP32 ONNX returned identical ordered
top-8 results for all 192 queries, with unchanged per-language nDCG. Token IDs
and attention masks matched the pinned AutoTokenizer on all 203 checks.
This isolates the observed Marathi loss to the quantized runtime on this screen,
not the FP32 export or that tokenizer boundary. FP32 still used 1295.6 MiB peak
RSS; its quality pass does not solve hosting.

## Lighter tokenizer screen

`scripts/check_light_tokenizer.py` tests a different in-memory representation of
the same vocabulary. It converts the saved Unigram vocabulary and scores into a
SentencePiece binary, preserving IDs and recording any float32 score rounding.
A tiny Rust tokenizer retains the original
normalizer and pre-tokenizer; it does not load the 250,002-entry Rust Unigram
model. SentencePiece receives already normalized/split pieces with its own
normalization disabled. Added special tokens, boundary tokens, and right
truncation at 512 remain explicit. This is experimental, not a claim of universal
equivalence or a selected serving adapter.

The implementation uses the official
[SentencePiece model schema](https://github.com/google/sentencepiece/blob/master/src/sentencepiece_model.proto),
[Python encoding API](https://github.com/google/sentencepiece/blob/master/python/README.md),
and Hugging Face [normalizer](https://huggingface.co/docs/tokenizers/api/normalizers)
and [pre-tokenizer](https://huggingface.co/docs/tokenizers/api/pre-tokenizers) APIs.

In the **same Colab runtime**, run the following new cell and upload only
`check_light_tokenizer.py`. The small SentencePiece package may be downloaded;
the script itself is offline and does not fetch models, datasets, or credentials.
Existing protobuf/tokenizers/ONNX packages from the notebook are reused.

```python
from google.colab import files
from pathlib import Path
import subprocess, sys

subprocess.run([sys.executable, "-m", "pip", "install", "sentencepiece==0.2.1"],
               check=True)
uploaded = files.upload()
if len(uploaded) != 1:
    raise ValueError("Upload only check_light_tokenizer.py.")
with subprocess.Popen(
    [sys.executable, "-u", next(iter(uploaded)), "--work", str(WORK)],
    env=child_env, stdout=subprocess.PIPE, stderr=subprocess.STDOUT,
    text=True, bufsize=1,
) as process:
    for line in process.stdout:
        print(line, end="", flush=True)
    if process.wait() != 0:
        raise RuntimeError("Diagnostic failed; share the actual error printed above.")
report = max(Path(WORK).glob("light-tokenizer-*/light-tokenizer-report.json"),
             key=lambda path: path.stat().st_mtime)
files.download(str(report))
```

Conversion, original tokenization, candidate tokenization, and full inference
run in separate processes. The screen compares all 192 frozen queries, 43 edge
cases, and 200 seeded mixed-Unicode cases (435 total). Any token/mask mismatch
rejects the candidate and skips full inference, while still producing a report.
Only an exact pass proceeds to measure the existing uint8 model with all three
indexes and short/512-token probes; it also checks vectors and rankings against
the original uint8 result. The unchanged quantization still has its known Marathi
quality failure, even if tokenizer memory improves. Preparation memory is not
counted as inference memory, and no allocator trimming is used to hide a peak.

Reports and converted tokenizer files go in a fresh `light-tokenizer-*` directory.
Original input/report/model/tokenizer hashes are checked before and after. Share
`light-tokenizer-report.json`; do not upload the artifacts or deploy them.

The first conversion attempt stopped before inference because the helper assumed
`<mask>` had `lstrip=True`. Inspection of the actual saved tokenizer confirmed
`lstrip=False` for all five special tokens. The helper now records and follows
the saved flag, accepting either mask setting while retaining the other
special-token validation and exact-output gate. Re-upload the corrected helper;
existing models/indexes and previous diagnostic folders need no deletion.

The next attempt stopped at the helper's requirement that every vocabulary score
remain bit-exact after conversion. The saved scores do not all round-trip through
SentencePiece's float32 score field. The diagnostic now records the number of
changed scores, maximum absolute/relative error, and up to five numeric examples
in `light_manifest.score_conversion`. Non-finite scores and float32 overflow
still stop conversion. Rounding is not assumed harmless: the unchanged exact
token-ID/attention-mask gate must pass before any full-pipeline measurement.
Even a finite-suite pass does not prove equivalence for all possible inputs.

## Lighter tokenizer result: one mismatch, not accepted

The returned report matches script SHA-256
`b8b66f5b22ddd80be600400324b0df8b9002de4eb523f1900f76c9cb66b846f8`.
All 192 retrieval queries and 43 edge cases matched, along with 199 of the 200
seeded synthetic cases. `fuzz:7` differed in token order. The exact-token gate
correctly skipped full inference; this candidate remains rejected.

Tokenizer-only process peak RSS fell from 328.0 MiB to 102.4 MiB. This is not
the full pipeline's memory use and does not establish a hosting fit. Conversion
rounded 65,856 vocabulary scores, with maximum absolute error approximately
3.55e-15. That small storage error alone does not identify the mismatch's cause.

### Focused mismatch diagnosis (no embedding model or indexes)

`scripts/diagnose_tokenizer_mismatch.py` reuses the completed failed report,
converted tokenizer, and exact previously uploaded helper (verified by hash).
It reproduces the synthetic input, compares core segmentation on identical
normalized pieces, and checks that those pieces reconstruct the full tokenizer
outputs. It then reduces a differing piece through bounded deletion checks and
constructs a tiny vocabulary retaining every possible substring match plus
unknown/minimum-score entries. Reduced vocabularies must reproduce both full
models before their comparison can be interpreted.

The diagnostic compares the Rust segmenter using original and float32-rounded
scores against SentencePiece using those same rounded scores. It reports token
pieces, IDs, scores, and cumulative path sums without changing either tokenizer.
The implementation sources show a relevant hypothesis: [Tokenizers 0.23.2 uses
f64 path scores](https://github.com/huggingface/tokenizers/blob/v0.23.2/tokenizers/src/models/unigram/model.rs),
whereas [SentencePiece 0.2.1 uses float path scores](https://github.com/google/sentencepiece/blob/v0.2.1/src/unigram_model.cc).
This numerical difference is not yet a confirmed explanation for `fuzz:7`.
Path totals alone cannot replace examining competing segmentation decisions.

In the same Colab runtime, upload only `diagnose_tokenizer_mismatch.py` with this
cell. Keep the prior `check_light_tokenizer.py` upload (a renamed copy is fine).
No package install, model inference, corpus/index loading, or download is required.
Only frozen query metadata/tokenizer files and prior diagnostic reports are read;
the report exports synthetic inputs only, not dataset query text.

```python
from google.colab import files
from pathlib import Path
import subprocess, sys

uploaded = files.upload()
if len(uploaded) != 1:
    raise ValueError("Upload only diagnose_tokenizer_mismatch.py.")
with subprocess.Popen(
    [sys.executable, "-u", next(iter(uploaded)), "--work", str(WORK)],
    env=child_env, stdout=subprocess.PIPE, stderr=subprocess.STDOUT,
    text=True, bufsize=1,
) as process:
    for line in process.stdout:
        print(line, end="", flush=True)
    if process.wait() != 0:
        raise RuntimeError("Share the actual error printed above.")
report = max(Path(WORK).glob("tokenizer-mismatch-*/tokenizer-mismatch-report.json"),
             key=lambda path: path.stat().st_mtime)
files.download(str(report))
```

Share `tokenizer-mismatch-report.json`. The diagnostic writes a unique report
folder, verifies original files are unchanged, and does not retry full inference
or relax the failed parity gate. No serving changes, uploads, or deployment.

## Mismatch isolated and float64 replacement prepared

The returned mismatch report matches diagnostic script SHA-256
`676058399dddbba23a10164a8f95c96f74704a827f1c9fd9cb90eff341fccfc5`.
Both core reconstructions matched their full tokenizers. The differing piece
reduced to `?!?!?!`: the original chose `?!?` + `!` + `?!`, while SentencePiece
chose `?!` + `?!?` + `!`. A six-entry vocabulary reproduced both outputs, even
with identical stored scores. A local best-path replay using float64 versus
float32 accumulation reproduced the respective choices. This identifies an
arithmetic-dependent segmentation difference for this case, not normalization
or storage-rounding loss alone.

`scripts/check_float64_tokenizer.py` replaces only the candidate's segmenter:

- Preserve vocabulary IDs and reference-runtime binary64 scores, with a
  serialized round-trip check for every entry. Preparation captures scores from
  the loaded reference tokenizer to avoid assuming Python and Rust JSON parsers
  round every decimal identically; any parser differences are counted.
- Store vocabulary strings in a compact MARISA trie and original IDs/scores in
  a separate fixed-width table. Both are read-only memory maps. Separating binary
  records from keys also handles embedded control characters in input safely.
- Compute deterministic best paths with binary64 sums, left-to-right traversal,
  strict-greater score replacement, original unknown-token fallback/fusion, and
  no hard-coded punctuation exception or unbounded cache.
- Reuse the previously checked normalization/special-token/truncation wrapper
  from the hash-verified `check_light_tokenizer.py` upload. SentencePiece is not
  imported by the new inference process.

This uses [MARISA persistence and prefix lookup](https://marisa-trie.readthedocs.io/en/latest/tutorial.html)
and follows the original [Tokenizers 0.23.2 segmentation behavior](https://github.com/huggingface/tokenizers/blob/v0.23.2/tokenizers/src/models/unigram/model.rs).
Memory mapping is not free container memory: resident file pages still count.
Python-side scoring may affect latency, so no memory/latency improvement is
claimed until the real screen runs. The small local reproducer and end-to-end
fixture pass, including the unchanged rejection path for token mismatches.

In the same Colab runtime, run this new cell and upload only
`check_float64_tokenizer.py`. Keep the earlier helper uploads and reports.
The small MARISA package is the only new download; models and indexes are reused.

```python
from google.colab import files
from pathlib import Path
import subprocess, sys

subprocess.run([sys.executable, "-m", "pip", "install", "marisa-trie==1.4.1"], check=True)
uploaded = files.upload()
if len(uploaded) != 1:
    raise ValueError("Upload only check_float64_tokenizer.py.")
with subprocess.Popen(
    [sys.executable, "-u", next(iter(uploaded)), "--work", str(WORK)],
    env=child_env, stdout=subprocess.PIPE, stderr=subprocess.STDOUT,
    text=True, bufsize=1,
) as process:
    for line in process.stdout:
        print(line, end="", flush=True)
    if process.wait() != 0:
        raise RuntimeError("Share the actual error printed above.")
report = max(Path(WORK).glob("float64-tokenizer-*/float64-tokenizer-report.json"),
             key=lambda path: path.stat().st_mtime)
files.download(str(report))
```

Preparation, raw tokenization, candidate tokenization, and inference run in
separate processes. The unchanged 435-input suite plus 64 punctuation-family
regressions must all match (499 total). Only then run the existing full-index
uint8 measurement, compare vectors/rankings, and report memory and latency.
The model's separate Marathi quality failure is not fixed by this tokenizer
change. Original files stay intact; new outputs use a `float64-tokenizer-*`
folder. Share `float64-tokenizer-report.json`; nothing is uploaded or deployed.

## Float64 result and disk-backed retrieval screen

The returned float64 report matches script SHA-256
`4d4e39bf27366b93e05963d0e9debca53901afb19b83d981428197f9de8424a1`.
All 499 tokenization checks passed, and all 192 query embeddings and ordered
rankings matched the original uint8 candidate. Tokenizer-only peak RSS fell
from 328.9 to 40.4 MiB. Full-pipeline peak was still **566.6 MiB**, above both
the 400 MiB screening target and the free host's 512 MiB limit. Per-language
retrieval p95 was approximately 18–24 ms. The unchanged quantized model still
lost about 4.4% Marathi nDCG against the full-precision reference; the tokenizer
fix does not resolve that separate quality gate.

`scripts/check_disk_retrieval.py` now screens document/index residency separately:

- Convert each existing document JSONL into a new SQLite store, preserving and
  checking every original byte payload and its index position. Reads materialize
  only matched documents; evaluation's ID scan streams records.
- Measure disk-backed documents with all three original HNSW indexes loaded.
- Measure the same stores with only one language's index retained, releasing the
  previous index before loading another. No vectors or indexes are rebuilt.
- Compare all frozen embeddings and ordered rankings, and time nine additional
  interleaved-language requests including encoding, index switching, search,
  and document reads. Recheck original file hashes after both runs.

The one-index variant is a sequential diagnostic, not a concurrent server
implementation. Its first request includes lazy index loading; its startup
number therefore does not represent the same readiness point as eager loading.
OS file caches are not cleared, and validation/ID scans warm files: this is not
a cold-disk benchmark. Process RSS alone does not establish container fit;
filesystem caches and concurrent requests need separate validation. Memory and
latency results must pass without relaxing the existing quality gate.

The tiny local end-to-end fixture passed for both variants: document contents,
metadata and order, embeddings, rankings, language switches, index-count bounds,
and unchanged original files. Its memory/timings are not evidence of real-corpus
hosting fit. Production serving code is unchanged.

In the **same Colab runtime**, keep `WORK`, `child_env`, earlier helper uploads,
and the completed float64 run. No installs, model downloads, index rebuilds,
credentials, or uploads to Hugging Face are needed. If that runtime/files are
gone, stop and report it instead of rebuilding automatically.

Run this cell and upload only `check_disk_retrieval.py`:

```python
from google.colab import files
from pathlib import Path
import subprocess, sys

uploaded = files.upload()
if len(uploaded) != 1:
    raise ValueError("Upload only check_disk_retrieval.py.")
with subprocess.Popen(
    [sys.executable, "-u", next(iter(uploaded)), "--work", str(WORK)],
    env=child_env, stdout=subprocess.PIPE, stderr=subprocess.STDOUT,
    text=True, bufsize=1,
) as process:
    for line in process.stdout:
        print(line, end="", flush=True)
    if process.wait() != 0:
        raise RuntimeError("Share the actual error printed above.")
report = max(Path(WORK).glob("disk-retrieval-*/disk-retrieval-report.json"),
             key=lambda path: path.stat().st_mtime)
files.download(str(report))
```

Share `disk-retrieval-report.json`. New stores and reports stay in a unique
`disk-retrieval-*` directory; the source corpus, model, indexes, and previous
reports remain intact. Nothing is pushed, uploaded, or deployed by this helper.
