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

## Disk-backed result and restart checkpoint

The returned disk report matches script SHA-256
`5ebc4544ae6b4ad5b692c00ce4b6c0f15901f14c395899c6cb7c39f7f0b6d63d`.
In the rebuilt runtime, the float64 baseline peaked at 577.2 MiB. Disk documents
with all indexes peaked at 524.5 MiB; retaining one index peaked at 455.0 MiB.
Both preserved all 192 embeddings and ordered rankings, all document payloads,
and the interleaved-query rankings. One-index grouped retrieval p95 was 13–17 ms;
the nine-query language-switch screen had p95 114.2 ms. Neither meets the 400 MiB
memory target, and the unchanged model still fails Marathi's quality threshold.

The next index-load event after long-query probes records a peak increase from
339.1 to 455.0 MiB in the one-index process. Long-query inference is a plausible
contributor, not a proven isolated cause; stage-level profiling is still needed.
No serving changes or relaxed limits follow from this report.

Before further experiments, `scripts/checkpoint_colab.py` saves a private local
checkpoint outside `WORK`. It includes all experiment model/index/store/report
files, the extracted backend Python source and package metadata, and the three
hash-verified tokenizer/retrieval helpers. The archive records original paths
and installed package versions, verifies every archived file's checksum, and
checks that inputs did not change during backup. Unknown experiment file types
and symlinks stop the backup for inspection rather than being silently followed.
It does not collect environment variables, environment files, account caches,
or the installed Python environment. No package installation, credential prompt,
model execution, or cloud upload is involved.

Keep the archive private: it includes private dataset text. Do not commit it to
GitHub or attach the large archive in chat. A local tiny-fixture round trip,
checksum-corruption rejection, and environment/cache exclusions passed; the real
archive must still be downloaded and retained by the user. Recording dependency
versions is not a guarantee that a future Colab image will reproduce that Python
environment automatically. Restoration must preserve the recorded work/source
paths because existing reports contain absolute paths.

In the current Colab notebook, add a new cell and upload only
`checkpoint_colab.py`. Use the existing `run_logged`, `WORK`, and `child_env`:

```python
from google.colab import files
from pathlib import Path
import json, sys

uploaded = files.upload()
if len(uploaded) != 1:
    raise ValueError("Upload only checkpoint_colab.py.")
run_logged(
    [sys.executable, "-u", next(iter(uploaded)), "--work", str(WORK)],
    env=child_env,
)
report = max(
    Path(WORK).parent.glob("subarag-checkpoint-*/checkpoint-report.json"),
    key=lambda path: path.stat().st_mtime,
)
details = json.loads(report.read_text())
files.download(details["archive"])
files.download(str(report))
```

Wait for both browser downloads to finish. Keep
`SubaRAG-private-checkpoint.tar.gz` on the device; share only the small
`checkpoint-report.json` and confirm the archive downloaded. A successful Colab
archive verification alone does not prove the browser download completed.

## Checkpoint verified; profile the remaining peak

The downloaded checkpoint contains 107 files and is 537,621,418 bytes. Its local
SHA-256 matches the Colab verification report:
`4e014739505d92bf6213fa099d26917e0aff0cca08ef97d74f700e8fa39afad7`.
It preserves `/content/subarag-compact-rdjzfmcy` and records the original source
path. Keep this private archive; no cloud upload or production promotion occurred.

`scripts/profile_query_memory.py` reuses the hash-verified disk helper and its
one-index stores. It adds before/after RSS and process high-water measurements
around imports, compact-tokenizer loading, encoder/session construction,
tokenization, ONNX inference, index access, and search/document reads. It repeats
the existing schedule unchanged: three warmups, 192 frozen queries, six long
probes alternating with six short probes, and nine language-switch requests.
The profile records encoded token counts and phase labels, not query text.
The 512-token ceiling, model, indexes, and quality limits are unchanged.

Before running, the helper verifies prior source/store checksums. It redirects
the reused helper's output into a new `query-memory-*` directory rather than
overwriting its previous measurement, then checks vectors, ordered rankings,
mixed-language rankings, and original hashes again. No new dependencies,
downloads, model export, index rebuild, or credentials are required.

The tiny local fixture passed with exact vectors/rankings, unchanged originals,
one resident index, correct request counts, and all six long probes still
encoding to 512 tokens. These tiny-fixture measurements are not real-model
memory results. Instrumentation adds overhead, so this is not a latency
benchmark. Nested phase windows overlap: their peak increments must not be
summed as component sizes. A high-water increase locates a peak in time but
does not identify the responsible ONNX operator or prove container safety.

In the same Colab runtime, upload only `profile_query_memory.py` with this cell.
Keep the earlier three helper uploads and the existing `WORK` and `child_env`.
If the runtime has reset, stop and restore the saved checkpoint first.

```python
from google.colab import files
from pathlib import Path
import sys

uploaded = files.upload()
if len(uploaded) != 1:
    raise ValueError("Upload only profile_query_memory.py.")
run_logged(
    [sys.executable, "-u", next(iter(uploaded)), "--work", str(WORK)],
    env=child_env,
)
report = max(
    Path(WORK).glob("query-memory-*/query-memory-report.json"),
    key=lambda path: path.stat().st_mtime,
)
files.download(str(report))
```

Share `query-memory-report.json`. The known Marathi quantization loss remains a
separate unresolved gate; profiling memory neither fixes nor waives it.

## Long-query peak isolated; allocator settings screen

The returned query-memory report matches profiler SHA-256
`5f092db8661f3a40c8ee3b02d903cf8912c695dc18f4f1210379e861c0309544`.
Session construction and all 192 frozen queries stayed within a 339.4 MiB
process peak. Those queries reached at most 26 encoded tokens. The 512-token
probes raised the observed high-water mark to about 443.1 MiB during ONNX
inference. Tokenization and index access introduced no new peak in this run.
This identifies the phase, not an individual operator or exact allocation.
All frozen vectors and rankings remained identical. Later high-water readings
were slightly lower; subsequent screens retain the maximum observed reading
rather than assuming the final snapshot captures every observed peak.

`scripts/check_inference_memory.py` compares three isolated child processes:

1. Control: existing non-arena CPU allocation and disabled memory-pattern cache.
2. Requested-size arena: registered CPU arena, `arena_extend_strategy=1`
   (`kSameAsRequested`), `max_dead_bytes_per_chunk=256`.
3. The same arena, with unused arena regions eligible for release after each
   inference via the `memory.enable_memory_arena_shrinkage=cpu:0` run option.

The controls follow the official [allocator and shrinkage documentation](https://onnxruntime.ai/docs/get-started/with-c.html),
[Python allocator API](https://onnxruntime.ai/docs/api/python/api_summary.html#allocators),
and [arena configuration definitions](https://github.com/microsoft/onnxruntime/blob/main/include/onnxruntime/core/framework/allocator.h).
The arena's `max_mem=0` uses the runtime default; it is not a process-memory cap.
Sharing an environment allocator here configures a single session per child,
not multiple concurrent sessions. Shrinkage can affect retained memory and
latency; it is not assumed to reduce transient peaks.

All variants retain the same graph optimization, precision, model bytes,
tokenizer, 512-token limit, one-index policy, and sequential query schedule.
Memory-pattern caching remains disabled. Each executes all 216 model requests
and verifies that all six long probes still reach 512 tokens. Output hashes,
shapes, and encoded lengths for every request are compared to the fresh control,
including long inputs and language switching. The 192 frozen embeddings and
ordered rankings also compare against the original uint8 baseline. The original
2% per-language quality, 400 MiB memory, and 500 ms retrieval p95 gates remain.
A memory pass cannot override the existing Marathi failure.

The tiny local end-to-end fixture passed all three settings, with unchanged
outputs, rankings, original hashes, and a single resident index. An initial
zero-byte splitting threshold caused a local allocator failure and is not used;
the verified configuration uses 256 bytes. Fixture memory does not establish
real-model fit. Non-control execution failures are reported, never silently
replaced with other settings. A failed control stops the screen. No application
code or deployment configuration changes are made.

Run this new cell in the same Colab runtime, uploading only
`check_inference_memory.py`. Keep the previous helper uploads and profile
artifacts. No installs, model downloads, index rebuilds, or credentials are needed.
If the runtime has reset, stop and restore the private checkpoint first; the
later memory profile may need rerunning, but the model need not be rebuilt.

```python
from google.colab import files
from pathlib import Path
import sys

uploaded = files.upload()
if len(uploaded) != 1:
    raise ValueError("Upload only check_inference_memory.py.")
run_logged(
    [sys.executable, "-u", next(iter(uploaded)), "--work", str(WORK)],
    env=child_env,
)
report = max(
    Path(WORK).glob("inference-memory-*/inference-memory-report.json"),
    key=lambda path: path.stat().st_mtime,
)
files.download(str(report))
```

Share `inference-memory-report.json`. Original files remain intact; new reports
use a unique `inference-memory-*` folder. Process RSS includes instrumentation
overhead and excludes container-wide accounting and concurrent workloads. OS
file caches are not cleared. There is no automatic upload or promotion.

## Allocator passes memory screen; matrix-quantization quality comparison

The returned allocator report matches helper SHA-256
`cf88c2b249e519c1023c04b16243d9bbcf2d68198607b80a4798017449738389`.
Control peaked at 457.0 MiB, requested-size allocation at 350.9 MiB, and the
shrink-after-run variant at 348.6 MiB. Both allocator variants preserved every
inference output, frozen embedding, and retrieval ranking, including the
512-token probes. Both passed memory and latency gates, but neither passed
Marathi quality. Carry the non-shrinking requested-size allocator forward for
experiments: shrinkage saved only about 2 MiB and was slower for long inference
and switching in this run. This remains a process-level Colab screen, not a
deployment guarantee.

`scripts/check_quantization_quality.py` makes two bounded comparisons against a
fresh original-uint8 control, using the passing allocator in every process:

- `per-channel`: quantize matrix weights per output channel instead of one scale
  for an entire weight tensor. Keep unsigned 8-bit types and full quantization
  range; do not change the embedding tables.
- `fp32-matrices`: retain the original FP32 matrix weights and quantize only
  Gather embedding tables. This is the converse of the earlier matrix-only
  quantization experiment; its larger model may fail memory but helps distinguish
  matrix quantization loss from embedding-table loss.

This follows [ONNX Runtime's quantization/debugging guidance](https://onnxruntime.ai/docs/performance/model-optimizations/quantization.html).
Conversion starts from the existing `e5-fp32.onnx`, not a new model download or
PyTorch export. Graph audits require quantized Gather nodes, weights, scales,
and zero points to match the original uint8 model. Per-channel matrix scale
shapes/counts are checked; FP32 matrix tensor bytes, shapes, and types must match
the saved original graph. Storage-only protobuf flags are not tensor values and
are excluded from that FP32 tensor comparison.

Each conversion and each inference measurement runs in its own child process;
conversion's memory cannot contaminate inference's peak. Candidate manifests
and tokenizer hashes are checked before redirecting only the model file loaded
by the verified adapter. Pooling, prefix, dimension, tokenizer, 512-token ceiling,
disk documents, existing indexes, and one-index residency remain unchanged.
Original files are checksum-verified before and after. New models/reports go
under a unique `quantization-quality-*` folder.

A changed quantizer may change embeddings and rankings: those differences are
reported rather than rejected merely for disagreeing with the weaker uint8
baseline. Every language must still retain at least 98% of its FP32 reference
nDCG@8, with the 400 MiB and 500 ms retrieval p95 gates unchanged. Repeated
language-switch results must match that candidate's own initial results. The
192 reused queries are development data; any promising candidate still needs
independent evaluation before we claim a general quality improvement.

The local tiny-model fixture passed control, both actual conversions, graph
audits, full-length probes, switching consistency, and original-file checks.
This checks the experiment's wiring, not real multilingual model quality or
hosting fit. The live application and deployment configuration are unchanged.

In the current Colab runtime, upload only `check_quantization_quality.py` using
the cell below. Keep `check_inference_memory.py` and the earlier helper uploads.
There are no new installations, credentials, model downloads, or passage-index
rebuilds. It does create two new local model variants, so conversion will take
longer than the preceding settings-only checks.

```python
from google.colab import files
from pathlib import Path
import sys

uploaded = files.upload()
if len(uploaded) != 1:
    raise ValueError("Upload only check_quantization_quality.py.")
run_logged(
    [sys.executable, "-u", next(iter(uploaded)), "--work", str(WORK)],
    env=child_env,
)
report = max(
    Path(WORK).glob("quantization-quality-*/quantization-quality-report.json"),
    key=lambda path: path.stat().st_mtime,
)
files.download(str(report))
```

Share `quantization-quality-report.json`. Candidate errors are recorded rather
than replaced with fallback settings; control errors stop the experiment.
Nothing is automatically uploaded, deployed, or selected for serving.

## Resume after a runtime reset

The private checkpoint archive is still verified locally against SHA-256
`4e014739505d92bf6213fa099d26917e0aff0cca08ef97d74f700e8fa39afad7`.
It contains 107 files (922,318,380 bytes unpacked) through the disk-retrieval
screen, not the later allocator/quantization variants. Later downloaded reports
remain evidence of those experiments; missing candidate model files must not be
treated as restored. The latest matrix-quality run rejected both alternatives:
per-channel had 6.68% Marathi loss at 356.6 MiB; full-precision matrices had
2.98% Marathi loss at 450.1 MiB. The next agreed comparison is explicit ONNX
preprocessing, with FP32 parity checked before quantization, in response to the
preprocessing warning. Do not silently suppress that warning or loosen gates.

`scripts/restore_colab_checkpoint.py` is pinned to this exact archive's size and
checksum. It stages and verifies every file before restoring original absolute
work/source paths, preserving references inside the saved reports. It rejects
unsafe paths, links, duplicates, and conflicting existing files. Identical files
can be reused on a repeat restore. Verified staging files are published without
overwriting existing files, then the restored files are checked again. Temporary
staging is removed; the uploaded archive is retained.

The helper's optional `--install` reinstalls the required application, ONNX,
tokenizer, and index libraries from recorded versions, not the entire Colab
environment. No Torch/model download, Hugging Face token, index rebuild, or
model execution is involved. The checkpoint recorded stale duplicate metadata
for NumPy (1.26.4 and 2.1.3) and packaging (24.0 and 26.3): restoration explicitly
selects 2.1.3 and 26.3 and records that choice. NumPy 2 satisfies the archived
project requirement. This does not prove which duplicate was active previously;
resumed experiments still need a fresh numerical control. Imports and installed
versions are checked in a fresh child interpreter after installation.

Local tiny-archive checks passed checksum round trip, repeat restore, conflict
refusal, unsafe path/symlink rejection, and corrupt-file rejection. Dependency
command wiring was checked without installing the Colab environment on the
developer VM. Actual installation/import validation occurs in Colab and stops
with an error if the recorded versions cannot be installed or imported.

In a fresh free CPU runtime, keep the existing notebook. Do not rerun the original
setup/export cells. Add this cell and select both `restore_colab_checkpoint.py`
and the private `SubaRAG-private-checkpoint.tar.gz` when prompted. The upload is
about 513 MiB; leave the browser connected until restoration completes.

```python
from google.colab import files
from pathlib import Path
import json, os, subprocess, sys

def run_logged(arguments, *, env=None):
    with subprocess.Popen(
        arguments, env=env, stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT, text=True, bufsize=1,
    ) as process:
        for line in process.stdout:
            print(line, end="", flush=True)
        if process.wait() != 0:
            raise RuntimeError("Stop and share the error above; keep the restored files.")

uploaded = files.upload()
scripts = [name for name in uploaded if name.endswith(".py")]
archives = [name for name in uploaded if name.endswith(".gz")]
if len(uploaded) != 2 or len(scripts) != 1 or len(archives) != 1:
    raise ValueError("Upload only the restore helper and private checkpoint archive.")
del uploaded
run_logged([sys.executable, "-u", scripts[0], "--archive", archives[0], "--install"])

report = max(Path("/content").glob("subarag-restore-*/restore-report.json"),
             key=lambda path: path.stat().st_mtime)
state = json.loads(report.read_text())
if not state.get("imports_checked"):
    raise RuntimeError("Dependency checks did not finish; share the output above.")
WORK = Path(state["work"])
backend = Path(state["backend"])
child_env = dict(os.environ, PYTHONPATH=str(backend / "src"),
                 OMP_NUM_THREADS="1", OPENBLAS_NUM_THREADS="1",
                 TOKENIZERS_PARALLELISM="false", HF_HUB_OFFLINE="1",
                 TRANSFORMERS_OFFLINE="1")
child_env.pop("HF_TOKEN", None)
print("Restored WORK:", WORK)
files.download(str(report))
```

Share `restore-report.json` before starting another experiment. Keep the private
archive on the device. Neither restoration nor installation uploads to Hugging
Face or changes the live app. If installation fails after restoration, preserve
the restored files: they do not need to be reuploaded or rebuilt to fix a library
installation problem.

## Resumed control and ONNX preprocessing screen

The restored checkpoint passed all 107 file checks, required-library installation,
and fresh-process import checks, with the same Python version. It restores the
disk-retrieval experiment, not the later allocator/quantization variants. The new
`scripts/check_onnx_preprocessing.py` starts directly from those restored artifacts;
there is no need to recreate the missing experimental models first.

This diagnostic follows ONNX Runtime's
[preprocessing guidance](https://onnxruntime.ai/docs/performance/model-optimizations/quantization.html):
symbolic shape inference, basic graph optimization, then ONNX shape inference on
a separate FP32 copy. It does not insert a metadata marker itself or suppress
warnings. Logged preprocessing errors also fail the stage, because the upstream
helper can catch an optimizer exception and continue with a partially processed
model. There is no automatic retry with stages disabled.

The fresh-process stages are:

1. Restore control: reproduce the saved uint8 vectors and rankings exactly using
   the previously selected requested-size allocator, disk documents, float64
   tokenizer, and one resident index. Recheck runtime gates, including all six
   512-token probes. The known Marathi quality failure does not block this
   control; a numerical, memory, or latency regression does.
2. Measure the saved FP32 graph and its preprocessed copy separately. Compare all
   216 raw outputs (192 development queries plus warmup, long, and switching
   requests): `rtol=1e-4`, `atol=1e-5`, minimum cosine 0.999999, identical ordered
   retrieval results, and stable language switches. Also require original FP32
   rankings to reproduce the saved full-precision reference. Stop before
   quantization if any check fails.
3. Quantize the preprocessed copy using the original per-tensor QUInt8
   MatMul/Gather settings. Recheck memory, latency, and each language's nDCG@8
   against the saved full-precision reference. Audit graph operators and actual
   quantized Gather/matrix presence; a marker alone is not proof of conversion.

Preparation/graph inspection never enters the measured processes. Only the two
FP32 diagnostic runs retain raw arrays; uint8 memory measurements use the previous
unchanged measurement harness. Each run creates a new directory and verifies
original source checksums again at the end. Failures retain a partial report and
their logs; existing reports, models, SQLite stores, and indexes are not overwritten.

The full tiny synthetic pipeline passed locally with ONNX Runtime 1.31.0, including
the 512-token and switching schedule. A deliberately changed FP32 output failed
the numerical gate. These are harness checks, not evidence that E5 quality improves.
The real model still needs the Colab run below. The 192 queries are reused
development data, not an independent holdout, and process RSS is not a serving
container/concurrency guarantee. No runtime defaults, cloud artifacts, or serving
code change in this slice.

Keep the restored Colab session. Add this cell, not a full notebook rerun. Upload
these three local helpers when prompted:

- `scripts/check_onnx_preprocessing.py` (new)
- `scripts/check_quantization_quality.py` (unchanged dependency)
- `scripts/check_inference_memory.py` (unchanged dependency)

The other three tokenizer/disk helpers were restored into `/content` already.
Renamed uploads such as `name (1).py` are accepted by checksum. No token, dataset
upload, model download, or dependency installation is needed.

```python
from google.colab import files
from pathlib import Path
import hashlib, os, sys

if "WORK" not in globals() or not Path(WORK).is_dir():
    raise RuntimeError("The restored runtime is missing. Stop here; do not rebuild.")
if "child_env" not in globals() or "run_logged" not in globals():
    raise RuntimeError("Use the session with the completed checkpoint-restore cell.")
os.chdir("/content")
uploaded = files.upload()
main_hash = "92ad1be0da7e11a32d3848cf447a2458003e310794c5f6846cebc5dcdf0fef22"
expected = {
    main_hash,
    "5c4df33a42d92fcd1aa796506c88c5de5b71b6769047051d45994dde74d34775",
    "cf88c2b249e519c1023c04b16243d9bbcf2d68198607b80a4798017449738389",
}
by_hash = {hashlib.sha256(data).hexdigest(): name for name, data in uploaded.items()}
if len(uploaded) != 3 or set(by_hash) != expected:
    raise ValueError("Select exactly the three helper files listed above, using their latest local copies.")
script = by_hash[main_hash]
del uploaded
before = set(WORK.glob("onnx-preprocessing-*/onnx-preprocessing-report.json"))
try:
    run_logged([sys.executable, "-u", script, "--work", str(WORK)], env=child_env)
finally:
    reports = set(WORK.glob("onnx-preprocessing-*/onnx-preprocessing-report.json")) - before
    if len(reports) == 1:
        files.download(str(reports.pop()))
```

Share `onnx-preprocessing-report.json` and any errors printed above it. Keep the
runtime open. Even a passing screen does not automatically select or deploy the
candidate; independent quality and memory-limited serving checks remain necessary.

### Explicit retry for the symbolic-shape failure

The first real run (`onnx-preprocessing-3i6u1wj8`) reproduced the original uint8
vectors/rankings exactly, peaked at 361.7 MiB, and left original files unchanged.
Marathi still showed a 4.4% quality loss, exceeding the 2% allowance. The FP32 reference completed, but
strict preprocessing stopped with `unsupported broadcast between Min(512,
sequence) sequence` / `Incomplete symbolic shape inference`. No preprocessed
FP32 comparison or new uint8 quality result exists from that run.

The source of the failure is symbolic dimension unification, not a SIGKILL or
missing dependency. With sequence length at most 512, `Min(512, sequence)` equals
`sequence`. ONNX Runtime's
[symbolic shape-inference implementation](https://github.com/microsoft/onnxruntime/blob/main/onnxruntime/python/tools/symbolic_shape_infer.py)
provides `auto_merge` for unresolved broadcast dimensions. This is a diagnostic
assumption, not a generally safe rewrite for unrestricted inputs: other symbolic
merges may also occur. Keep all FP32 numerical/ranking gates and the existing
512-token truncation contract. The revised helper additionally requires unchanged
input types and dynamic dimension declarations after preprocessing.

Use the new explicit `--auto-merge` option for this retry. The default remains
strict; there is no silent fallback and no preprocessing stage is skipped. The
option is recorded in the report and forwarded to child processes. A tiny
Slice/Add graph reproduced the same strict failure locally; the merge variant
passed the complete synthetic harness and exact-output checks at lengths 1–512.
That verifies the retry machinery, not the real E5 graph or its retrieval quality.

For this known failure, use the cell below rather than rerunning the strict-mode
cell above. Upload **only the updated `check_onnx_preprocessing.py`**; the other
helpers remain in `/content`. Duplicate upload names are accepted by checksum.
It makes a fresh experiment folder and repeats the control/reference checks; it
does not export/download a model or rebuild indexes. Keep the earlier output.

```python
from google.colab import files
from pathlib import Path
import hashlib, os, sys

if "WORK" not in globals() or not Path(WORK).is_dir():
    raise RuntimeError("Restored runtime missing. Stop here; do not rebuild.")
if "child_env" not in globals() or "run_logged" not in globals():
    raise RuntimeError("Use the session with the completed restore cell.")
os.chdir("/content")
uploaded = files.upload()
expected = "92ad1be0da7e11a32d3848cf447a2458003e310794c5f6846cebc5dcdf0fef22"
if len(uploaded) != 1:
    raise ValueError("Upload only the updated check_onnx_preprocessing.py.")
script = next(iter(uploaded))
if hashlib.sha256(uploaded[script]).hexdigest() != expected:
    raise ValueError("This is not the updated helper. Select its latest local copy.")
del uploaded
pattern = "onnx-preprocessing-*/onnx-preprocessing-report.json"
before = set(WORK.glob(pattern))
try:
    run_logged([sys.executable, "-u", script, "--work", str(WORK), "--auto-merge"],
               env=child_env)
finally:
    reports = set(WORK.glob(pattern)) - before
    if len(reports) == 1:
        files.download(str(reports.pop()))
```

Share the new report and any errors. No push, upload, deployment, or serving
change is part of this correction.
