# Compact portfolio runtime

## Decision and scope

Use the preprocessed uint8 encoder provisionally and stop quantization searches.
The old 2% per-language quality threshold is not a release requirement. Its
Marathi regression remains documented; accepting a candidate does not establish
answer quality or prove that it fits the hosting limit.

The opt-in runtime reuses the checked float64 tokenizer, SQLite documents, and
one resident language index. A single background retrieval worker keeps health
checks responsive. The API admits one question at a time and also rejects new
questions while cancelled/timed-out native retrieval is still finishing. It
returns the existing grounded-answer contract with server-resolved citations.

No default hosting configuration changes. No push or deployment is performed by
this slice.

The accepted bundle is stored privately and immutably at:

- repository: `silentkirmada/SubaRAG-data`
- revision: `3f0ef0a35e7fd2cf733c6ff6e1ac9a9e49a539d6`
- path: `serving/compact/27cb87a265d9058a4c512d138519bfd9060c156bf1abef1a24a07919bcc36115`
- downloaded delivery SHA-256: `1ea5cfd101c10013ec6bbf6f56942ac2252fad6bf56694b3bd5809c48d853d02`

Remote startup downloads into a temporary cache directory, verifies the
content-addressed manifest and every declared file, and only then atomically
publishes the cache. It never loads a partial download.

Local verification: all 133 existing backend tests passed. A tiny synthetic
bundle exercised checksum verification, concurrent language changes, the real
query endpoint with mocked generation, citation resolution, busy/timeout
handling, health checks, cleanup, and the fresh-query measurement helper. This
does not substitute for the real-artifact Colab run below.

## Next Colab cell

Keep the existing notebook and cells. Use the active session containing `WORK`
and its completed preprocessing artifacts. If those artifacts are gone, stop:
the earlier checkpoint does not contain all later experiment results.

Download the current `SubaRAG-serving-source.zip` supplied with this change. Add
the following cell at the bottom of the existing notebook. Upload **only that
ZIP** when prompted. Renamed duplicate uploads are accepted; old uploads need
not be deleted. The archive supplies the new application code as well as the
helper, so uploading an individual helper is not sufficient.

```python
import io, os, subprocess, sys, tempfile, zipfile
from pathlib import Path, PurePosixPath
from google.colab import files

if "WORK" not in globals() or not Path(WORK).is_dir():
    raise RuntimeError("WORK is missing. Stop and ask for the restore step.")
uploaded = files.upload()
if len(uploaded) != 1 or not next(iter(uploaded)).lower().endswith(".zip"):
    raise ValueError("Upload only SubaRAG-serving-source.zip.")
source = Path(tempfile.mkdtemp(prefix="subarag-serving-source-", dir="/content"))
with zipfile.ZipFile(io.BytesIO(next(iter(uploaded.values())))) as archive:
    for item in archive.infolist():
        path = PurePosixPath(item.filename)
        if path.is_absolute() or ".." in path.parts or "\\" in item.filename:
            raise ValueError("Unsafe archive path")
        if not path.parts or path.parts[0] != "SubaRAG":
            raise ValueError("Unexpected archive layout")
        if (item.external_attr >> 16) & 0o170000 == 0o120000:
            raise ValueError("Archive links are not allowed")
    archive.extractall(source)
env = dict(os.environ, PYTHONPATH=str(source / "SubaRAG/backend/src"),
           HF_HUB_OFFLINE="1", TRANSFORMERS_OFFLINE="1",
           OMP_NUM_THREADS="1", OPENBLAS_NUM_THREADS="1",
           TOKENIZERS_PARALLELISM="false")
for key in tuple(env):
    if key.startswith("SUBARAG_") or key in ("HF_TOKEN", "GEMINI_API_KEY", "ELEVENLABS_API_KEY", "SARVAM_API_KEY"):
        env.pop(key)
before = set(Path(WORK).glob("compact-serving-*/compact-serving-report.json"))
process = subprocess.Popen(
    [sys.executable, "-u", str(source / "SubaRAG/scripts/check_compact_serving.py"),
     "--work", str(WORK)], env=env, stdout=subprocess.PIPE,
    stderr=subprocess.STDOUT, text=True)
for line in process.stdout:
    print(line, end="", flush=True)
if process.wait() != 0:
    raise RuntimeError("Stop and share the progress/error above. Keep the output folder.")
reports = set(Path(WORK).glob("compact-serving-*/compact-serving-report.json")) - before
if len(reports) != 1:
    raise RuntimeError("Expected one new report; share the output above.")
files.download(str(reports.pop()))
```

No keys, network API calls, model exports, uploads, or deployment are involved.
Use the restored Colab dependencies; do not install the heavyweight application
extras merely to run this check.

The check packages the existing artifacts, verifies their hashes, and runs the
actual serving retriever in separate compact and FP32 processes. It requires
exact reproduction of the old compact rankings, then evaluates 64 fresh
questions per language, excluding old IDs and duplicate normalized question
text. It includes language switches and six 512-token probes.

Return `compact-serving-report.json`. Keep the Colab output folder: its
`bundle/compact-bundle.json` and sibling files are the portable serving bundle.
The report contains fresh retrieval metrics, timing, and process memory, not
generated-answer quality. The questions come from the same prepared corpus;
this is not an independently curated benchmark.

## Opt-in application configuration

For an artifact-capable validation machine, install `backend[compact]` and either
set `SUBARAG_COMPACT_MANIFEST` to a local `bundle/compact-bundle.json`, or set
`SUBARAG_COMPACT_REPO`, `SUBARAG_COMPACT_REVISION`, and `SUBARAG_COMPACT_PATH` to
the immutable values above. Remote loading additionally requires a read-only
`HF_TOKEN`; both modes require `GEMINI_API_KEY`. Remove full-size retrieval
configuration. The usual transcription credentials and preview protection still
apply. Do not copy model artifacts to a machine that cannot accommodate them.

The optional Docker target is `compact`; the default remains `preview`. Use one
application worker. A local bundle may be mounted read-only; the pinned remote
mode downloads it once into the verified cache. The image never rebuilds the
embedding model.

## Remaining before release

1. Review the fresh-query report for practical regressions and runtime parity.
2. Exercise real generated answers and citations in all three languages,
   including unsupported questions, using the free provider allowance.
3. Check the complete service under the actual hosting memory limit, including
   startup, transcription, answer generation, long queries, and language changes.
   Colab process RSS below 400 MiB is only a screen, not proof of a 512 MiB fit.
4. After the checks pass, seek explicit push/deployment approval.

No more model tuning unless one of these checks exposes a concrete blocker.

### Real-answer check

`scripts/check_compact_answers.py` uses the verified serving retriever to select
two fresh cases per language whose top eight contains labeled evidence, then sends them through the
real `/api/queries` application boundary. A seventh fictional-star question
checks strict no-answer behavior. The helper records answers and server-owned
citation text for manual review; six same-corpus questions are not a broad
answer-quality benchmark. It reads `GEMINI_API_KEY` from the child environment,
removes it immediately, and never writes it to the report.
