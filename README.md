# SubaRAG

Voice questions and fast, grounded answers across English, Hindi, and Marathi.

## Current stage

The application connects microphone capture, provider-neutral transcription,
typed questions, the backend query API, and the response panel. The measured
retrieval layer now includes multilingual E5 embeddings, exact and FAISS HNSW
indexes, a versioned retrieval cache, extractive context compression, reproducible
benchmarks, and a server-owned citation boundary.

Gemini 3.5 Flash-Lite passed the corrected isolated generator gate and is now
wired behind that citation boundary. The application becomes RAG-ready only
when the Gemini key and all three checksum-verified language indexes are
configured. With incomplete configuration it produces **no generated answer or
invented citations**. See the
[experiment results](docs/experiments.md) and
[measured architecture](docs/architecture.md) for the measured decisions and
remaining end-to-end release gate.

An opt-in [compact portfolio runtime](docs/portfolio-runtime.md) now connects the
tested ONNX encoder, disk-backed passages, and one language index at a time to
the same answer API. Its accepted private artifact can be fetched only from a
pinned commit and content-addressed path. Fresh-query retrieval validation has
passed, and real-answer validation recorded grounded English and Marathi answers
plus conservative Hindi `no_answer` behavior. The compact runtime is deployed on
[Render Free](https://subarag.onrender.com), where typed, voice, citation, and
language-switching checks pass.

Data preparation runs remotely using the [Colab notebook](notebooks/prepare_data.ipynb)
and [data guide](docs/data.md). It produces a versioned passage corpus and
separate evaluation queries. It does not create embeddings or a retrieval index.
No full dataset processing is needed on a developer's computer.

## Local development

Run these in separate terminals from the repository root:

```bash
cd backend
python3 -m venv .venv
.venv/bin/pip install -e '.[dev,rag]'
cp .env.example .env
# Add ELEVENLABS_API_KEY and GEMINI_API_KEY using your editor.
.venv/bin/uvicorn subarag.api:app --host 127.0.0.1 --port 8000
```

```bash
cd frontend
npm ci
npm run dev -- --host 127.0.0.1
```

The browser uses the frontend URL. Vite forwards `/api` to the backend.
Recordings submit automatically after speech followed by 2.5 seconds of silence.
Capture stops at 29.5 seconds to allow encoding overhead beneath the 30-second
server limit. Audio is not retained as application history; temporary upload
files are closed after processing.
STT provider retention is governed by that provider's settings and terms.

ElevenLabs is the default. To switch later, set `SUBARAG_TRANSCRIPTION_PROVIDER`
to `sarvam`, configure `SARVAM_API_KEY`, and restart the backend. The browser uses
the server default; it never contains provider credentials. There is no automatic
fallback that silently sends recordings to a second provider.

Grounded answers require `SUBARAG_RETRIEVAL_MANIFEST_EN`,
`SUBARAG_RETRIEVAL_MANIFEST_HI`, and `SUBARAG_RETRIEVAL_MANIFEST_MR` to point at
prebuilt serving manifests. Configure all three together. The backend verifies
their checksums and embedding model before reporting RAG as ready; it never
embeds the corpus during startup.
Alternatively, the compact runtime accepts a local `SUBARAG_COMPACT_MANIFEST` or
the pinned `SUBARAG_COMPACT_REPO`, `SUBARAG_COMPACT_REVISION`, and
`SUBARAG_COMPACT_PATH` settings described in the hosting guide. Do not configure
the full and compact runtimes together.
Build the pinned pilot's English/Hindi/Marathi indexes remotely with the
[Colab serving-index guide](docs/rag-indexes.md) and
[notebook](notebooks/build_rag_indexes.ipynb). Index preparation does not enable
RAG on the live preview; serving memory and quality checks remain separate.

## API boundaries

| Endpoint | Purpose |
| --- | --- |
| `GET /healthz` | Running backend and configured STT adapters |
| `GET /api/status` | Selected STT provider, corpus availability, RAG availability |
| `POST /api/transcriptions` | Validated browser audio → normalized transcript |
| `POST /api/queries` | `{question, language}` → answer/status/citations |

`subarag.rag.RagEngine` is the application seam for retrieval and generation. An
`answered` result requires both text and citations; `unavailable` and `no_answer`
results cannot carry either. `GroundedRagEngine` additionally rejects generated
citation IDs that were not present in retrieved evidence. Query retries do not
repeat transcription.

## Checks

```bash
backend/.venv/bin/python -m pytest backend/tests
cd frontend
npm test
npm run build
```

See [hosting instructions](docs/hosting.md) for the password-protected Render Free
deployment. The default release image downloads the pinned prepared corpus and
compact serving bundle using a read-only `HF_TOKEN`, verifies both, and loads one
language index at a time. The lightweight `preview` and full-size experimental
`rag` Docker targets remain explicit alternatives.
