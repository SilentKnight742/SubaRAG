# SubaRAG

Voice questions and fast, grounded answers across English, Hindi, and Marathi.

## Current stage

Step one connects microphone capture, ElevenLabs transcription, typed questions,
the backend query API, and the response panel. The RAG engine is intentionally
unavailable: requests return an explicit status with **no generated answer or
invented citations**. Step two implements retrieval and generation behind the
same interface.

Data preparation runs remotely using the [Colab notebook](notebooks/prepare_data.ipynb)
and [data guide](docs/data.md). It produces a versioned passage corpus and
separate evaluation queries. It does not create embeddings or a retrieval index.
No full dataset processing is needed on a developer's computer.

## Local development

Run these in separate terminals from the repository root:

```bash
cd backend
python3 -m venv .venv
.venv/bin/pip install -e '.[dev]'
cp .env.example .env
# Add ELEVENLABS_API_KEY to .env using your editor.
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

## API boundaries

| Endpoint | Purpose |
| --- | --- |
| `GET /healthz` | Running backend and configured STT adapters |
| `GET /api/status` | Selected STT provider, corpus availability, RAG availability |
| `POST /api/transcriptions` | Validated browser audio → normalized transcript |
| `POST /api/queries` | `{question, language}` → answer/status/citations |

`subarag.rag.RagEngine` is the sole implementation seam for step two. An
`answered` result requires both text and citations; `unavailable` and `no_answer`
results cannot carry either. Query retries do not repeat transcription.

## Checks

```bash
backend/.venv/bin/python -m pytest backend/tests
cd frontend
npm test
npm run build
```

See [hosting instructions](docs/hosting.md) for the password-protected Render Free preview.
Publishing and deployment are separate, explicitly authorized actions.
