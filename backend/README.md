# SubaRAG backend

This package exposes speech transcription and the application query API. ElevenLabs is the selected default; Sarvam remains available through a server configuration change. The measured RAG runtime uses multilingual E5, prebuilt language-specific HNSW indexes, and Gemini 3.5 Flash-Lite behind the `RagEngine` interface.

The first implementation uses completed WebM recordings because both providers accept that browser-native format:

- [Sarvam Speech-to-Text](https://docs.sarvam.ai/api-reference/speech-to-text/transcribe) accepts synchronous recordings up to 30 seconds.
- [ElevenLabs Speech-to-Text](https://elevenlabs.io/docs/api-reference/speech-to-text/convert) accepts recordings well beyond the interface limit.

`TranscriptionService` requires measured duration metadata, applies SubaRAG's own 30-second limit before provider selection, validates provider capabilities before upload, routes to an explicitly selected provider or the configured default, and returns one normalized result with timing data for later A/B evaluation. The HTTP upload boundary measures media duration before constructing a service request; it never trusts a browser-supplied duration. Provider credentials remain server-side.

The concrete async HTTP adapters use:

- Sarvam `saaras:v3` in `transcribe` mode with timestamps disabled.
- ElevenLabs `scribe_v2` with audio-event tagging, diarization, and timestamps disabled for the lowest-overhead voice-search response.

Copy `.env.example` to `.env` when local authenticated testing begins. Never commit the populated file.

The local API exposes:

- `POST /api/transcriptions` with an `audio` file plus optional `language` and `provider` multipart fields.
- `GET /healthz` for readiness and configured-provider names; it never returns credentials.
- `POST /api/queries` for validated typed/transcribed questions. It returns `unavailable` with no answer or citations until every RAG dependency is configured.
- `GET /api/status` for selected transcription provider, prepared corpus availability, and RAG availability.

Uploads are capped at 8 MiB, restricted to browser WebM/Opus or MP4/AAC audio, decoded server-side, and rejected before provider work when their measured duration exceeds 30 seconds. Browser provider overrides are disabled by default. Set `SUBARAG_ALLOW_PROVIDER_SELECTION=true` only for deliberate adapter testing. At most two provider requests run concurrently by default.

`SUBARAG_FRONTEND_DIST` enables serving the compiled frontend from this process; the Docker image sets it automatically. `SUBARAG_DATA_MANIFEST` optionally points to a prepared-corpus manifest and its local shards. The three `SUBARAG_RETRIEVAL_MANIFEST_*` variables enable grounded answers only when they are configured together with `GEMINI_API_KEY`. Invalid, incomplete, or checksum-failing index configuration prevents startup. No corpus or index is downloaded or rebuilt implicitly. See the root [README](../README.md), [data guide](../docs/data.md), and [hosting guide](../docs/hosting.md).

Realtime partial transcripts are intentionally deferred. Both vendors' realtime APIs require PCM-style streams rather than the current MediaRecorder WebM output, so that path needs a separate browser audio-worklet design.

## Run tests

```bash
cd backend
python3 -m venv .venv
.venv/bin/pip install -e '.[dev,rag]'
.venv/bin/python -m pytest
```

## Run locally

```bash
cd backend
.venv/bin/uvicorn subarag.api:app --reload
```
