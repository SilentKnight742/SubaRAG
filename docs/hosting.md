# Free application preview

Use **Render Free** for the Docker application, **free Colab CPU** for data
preparation, and a private Hugging Face dataset repository for prepared artifacts.
No paid Space, GPU, database, disk, or background worker is needed for this slice.

## Account handoff

1. Sign in to [Render](https://dashboard.render.com/) with GitHub. Do not add a
   payment method or select a paid plan. Stop if a payment is requested.
2. After explicit permission to push and deploy, publish the
   `feature/rag-experiments` branch. In Render, create a Blueprint from
   `SilentKnight742/SubaRAG`, selecting this branch and its `render.yaml`.
3. Supply `ELEVENLABS_API_KEY` in Render's secret prompt, never in source code.
   Keep the ElevenLabs account on its free allowance; do not enable paid usage.
4. Creating the Blueprint starts the first deployment. Subsequent automatic
   deploys are disabled: manually deploy only after explicit approval.
5. Open the service's HTTPS URL. The browser asks for username `subarag` and the
   generated `SUBARAG_PREVIEW_PASSWORD` visible in Render's Environment settings.
   Share the URL, not keys or this password in public.

The [Blueprint](https://render.com/docs/blueprint-spec) selects Docker, free
compute, Singapore, and port 7860. Runtime secrets never enter the image. The
Docker context excludes `.env`, local dependencies, Git history, and artifacts.
The password gate covers the interface and API; only `/healthz` is public.
For a manually created service, set a strong `SUBARAG_PREVIEW_PASSWORD` before
exposing it. Local development leaves this optional and binds to localhost.

## Free-tier limits

[Render Free](https://render.com/docs/free) sleeps after 15 minutes without
traffic and can take about a minute to wake. Benchmark pipeline latency only
after wake-up, reporting cold starts separately. Free services have temporary
storage and limited memory: do not build embeddings or keep the full corpus in
the web process. Exceeding free allowances without a payment method can suspend
the preview. Do not upgrade it to bypass limits.

ElevenLabs remains the selected provider while free credits last. No automatic
paid fallback or top-up is configured by this project. Typed questions remain
available without speech credits. Sarvam is retained as an explicit future switch,
not described as an unlimited free service.

## Verification

Build locally with `docker build -t subarag:step-one .`, then run:

```bash
docker run --rm --env-file backend/.env -p 127.0.0.1:7860:7860 subarag:step-one
```

Check the deployed HTTPS preview:

- A signed-out browser receives a password prompt; health checks still work.
- One recording triggers transcription, then the query API. Retrying a failed
  query does not transcribe again.
- Typed questions use the same query API without a speech request.
- The response says answers are coming soon; it fabricates no answer or citation.
- `/api/status` reports ElevenLabs, corpus availability and RAG availability.

## Data handoff

Run the [Colab notebook and data guide](data.md), then share its downloaded
`manifest.json`. Keep artifacts remote; no corpus download to this VM is needed.
Before Colab's temporary session ends, upload the prepared manifest and shards
to the chosen private dataset repository, excluding `_work/`.

The backend supports `SUBARAG_DATA_MANIFEST` pointing to a local prepared artifact
and validates metadata and shard sizes at startup. No artifact repository or
access token has been supplied yet, so remote artifact download is not configured.
That connection is the next handoff after the pilot and repository are available;
Render cannot mount a Hugging Face dataset directly. Without a configured artifact,
the API honestly reports `not_configured`.

Step one is not remotely complete until the corpus is prepared, its download is
connected, the three retrieval manifests and Gemini key are configured, and the
hosted preview is checked. Until then, RAG remains intentionally unavailable.
