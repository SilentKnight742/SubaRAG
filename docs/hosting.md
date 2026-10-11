# Free application deployment

Use **Render Free** for the Docker application, **free Colab CPU** for data
preparation, and a private Hugging Face dataset repository for prepared artifacts.
No paid Space, GPU, database, disk, or background worker is needed for this slice.

## Account handoff

1. Sign in to [Render](https://dashboard.render.com/) with GitHub. Do not add a
   payment method or select a paid plan. Stop if a payment is requested.
2. Publish `feature/compact-rag-serving`. For the existing `subarag` service,
   change its source branch to that branch. A new service can instead use the
   branch's `render.yaml` Blueprint.
3. Supply `ELEVENLABS_API_KEY`, `GEMINI_API_KEY`, and the dataset-scoped
   read-only `HF_TOKEN` in Render's secret prompts, never in source code.
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

The default Docker target is `release`, which inherits the compact runtime. The
lightweight interface remains available explicitly with
`docker build --target preview -t subarag:preview .`; the full-size experimental
runtime remains available as the separate `rag` target.

## Free-tier limits

[Render Free](https://render.com/docs/free) sleeps after 15 minutes without
traffic and can take about a minute to wake. Benchmark pipeline latency only
after wake-up, reporting cold starts separately. Free services have temporary
storage and limited memory: do not build embeddings or keep the full corpus in
the web process. A cold instance re-downloads and verifies the roughly 305 MiB
compact bundle before becoming healthy. Exceeding free allowances without a
payment method can suspend the service. Do not upgrade it to bypass limits.

ElevenLabs remains the selected provider while free credits last. No automatic
paid fallback or top-up is configured by this project. Typed questions remain
available without speech credits. Sarvam is retained as an explicit future switch,
not described as an unlimited free service.

## Verification

Build locally with `docker build -t subarag:release .`, then run:

```bash
docker run --rm --env-file backend/.env -p 127.0.0.1:7860:7860 subarag:release
```

Check the deployed HTTPS preview:

- A signed-out browser receives a password prompt; health checks still work.
- One recording triggers transcription, then the query API. Retrying a failed
  query does not transcribe again.
- Typed questions use the same query API without a speech request.
- Supported questions return a grounded answer with server-owned citations;
  unsupported or insufficiently supported questions return `no_answer`.
- `/api/status` reports ElevenLabs, corpus availability and RAG readiness.

## Data handoff

Run the [Colab notebook and data guide](data.md), then share its downloaded
`manifest.json`. Keep artifacts remote; no corpus download to this VM is needed.
Before Colab's temporary session ends, upload the prepared manifest and shards
to the chosen private dataset repository, excluding `_work/`.

The verified pilot is in private repository `silentkirmada/SubaRAG-data` at commit
`e29f5c990a4fc0d8e05771be4007f1c6560adfc2`. Its manifest reports 4,000 accepted
source rows, zero rejected rows, 59,594 passages and 5,997 evaluation queries.

The Blueprint sets `SUBARAG_DATA_REPO` and `SUBARAG_DATA_REVISION` to that exact
artifact. At startup, the backend downloads only its manifest and listed shards,
streaming each file to disk. It checks every byte count, SHA-256 and row count
before reporting data ready. Partial downloads are not published as a cache.
The cache is keyed by repository and commit, and revalidated on reuse. A missing
or invalid token, incomplete configuration or corrupt artifact prevents startup;
the app never silently serves an unverified corpus. The preview limits the
prepared artifact to 256 MiB and downloads to three minutes. A larger corpus
requires a separately sized serving plan, not simply raising these limits.

Render's local storage is temporary; a fresh instance re-downloads the pinned
artifact from Hugging Face. The prepared files stay on disk, not loaded wholesale
into memory, and nothing is embedded or indexed during startup. `/api/status`
reports `data.status: ready`; its existing `data.revision` is the upstream
MSMARCO-XI revision, not the prepared-artifact commit above.

For local development, uncomment the two remote dataset settings in `.env.example`
when copying them into `.env`; `HF_TOKEN` alone does not trigger a download.
Alternatively, use `SUBARAG_DATA_MANIFEST` for a local prepared artifact. Do not
configure both sources. Without either, data remains `not_configured`.

The authenticated Hub metadata check confirmed the private repository and commit.
The compact release is deployed at `https://subarag.onrender.com`. Its public
health check returns 200, unauthenticated status access returns 401, and the
authenticated status reports transcription, RAG, and data as ready. This proves
that the verified runtime can complete cold startup on the free instance. The
user also verified grounded typed and voice answers, rendered citations, and a
Marathi language switch on the hosted application. If a later change exceeds the
512 MiB limit, restore this deploy rather than selecting a paid plan.

The compact artifact is pinned separately to private repository
`silentkirmada/SubaRAG-data`, revision
`3f0ef0a35e7fd2cf733c6ff6e1ac9a9e49a539d6`, path
`serving/compact/27cb87a265d9058a4c512d138519bfd9060c156bf1abef1a24a07919bcc36115`.
Startup verifies its content-addressed manifest and all declared files before
the application becomes ready.
