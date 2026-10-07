FROM node:24-bookworm-slim AS frontend
WORKDIR /build
COPY frontend/package.json frontend/package-lock.json ./
RUN npm ci
COPY frontend/ ./
RUN npm run build

FROM python:3.12-slim-bookworm AS runtime
ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    HF_HOME=/app/.cache/huggingface \
    SUBARAG_DATA_CACHE_DIR=/app/.cache/corpus \
    SUBARAG_RETRIEVAL_CACHE_DIR=/app/.cache/retrieval \
    SUBARAG_FRONTEND_DIST=/app/frontend/dist \
    SUBARAG_TRANSCRIPTION_PROVIDER=elevenlabs \
    SUBARAG_ALLOW_PROVIDER_SELECTION=false
RUN useradd --create-home --uid 1000 app
WORKDIR /app
COPY backend/pyproject.toml /app/backend/pyproject.toml
COPY backend/src/ /app/backend/src/
RUN pip install --no-cache-dir /app/backend \
    && mkdir -p /app/.cache \
    && chown -R app:app /app/.cache
COPY --from=frontend --chown=1000:1000 /build/dist /app/frontend/dist
USER 1000
EXPOSE 7860
HEALTHCHECK --interval=30s --timeout=5s --start-period=30s --retries=3 \
    CMD python -c "import urllib.request; urllib.request.urlopen('http://127.0.0.1:7860/healthz', timeout=3)"
CMD ["uvicorn", "subarag.api:app", "--host", "0.0.0.0", "--port", "7860"]

# Optional second-stage image; the free first-stage preview does not load models.
FROM runtime AS rag
USER root
RUN pip install --no-cache-dir '/app/backend[rag]'
USER 1000
RUN python -c "from sentence_transformers import SentenceTransformer; SentenceTransformer('intfloat/multilingual-e5-small')"
ENV HF_HUB_OFFLINE=1

# Default build target, including on Render.
FROM runtime AS preview
