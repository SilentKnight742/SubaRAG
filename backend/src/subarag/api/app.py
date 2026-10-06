"""FastAPI application exposing the provider-neutral transcription service."""

import asyncio
from collections.abc import AsyncIterator, Callable
from concurrent.futures import ThreadPoolExecutor
from contextlib import AsyncExitStack, asynccontextmanager
from functools import partial
from pathlib import Path
from time import perf_counter
from typing import Protocol, cast
from uuid import uuid4

import httpx
from fastapi import FastAPI, Request
from fastapi.exceptions import RequestValidationError
from fastapi.responses import FileResponse, JSONResponse
from fastapi.staticfiles import StaticFiles
from python_multipart.exceptions import MultipartParseError
from starlette.datastructures import FormData, UploadFile as StarletteUploadFile
from starlette.exceptions import HTTPException as StarletteHTTPException
from starlette.types import ASGIApp, Message, Receive, Scope, Send

from ..transcription import (
    AudioPayload,
    ElevenLabsTranscriptionProvider,
    SarvamTranscriptionProvider,
    TranscriptionProviderError,
    TranscriptionRequest,
    TranscriptionResult,
    TranscriptionService,
    UnknownTranscriptionProviderError,
    create_transcription_http_client,
)
from ..data.manifest import load_manifest
from ..rag import (
    RagEngine,
    RagQuery,
    UnavailableRagEngine,
    build_grounded_rag_engine,
)
from ..rag.providers import create_generation_http_client
from .media import (
    ALLOWED_MEDIA_TYPES,
    InspectedAudio,
    MediaValidationError,
    inspect_audio,
    normalize_media_type,
)
from .schemas import (
    ApplicationStatus,
    DataStatus,
    ErrorResponse,
    HealthResponse,
    QueryResponse,
    RagStatus,
    TranscriptionResponse,
    TranscriptionStatus,
)
from .settings import Settings
from .access import PreviewAccessMiddleware


UPLOAD_CHUNK_BYTES = 64 * 1024
MULTIPART_OVERHEAD_BYTES = 64 * 1024
MAX_MULTIPART_FIELD_BYTES = 64 * 1024
MEDIA_INSPECTION_WORKERS = 2
ALLOWED_MULTIPART_FIELDS = frozenset({"audio", "language", "provider"})


class TranscriptionServiceLike(Protocol):
    @property
    def available_providers(self) -> tuple[str, ...]: ...

    async def transcribe(
        self,
        request: TranscriptionRequest,
        *,
        provider: str | None = None,
    ) -> TranscriptionResult: ...


MediaInspector = Callable[..., InspectedAudio]
ClientFactory = Callable[[], httpx.AsyncClient]
RagEngineBuilder = Callable[..., RagEngine]


class ApiRequestError(ValueError):
    def __init__(self, status_code: int, code: str, message: str) -> None:
        super().__init__(message)
        self.status_code = status_code
        self.code = code


class _RequestBodyTooLarge(Exception):
    pass


def _error_response(
    *,
    status_code: int,
    code: str,
    message: str,
    retryable: bool = False,
) -> JSONResponse:
    return JSONResponse(
        status_code=status_code,
        content={
            "error": {
                "code": code,
                "message": message,
                "retryable": retryable,
            }
        },
    )


class RequestBodyLimitMiddleware:
    """Reject oversized bodies before multipart parsing or provider work."""

    def __init__(self, app: ASGIApp, *, max_bytes: int) -> None:
        self.app = app
        self.max_bytes = max_bytes

    async def __call__(self, scope: Scope, receive: Receive, send: Send) -> None:
        if scope["type"] != "http":
            await self.app(scope, receive, send)
            return

        headers = dict(scope.get("headers", []))
        content_length = headers.get(b"content-length")
        if content_length is not None:
            try:
                if int(content_length) > self.max_bytes:
                    await _error_response(
                        status_code=413,
                        code="request_too_large",
                        message="The upload request is too large.",
                    )(scope, receive, send)
                    return
            except ValueError:
                await _error_response(
                    status_code=400,
                    code="invalid_content_length",
                    message="The Content-Length header is invalid.",
                )(scope, receive, send)
                return

        received_bytes = 0
        limit_exceeded = False
        response_started = False

        async def limited_receive() -> Message:
            nonlocal limit_exceeded, received_bytes
            if limit_exceeded:
                return {"type": "http.disconnect"}

            message = await receive()
            if message["type"] == "http.request":
                received_bytes += len(message.get("body", b""))
                if received_bytes > self.max_bytes:
                    limit_exceeded = True
                    raise _RequestBodyTooLarge
            return message

        async def tracked_send(message: Message) -> None:
            nonlocal response_started
            if limit_exceeded:
                return
            if message["type"] == "http.response.start":
                response_started = True
            await send(message)

        try:
            await self.app(scope, limited_receive, tracked_send)
        except _RequestBodyTooLarge:
            limit_exceeded = True

        if limit_exceeded:
            if response_started:
                raise RuntimeError("Request body limit was exceeded after response start.")
            await _error_response(
                status_code=413,
                code="request_too_large",
                message="The upload request is too large.",
            )(scope, receive, send)


def _build_service(
    settings: Settings,
    client: httpx.AsyncClient,
) -> TranscriptionService:
    providers = []
    if settings.sarvam_api_key is not None:
        providers.append(
            SarvamTranscriptionProvider(
                api_key=settings.sarvam_api_key.get_secret_value(),
                client=client,
            )
        )
    if settings.elevenlabs_api_key is not None:
        providers.append(
            ElevenLabsTranscriptionProvider(
                api_key=settings.elevenlabs_api_key.get_secret_value(),
                client=client,
            )
        )

    if not providers:
        raise RuntimeError("At least one transcription provider key must be configured.")

    return TranscriptionService(
        providers,
        default_provider=settings.transcription_provider,
        max_audio_seconds=settings.max_audio_seconds,
    )


def _configured_retrieval_manifests(settings: Settings) -> dict[str, Path] | None:
    manifests = {
        "en": settings.retrieval_manifest_en,
        "hi": settings.retrieval_manifest_hi,
        "mr": settings.retrieval_manifest_mr,
    }
    configured = {language: path for language, path in manifests.items() if path is not None}
    if not configured:
        return None
    if len(configured) != len(manifests):
        raise RuntimeError("English, Hindi, and Marathi retrieval manifests must be configured together.")
    if settings.gemini_api_key is None:
        raise RuntimeError("GEMINI_API_KEY is required when retrieval manifests are configured.")
    return cast(dict[str, Path], manifests)


async def _read_upload(upload: StarletteUploadFile, *, max_bytes: int) -> bytes:
    content = bytearray()
    while chunk := await upload.read(UPLOAD_CHUNK_BYTES):
        content.extend(chunk)
        if len(content) > max_bytes:
            raise ApiRequestError(
                413,
                "audio_too_large",
                "The audio recording is too large.",
            )

    if not content:
        raise ApiRequestError(422, "empty_audio", "The audio recording is empty.")
    return bytes(content)


def _validate_multipart_form(form: FormData) -> None:
    field_counts = {name: 0 for name in ALLOWED_MULTIPART_FIELDS}
    for name, value in form.multi_items():
        if name not in ALLOWED_MULTIPART_FIELDS:
            raise ApiRequestError(
                400,
                "unexpected_field",
                f"Unexpected multipart field: {name}",
            )

        field_counts[name] += 1
        if name == "audio" and not isinstance(value, StarletteUploadFile):
            raise ApiRequestError(422, "invalid_audio", "Audio must be uploaded as a file.")
        if name != "audio" and not isinstance(value, str):
            raise ApiRequestError(
                400,
                "invalid_field",
                f"Multipart field {name} must be text.",
            )

    if field_counts["audio"] != 1:
        raise ApiRequestError(
            400 if field_counts["audio"] > 1 else 422,
            "duplicate_audio" if field_counts["audio"] > 1 else "missing_audio",
            "Exactly one audio file is required.",
        )
    for name in ("language", "provider"):
        if field_counts[name] > 1:
            raise ApiRequestError(
                400,
                "duplicate_field",
                f"Multipart field {name} may only be provided once.",
            )


async def _read_transcription_form(
    request: Request,
    *,
    max_upload_bytes: int,
) -> tuple[bytes, str, str | None, str | None]:
    try:
        form = await request.form(
            max_files=1,
            max_fields=2,
            max_part_size=MAX_MULTIPART_FIELD_BYTES,
        )
    except MultipartParseError as error:
        raise ApiRequestError(
            400,
            "invalid_multipart",
            "The multipart request could not be parsed.",
        ) from error
    try:
        _validate_multipart_form(form)
        upload = cast(StarletteUploadFile, form["audio"])
        provider = cast(str | None, form.get("provider"))
        language = cast(str | None, form.get("language"))

        content_type = normalize_media_type(upload.content_type)
        if content_type not in ALLOWED_MEDIA_TYPES:
            raise MediaValidationError(
                "unsupported_media_type",
                "Only WebM or MP4 audio recordings are supported.",
                status_code=415,
            )

        content = await _read_upload(upload, max_bytes=max_upload_bytes)
        return content, content_type, provider, language
    finally:
        await form.close()


async def _inspect_uploaded_audio(
    application: FastAPI,
    content: bytes,
    *,
    content_type: str,
    max_duration_seconds: float,
) -> InspectedAudio:
    async with application.state.media_inspection_semaphore:
        loop = asyncio.get_running_loop()
        inspection = partial(
            application.state.media_inspector,
            content,
            content_type=content_type,
            max_duration_seconds=max_duration_seconds,
        )
        return await loop.run_in_executor(
            application.state.media_inspection_executor,
            inspection,
        )


def _provider_error_status(error: TranscriptionProviderError) -> int:
    if error.code == "audio_too_long":
        return 413
    if error.code == "unsupported_language":
        return 400
    if error.code == "timeout":
        return 504
    if error.retryable:
        return 503
    return 502


def create_app(
    *,
    settings: Settings | None = None,
    service: TranscriptionServiceLike | None = None,
    media_inspector: MediaInspector = inspect_audio,
    client_factory: ClientFactory = create_transcription_http_client,
    generation_client_factory: ClientFactory = create_generation_http_client,
    rag_engine: RagEngine | None = None,
    rag_engine_builder: RagEngineBuilder = build_grounded_rag_engine,
) -> FastAPI:
    runtime_settings = settings or Settings()

    @asynccontextmanager
    async def lifespan(application: FastAPI) -> AsyncIterator[None]:
        manifest_path = runtime_settings.data_manifest
        manifest = (
            load_manifest(manifest_path, verify_files=True)
            if manifest_path is not None
            else None
        )
        application.state.data_status = DataStatus(
            status="ready" if manifest is not None else "not_configured",
            revision=manifest.source.revision if manifest is not None else None,
        )
        application.state.transcription_slots = asyncio.Semaphore(
            runtime_settings.max_concurrent_transcriptions
        )
        media_executor = ThreadPoolExecutor(
            max_workers=MEDIA_INSPECTION_WORKERS,
            thread_name_prefix="subarag-media",
        )
        application.state.media_inspection_executor = media_executor
        application.state.media_inspection_semaphore = asyncio.Semaphore(
            MEDIA_INSPECTION_WORKERS
        )

        try:
            async with AsyncExitStack() as stack:
                if service is not None:
                    application.state.transcription_service = service
                else:
                    transcription_client = await stack.enter_async_context(client_factory())
                    application.state.transcription_service = _build_service(
                        runtime_settings,
                        transcription_client,
                    )

                active_engine = rag_engine
                if active_engine is None:
                    artifact_manifests = _configured_retrieval_manifests(runtime_settings)
                    if artifact_manifests is None:
                        active_engine = UnavailableRagEngine()
                    else:
                        generation_client = await stack.enter_async_context(
                            generation_client_factory()
                        )
                        active_engine = rag_engine_builder(
                            artifact_manifests=artifact_manifests,
                            gemini_api_key=runtime_settings.gemini_api_key.get_secret_value(),
                            generation_client=generation_client,
                            gemini_model=runtime_settings.gemini_model,
                        )
                application.state.rag_engine = active_engine
                yield
        finally:
            media_executor.shutdown(wait=True, cancel_futures=True)

    application = FastAPI(
        title="SubaRAG API",
        version="0.1.0",
        lifespan=lifespan,
    )
    application.state.settings = runtime_settings
    application.state.media_inspector = media_inspector
    application.add_middleware(
        RequestBodyLimitMiddleware,
        max_bytes=runtime_settings.max_upload_bytes + MULTIPART_OVERHEAD_BYTES,
    )
    if runtime_settings.preview_password is not None:
        application.add_middleware(
            PreviewAccessMiddleware,
            password=runtime_settings.preview_password.get_secret_value(),
        )

    @application.exception_handler(ApiRequestError)
    async def handle_api_error(
        _request: Request,
        error: ApiRequestError,
    ) -> JSONResponse:
        return _error_response(
            status_code=error.status_code,
            code=error.code,
            message=str(error),
        )

    @application.exception_handler(MediaValidationError)
    async def handle_media_error(
        _request: Request,
        error: MediaValidationError,
    ) -> JSONResponse:
        return _error_response(
            status_code=error.status_code,
            code=error.code,
            message=str(error),
        )

    @application.exception_handler(RequestValidationError)
    async def handle_request_validation_error(
        _request: Request,
        _error: RequestValidationError,
    ) -> JSONResponse:
        return _error_response(
            status_code=422,
            code="invalid_request",
            message="The request is invalid.",
        )

    @application.exception_handler(StarletteHTTPException)
    async def handle_http_exception(
        _request: Request,
        error: StarletteHTTPException,
    ) -> JSONResponse:
        if error.status_code == 400:
            return _error_response(
                status_code=400,
                code="invalid_multipart",
                message="The multipart request could not be parsed.",
            )
        return _error_response(
            status_code=error.status_code,
            code="http_error",
            message="The request could not be completed.",
        )

    @application.exception_handler(UnknownTranscriptionProviderError)
    async def handle_unknown_provider(
        _request: Request,
        _error: UnknownTranscriptionProviderError,
    ) -> JSONResponse:
        return _error_response(
            status_code=400,
            code="unknown_provider",
            message="The requested transcription provider is not available.",
        )

    @application.exception_handler(TranscriptionProviderError)
    async def handle_provider_error(
        _request: Request,
        error: TranscriptionProviderError,
    ) -> JSONResponse:
        status_code = _provider_error_status(error)
        safe_message = (
            str(error)
            if error.code in {"audio_too_long", "unsupported_language"}
            else f"{error.provider} transcription failed."
        )
        return _error_response(
            status_code=status_code,
            code=error.code,
            message=safe_message,
            retryable=error.retryable,
        )

    @application.get("/healthz", response_model=HealthResponse)
    async def health(request: Request) -> HealthResponse:
        active_service: TranscriptionServiceLike = request.app.state.transcription_service
        return HealthResponse(
            status="ok",
            providers=list(active_service.available_providers),
        )

    @application.get("/api/status", response_model=ApplicationStatus)
    async def application_status(request: Request) -> ApplicationStatus:
        return ApplicationStatus(
            transcription=TranscriptionStatus(provider=runtime_settings.transcription_provider),
            rag=RagStatus(status=request.app.state.rag_engine.status),
            data=request.app.state.data_status,
        )

    @application.post("/api/queries", response_model=QueryResponse)
    async def query(query: RagQuery) -> QueryResponse | JSONResponse:
        started_at = perf_counter()
        active_engine: RagEngine = application.state.rag_engine
        try:
            async with asyncio.timeout(runtime_settings.query_timeout_seconds):
                result = await active_engine.answer(query)
        except TimeoutError:
            return _error_response(
                status_code=504,
                code="query_timeout",
                message="The answer service took too long. Please try again.",
                retryable=True,
            )
        except Exception:
            return _error_response(
                status_code=502,
                code="query_failed",
                message="The answer service could not complete your question.",
                retryable=True,
            )
        return QueryResponse(
            **result.model_dump(),
            request_id=str(uuid4()),
            question=query.question,
            latency_ms=round((perf_counter() - started_at) * 1000, 2),
        )

    @application.post(
        "/api/transcriptions",
        response_model=TranscriptionResponse,
        openapi_extra={
            "requestBody": {
                "required": True,
                "content": {
                    "multipart/form-data": {
                        "schema": {
                            "type": "object",
                            "required": ["audio"],
                            "properties": {
                                "audio": {
                                    "type": "string",
                                    "format": "binary",
                                    "description": "Browser-recorded WebM or MP4 audio",
                                },
                                "provider": {"type": "string"},
                                "language": {"type": "string"},
                            },
                        }
                    }
                },
            }
        },
        responses={
            400: {"model": ErrorResponse},
            413: {"model": ErrorResponse},
            415: {"model": ErrorResponse},
            422: {"model": ErrorResponse},
            502: {"model": ErrorResponse},
            503: {"model": ErrorResponse},
            504: {"model": ErrorResponse},
        },
    )
    async def transcribe(
        request: Request,
    ) -> TranscriptionResponse:
        content, content_type, provider, language = await _read_transcription_form(
            request,
            max_upload_bytes=runtime_settings.max_upload_bytes,
        )
        selected_provider = provider.strip().lower() if provider is not None else None
        if selected_provider == "":
            raise ApiRequestError(
                400,
                "invalid_provider",
                "Provider cannot be empty.",
            )
        selected_language = language.strip().lower() if language is not None else None
        if selected_language in {"", "auto"}:
            selected_language = None
        if selected_language is not None and len(selected_language) > 32:
            raise ApiRequestError(
                400,
                "invalid_language",
                "Language code is too long.",
            )

        if selected_provider is not None and not runtime_settings.allow_provider_selection:
            raise ApiRequestError(
                400,
                "provider_selection_disabled",
                "Provider selection is disabled.",
            )

        active_service: TranscriptionServiceLike = request.app.state.transcription_service
        if (
            selected_provider is not None
            and selected_provider not in active_service.available_providers
        ):
            raise UnknownTranscriptionProviderError(
                f"Transcription provider is not registered: {selected_provider}"
            )

        inspected = await _inspect_uploaded_audio(
            request.app,
            content,
            content_type=content_type,
            max_duration_seconds=runtime_settings.max_audio_seconds,
        )
        extension = "webm" if inspected.content_type == "audio/webm" else "m4a"
        transcription_request = TranscriptionRequest(
            audio=AudioPayload(
                content=content,
                content_type=inspected.content_type,
                filename=f"recording.{extension}",
                duration_seconds=inspected.duration_seconds,
            ),
            language=selected_language,
        )

        slots = request.app.state.transcription_slots
        try:
            await asyncio.wait_for(slots.acquire(), timeout=0.1)
        except TimeoutError:
            return _error_response(
                status_code=503,
                code="service_busy",
                message="Transcription is busy. Please try again shortly.",
                retryable=True,
            )
        try:
            result = await active_service.transcribe(
                transcription_request,
                provider=selected_provider,
            )
        finally:
            slots.release()
        return TranscriptionResponse(
            text=result.text,
            provider=result.provider,
            latency_ms=result.latency_ms,
            duration_seconds=inspected.duration_seconds,
            language=result.language,
            language_confidence=result.language_confidence,
            segments=list(result.segments),
            provider_request_id=result.provider_request_id,
        )

    if runtime_settings.frontend_dist is not None:
        frontend_dist = runtime_settings.frontend_dist.resolve()
        index = frontend_dist / "index.html"
        if not index.is_file() or not (frontend_dist / "assets").is_dir():
            raise RuntimeError("Frontend build is missing. Run npm run build before serving it.")
        application.mount("/assets", StaticFiles(directory=frontend_dist / "assets"), name="assets")

        @application.get("/", include_in_schema=False)
        async def index_page() -> FileResponse:
            return FileResponse(index, headers={"Cache-Control": "no-cache"})

    return application


app = create_app()
