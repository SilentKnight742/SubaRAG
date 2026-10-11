import asyncio
from dataclasses import dataclass, field

import httpx
import pytest

from subarag.api.app import create_app
from subarag.api.media import InspectedAudio, MediaValidationError
from subarag.api.settings import Settings
from subarag.transcription import (
    TranscriptSegment,
    TranscriptionProviderError,
    TranscriptionRequest,
    TranscriptionResult,
)


@dataclass
class FakeService:
    available_providers: tuple[str, ...] = ("sarvam", "elevenlabs")
    error: Exception | None = None
    requests: list[tuple[TranscriptionRequest, str | None]] = field(default_factory=list)

    async def transcribe(
        self,
        request: TranscriptionRequest,
        *,
        provider: str | None = None,
    ) -> TranscriptionResult:
        self.requests.append((request, provider))
        if self.error is not None:
            raise self.error
        return TranscriptionResult(
            text="गोवा की राजधानी पणजी है।",
            provider=provider or "sarvam",
            latency_ms=123.4,
            language="hi-IN",
            language_confidence=0.98,
            segments=(
                TranscriptSegment(
                    text="गोवा की राजधानी पणजी है।",
                    start_seconds=0,
                    end_seconds=2.4,
                ),
            ),
            provider_request_id="request-1",
        )


@dataclass
class FakeInspector:
    result: InspectedAudio = InspectedAudio(
        duration_seconds=2.4,
        content_type="audio/webm",
        codec="opus",
    )
    error: Exception | None = None
    calls: list[tuple[bytes, str, float]] = field(default_factory=list)

    def __call__(
        self,
        content: bytes,
        *,
        content_type: str,
        max_duration_seconds: float,
    ) -> InspectedAudio:
        self.calls.append((content, content_type, max_duration_seconds))
        if self.error is not None:
            raise self.error
        return self.result


def make_settings(**overrides: object) -> Settings:
    overrides.setdefault("allow_provider_selection", True)
    return Settings(_env_file=None, **overrides)


def send_request(
    service: FakeService,
    method: str,
    path: str,
    *,
    inspector: FakeInspector | None = None,
    request_kwargs: dict[str, object] | None = None,
    **settings_overrides: object,
) -> httpx.Response:
    async def run_request() -> httpx.Response:
        application = create_app(
            settings=make_settings(**settings_overrides),
            service=service,
            media_inspector=inspector or FakeInspector(),
        )
        async with application.router.lifespan_context(application):
            async with httpx.AsyncClient(
                transport=httpx.ASGITransport(app=application),
                base_url="http://testserver",
            ) as client:
                return await client.request(
                    method,
                    path,
                    **(request_kwargs or {}),
                )

    return asyncio.run(run_request())


def test_health_reports_configured_providers_without_secrets() -> None:
    service = FakeService()

    response = send_request(service, "GET", "/healthz")

    assert response.status_code == 200
    assert response.json() == {
        "status": "ok",
        "providers": ["sarvam", "elevenlabs"],
    }


def test_transcription_upload_is_measured_and_normalized() -> None:
    service = FakeService()
    inspector = FakeInspector()

    response = send_request(
        service,
        "POST",
        "/api/transcriptions",
        inspector=inspector,
        request_kwargs={
            "files": {
                "audio": (
                    "untrusted-name.webm",
                    b"browser-audio",
                    "audio/webm; codecs=opus",
                )
            },
            "data": {"provider": "elevenlabs", "language": "hi"},
        },
    )

    assert response.status_code == 200
    assert response.json() == {
        "text": "गोवा की राजधानी पणजी है।",
        "provider": "elevenlabs",
        "latency_ms": 123.4,
        "duration_seconds": 2.4,
        "language": "hi-IN",
        "language_confidence": 0.98,
        "segments": [
            {
                "text": "गोवा की राजधानी पणजी है।",
                "start_seconds": 0.0,
                "end_seconds": 2.4,
                "confidence": None,
                "speaker": None,
            }
        ],
        "provider_request_id": "request-1",
    }
    assert inspector.calls == [(b"browser-audio", "audio/webm", 30)]

    request, provider = service.requests[0]
    assert provider == "elevenlabs"
    assert request.language == "hi"
    assert request.audio.filename == "recording.webm"
    assert request.audio.duration_seconds == 2.4


def test_auto_language_and_default_provider_are_omitted() -> None:
    service = FakeService()

    response = send_request(
        service,
        "POST",
        "/api/transcriptions",
        request_kwargs={
            "files": {"audio": ("recording.webm", b"audio", "audio/webm")},
            "data": {"language": "auto"},
        },
    )

    assert response.status_code == 200
    request, provider = service.requests[0]
    assert provider is None
    assert request.language is None


def test_provider_override_can_be_disabled() -> None:
    service = FakeService()

    response = send_request(
        service,
        "POST",
        "/api/transcriptions",
        request_kwargs={
            "files": {"audio": ("recording.webm", b"audio", "audio/webm")},
            "data": {"provider": "elevenlabs"},
        },
        allow_provider_selection=False,
    )

    assert response.status_code == 400
    assert response.json()["error"]["code"] == "provider_selection_disabled"
    assert service.requests == []


def test_empty_provider_is_rejected_as_a_client_error() -> None:
    service = FakeService()

    response = send_request(
        service,
        "POST",
        "/api/transcriptions",
        request_kwargs={
            "files": {"audio": ("recording.webm", b"audio", "audio/webm")},
            "data": {"provider": "  "},
        },
    )

    assert response.status_code == 400
    assert response.json()["error"]["code"] == "invalid_provider"
    assert service.requests == []


def test_missing_audio_uses_stable_error_shape() -> None:
    response = send_request(FakeService(), "POST", "/api/transcriptions")

    assert response.status_code == 422
    assert response.json() == {
        "error": {
            "code": "missing_audio",
            "message": "Exactly one audio file is required.",
            "retryable": False,
        }
    }


@pytest.mark.parametrize(
    ("files", "expected_code"),
    [
        (
            [
                ("audio", ("one.webm", b"one", "audio/webm")),
                ("audio", ("two.webm", b"two", "audio/webm")),
            ],
            "invalid_multipart",
        ),
        (
            [
                ("audio", ("recording.webm", b"audio", "audio/webm")),
                ("notes", (None, "unexpected")),
            ],
            "unexpected_field",
        ),
    ],
)
def test_multipart_cardinality_is_enforced(
    files: list[tuple[str, tuple[object, ...]]],
    expected_code: str,
) -> None:
    service = FakeService()

    response = send_request(
        service,
        "POST",
        "/api/transcriptions",
        request_kwargs={"files": files},
    )

    assert response.status_code == 400
    assert response.json()["error"]["code"] == expected_code
    assert service.requests == []


@pytest.mark.parametrize(
    ("content_type", "body"),
    [
        ("multipart/form-data", b"missing boundary"),
        ("multipart/form-data; boundary=test", b"not-a-valid-boundary"),
    ],
)
def test_malformed_multipart_uses_stable_error_shape(
    content_type: str,
    body: bytes,
) -> None:
    response = send_request(
        FakeService(),
        "POST",
        "/api/transcriptions",
        request_kwargs={
            "content": body,
            "headers": {"content-type": content_type},
        },
    )

    assert response.status_code == 400
    assert response.json() == {
        "error": {
            "code": "invalid_multipart",
            "message": "The multipart request could not be parsed.",
            "retryable": False,
        }
    }


def test_unknown_provider_is_rejected_before_media_inspection() -> None:
    service = FakeService()
    inspector = FakeInspector()

    response = send_request(
        service,
        "POST",
        "/api/transcriptions",
        inspector=inspector,
        request_kwargs={
            "files": {"audio": ("recording.webm", b"audio", "audio/webm")},
            "data": {"provider": "missing"},
        },
    )

    assert response.status_code == 400
    assert response.json()["error"]["code"] == "unknown_provider"
    assert inspector.calls == []
    assert service.requests == []


@pytest.mark.parametrize(
    ("files", "expected_status", "expected_code"),
    [
        ({"audio": ("recording.txt", b"audio", "text/plain")}, 415, "unsupported_media_type"),
        ({"audio": ("recording.webm", b"", "audio/webm")}, 422, "empty_audio"),
    ],
)
def test_upload_validation_errors(
    files: dict[str, tuple[str, bytes, str]],
    expected_status: int,
    expected_code: str,
) -> None:
    service = FakeService()

    response = send_request(
        service,
        "POST",
        "/api/transcriptions",
        request_kwargs={"files": files},
    )

    assert response.status_code == expected_status
    assert response.json()["error"]["code"] == expected_code
    assert service.requests == []


def test_upload_size_is_bounded_before_provider_call() -> None:
    service = FakeService()

    response = send_request(
        service,
        "POST",
        "/api/transcriptions",
        request_kwargs={
            "files": {"audio": ("recording.webm", b"five!", "audio/webm")}
        },
        max_upload_bytes=4,
    )

    assert response.status_code == 413
    assert response.json()["error"]["code"] == "audio_too_large"
    assert service.requests == []


def test_oversized_content_length_is_rejected_before_parsing() -> None:
    response = send_request(
        FakeService(),
        "POST",
        "/api/transcriptions",
        request_kwargs={
            "content": b"ignored",
            "headers": {
                "content-length": str(100_000),
                "content-type": "multipart/form-data; boundary=test",
            },
        },
        max_upload_bytes=4,
    )

    assert response.status_code == 413
    assert response.json()["error"]["code"] == "request_too_large"


def test_chunked_oversized_body_keeps_normalized_413_response() -> None:
    boundary = "subarag-test"
    body = (
        f"--{boundary}\r\n"
        'Content-Disposition: form-data; name="audio"; filename="recording.webm"\r\n'
        "Content-Type: audio/webm\r\n\r\n"
    ).encode() + (b"x" * 70_000) + f"\r\n--{boundary}--\r\n".encode()

    async def body_chunks():
        for offset in range(0, len(body), 4096):
            yield body[offset : offset + 4096]

    response = send_request(
        FakeService(),
        "POST",
        "/api/transcriptions",
        request_kwargs={
            "content": body_chunks(),
            "headers": {"content-type": f"multipart/form-data; boundary={boundary}"},
        },
        max_upload_bytes=4,
    )

    assert response.status_code == 413
    assert response.json() == {
        "error": {
            "code": "request_too_large",
            "message": "The upload request is too large.",
            "retryable": False,
        }
    }


def test_media_validation_failure_skips_provider_call() -> None:
    service = FakeService()
    inspector = FakeInspector(
        error=MediaValidationError(
            "audio_too_long",
            "Audio is too long.",
            status_code=413,
        )
    )

    response = send_request(
        service,
        "POST",
        "/api/transcriptions",
        inspector=inspector,
        request_kwargs={
            "files": {"audio": ("recording.webm", b"audio", "audio/webm")}
        },
    )

    assert response.status_code == 413
    assert response.json()["error"]["code"] == "audio_too_long"
    assert service.requests == []


@pytest.mark.parametrize(
    ("error", "expected_status", "safe_message"),
    [
        (
            TranscriptionProviderError(
                "sarvam",
                "secret upstream details",
                code="invalid_response",
            ),
            502,
            "sarvam transcription failed.",
        ),
        (
            TranscriptionProviderError(
                "elevenlabs",
                "Timed out",
                code="timeout",
                retryable=True,
            ),
            504,
            "elevenlabs transcription failed.",
        ),
        (
            TranscriptionProviderError(
                "sarvam",
                "Try later",
                code="rate_limit",
                retryable=True,
            ),
            503,
            "sarvam transcription failed.",
        ),
    ],
)
def test_provider_failures_have_stable_safe_http_errors(
    error: TranscriptionProviderError,
    expected_status: int,
    safe_message: str,
) -> None:
    service = FakeService(error=error)

    response = send_request(
        service,
        "POST",
        "/api/transcriptions",
        request_kwargs={
            "files": {"audio": ("recording.webm", b"audio", "audio/webm")}
        },
    )

    assert response.status_code == expected_status
    assert response.json()["error"] == {
        "code": error.code,
        "message": safe_message,
        "retryable": error.retryable,
    }
