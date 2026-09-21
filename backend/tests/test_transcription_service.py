import asyncio
from dataclasses import dataclass, field

import pytest

from subarag.transcription import (
    AudioPayload,
    DuplicateTranscriptionProviderError,
    ProviderTranscript,
    TranscriptionCapabilities,
    TranscriptionProviderError,
    TranscriptionRequest,
    TranscriptionService,
    UnknownTranscriptionProviderError,
)


@dataclass
class FakeProvider:
    name: str
    response: ProviderTranscript
    capabilities: TranscriptionCapabilities = field(
        default_factory=TranscriptionCapabilities
    )
    requests: list[TranscriptionRequest] = field(default_factory=list)

    async def transcribe(self, request: TranscriptionRequest) -> ProviderTranscript:
        self.requests.append(request)
        return self.response


def make_request(*, language: str | None = "auto") -> TranscriptionRequest:
    return TranscriptionRequest(
        audio=AudioPayload(
            content=b"voice-data",
            content_type="Audio/WebM; codecs=opus",
            filename="question.webm",
            duration_seconds=4.2,
        ),
        language=language,
    )


def test_audio_request_normalizes_browser_metadata() -> None:
    request = make_request()

    assert request.audio.content_type == "audio/webm"
    assert request.audio.duration_seconds == 4.2
    assert request.language is None


@pytest.mark.parametrize(
    ("kwargs", "message"),
    [
        ({"content": b"", "content_type": "audio/webm"}, "cannot be empty"),
        ({"content": b"voice", "content_type": "text/plain"}, "Unsupported"),
        (
            {"content": b"voice", "content_type": "audio/webm", "duration_seconds": 0},
            "greater than zero",
        ),
    ],
)
def test_audio_payload_rejects_invalid_input(kwargs: dict[str, object], message: str) -> None:
    with pytest.raises(ValueError, match=message):
        AudioPayload(**kwargs)  # type: ignore[arg-type]


def test_service_routes_to_the_default_provider() -> None:
    sarvam = FakeProvider(
        name="sarvam",
        response=ProviderTranscript(text="गोवा की राजधानी क्या है?", language="hi-IN"),
    )
    elevenlabs = FakeProvider(
        name="elevenlabs",
        response=ProviderTranscript(text="What is the capital of Goa?", language="en"),
    )
    request = make_request(language="hi-IN")
    service = TranscriptionService([sarvam, elevenlabs], default_provider="sarvam")

    result = asyncio.run(service.transcribe(request))

    assert result.provider == "sarvam"
    assert result.text == "गोवा की राजधानी क्या है?"
    assert result.language == "hi-IN"
    assert result.latency_ms >= 0
    assert sarvam.requests == [request]
    assert elevenlabs.requests == []


def test_service_allows_explicit_provider_selection() -> None:
    sarvam = FakeProvider(name="sarvam", response=ProviderTranscript(text="सरवम"))
    elevenlabs = FakeProvider(
        name="elevenlabs",
        response=ProviderTranscript(text="ElevenLabs", provider_request_id="request-42"),
    )
    service = TranscriptionService([sarvam, elevenlabs], default_provider="sarvam")

    result = asyncio.run(service.transcribe(make_request(), provider=" ELEVENLABS "))

    assert result.provider == "elevenlabs"
    assert result.text == "ElevenLabs"
    assert result.provider_request_id == "request-42"


def test_service_rejects_unknown_and_duplicate_providers() -> None:
    sarvam = FakeProvider(name="sarvam", response=ProviderTranscript(text="text"))
    service = TranscriptionService([sarvam], default_provider="sarvam")

    with pytest.raises(UnknownTranscriptionProviderError, match="not registered"):
        asyncio.run(service.transcribe(make_request(), provider="missing"))

    with pytest.raises(DuplicateTranscriptionProviderError, match="already registered"):
        TranscriptionService(
            [sarvam, FakeProvider(name=" SARVAM ", response=ProviderTranscript(text="other"))],
            default_provider="sarvam",
        )


def test_service_enforces_provider_duration_limit_before_upload() -> None:
    sarvam = FakeProvider(
        name="sarvam",
        response=ProviderTranscript(text="text"),
        capabilities=TranscriptionCapabilities(max_audio_seconds=30, supports_realtime=True),
    )
    service = TranscriptionService([sarvam], default_provider="sarvam")
    request = TranscriptionRequest(
        audio=AudioPayload(
            content=b"voice-data",
            content_type="audio/webm",
            duration_seconds=31,
        )
    )

    with pytest.raises(TranscriptionProviderError, match="exceeds") as exc_info:
        asyncio.run(service.transcribe(request))

    assert exc_info.value.code == "audio_too_long"
    assert sarvam.requests == []
