import asyncio

import httpx
import pytest

from subarag.transcription import (
    AudioPayload,
    ElevenLabsTranscriptionProvider,
    SarvamTranscriptionProvider,
    TranscriptionProviderError,
    TranscriptionRequest,
    create_transcription_http_client,
)


def make_request(*, language: str | None = "hi") -> TranscriptionRequest:
    return TranscriptionRequest(
        audio=AudioPayload(
            content=b"webm-audio",
            content_type="audio/webm",
            filename="question.webm",
            duration_seconds=4,
        ),
        language=language,
    )


def test_sarvam_builds_request_and_normalizes_response() -> None:
    async def run_test() -> None:
        async def handler(request: httpx.Request) -> httpx.Response:
            body = await request.aread()
            assert request.url == "https://api.sarvam.ai/speech-to-text"
            assert request.headers["api-subscription-key"] == "sarvam-secret"
            assert b'name="model"' in body and b"saaras:v3" in body
            assert b'name="language_code"' in body and b"hi-IN" in body
            assert b'filename="question.webm"' in body
            assert b"webm-audio" in body
            return httpx.Response(
                200,
                json={
                    "request_id": "sarvam-1",
                    "transcript": "गोवा की राजधानी क्या है?",
                    "language_code": "hi-IN",
                    "language_probability": 0.97,
                    "timestamps": {
                        "words": ["गोवा की राजधानी", "क्या है"],
                        "start_time_seconds": [0.1, 1.2],
                        "end_time_seconds": [1.1, 1.8],
                    },
                },
            )

        async with httpx.AsyncClient(transport=httpx.MockTransport(handler)) as client:
            provider = SarvamTranscriptionProvider(api_key="sarvam-secret", client=client)
            result = await provider.transcribe(make_request())

        assert result.text == "गोवा की राजधानी क्या है?"
        assert result.language == "hi-IN"
        assert result.language_confidence == 0.97
        assert result.provider_request_id == "sarvam-1"
        assert [segment.text for segment in result.segments] == [
            "गोवा की राजधानी",
            "क्या है",
        ]

    asyncio.run(run_test())


def test_elevenlabs_builds_request_and_normalizes_response() -> None:
    async def run_test() -> None:
        async def handler(request: httpx.Request) -> httpx.Response:
            body = await request.aread()
            assert request.url == "https://api.elevenlabs.io/v1/speech-to-text"
            assert request.headers["xi-api-key"] == "eleven-secret"
            assert b'name="model_id"' in body and b"scribe_v2" in body
            assert b'name="timestamps_granularity"' in body and b"none" in body
            assert b'name="tag_audio_events"' in body and b"false" in body
            assert b'name="language_code"' in body and b"hi" in body
            assert b"webm-audio" in body
            return httpx.Response(
                200,
                json={
                    "transcription_id": "eleven-1",
                    "text": "गोवा की राजधानी क्या है?",
                    "language_code": "hin",
                    "language_probability": 0.94,
                },
            )

        async with httpx.AsyncClient(transport=httpx.MockTransport(handler)) as client:
            provider = ElevenLabsTranscriptionProvider(api_key="eleven-secret", client=client)
            result = await provider.transcribe(make_request())

        assert result.text == "गोवा की राजधानी क्या है?"
        assert result.language == "hin"
        assert result.language_confidence == 0.94
        assert result.provider_request_id == "eleven-1"

    asyncio.run(run_test())


@pytest.mark.parametrize(
    ("language", "expected_code"),
    [
        ("ta", "ta-IN"),
        ("kok-IN", "kok-IN"),
        ("or", "od-IN"),
    ],
)
def test_sarvam_normalizes_supported_languages(
    language: str,
    expected_code: str,
) -> None:
    async def run_test() -> None:
        async def handler(request: httpx.Request) -> httpx.Response:
            body = await request.aread()
            assert b'name="language_code"' in body
            assert expected_code.encode() in body
            return httpx.Response(200, json={"transcript": "question"})

        async with httpx.AsyncClient(transport=httpx.MockTransport(handler)) as client:
            provider = SarvamTranscriptionProvider(api_key="key", client=client)
            result = await provider.transcribe(make_request(language=language))

        assert result.text == "question"

    asyncio.run(run_test())


def test_sarvam_rejects_unsupported_language_before_upload() -> None:
    async def run_test() -> None:
        requests = 0

        def handler(_request: httpx.Request) -> httpx.Response:
            nonlocal requests
            requests += 1
            return httpx.Response(200, json={"transcript": "question"})

        async with httpx.AsyncClient(transport=httpx.MockTransport(handler)) as client:
            provider = SarvamTranscriptionProvider(api_key="key", client=client)
            with pytest.raises(TranscriptionProviderError) as exc_info:
                await provider.transcribe(make_request(language="fr"))

        assert exc_info.value.code == "unsupported_language"
        assert requests == 0

    asyncio.run(run_test())


@pytest.mark.parametrize(
    ("provider_name", "status_code", "payload", "expected_code", "retryable"),
    [
        (
            "sarvam",
            429,
            {"error": {"code": "rate_limit", "message": "Try later"}},
            "rate_limit",
            True,
        ),
        (
            "elevenlabs",
            400,
            {"detail": {"status": "invalid_audio", "message": "Bad audio"}},
            "invalid_audio",
            False,
        ),
    ],
)
def test_provider_errors_are_normalized(
    provider_name: str,
    status_code: int,
    payload: dict[str, object],
    expected_code: str,
    retryable: bool,
) -> None:
    async def run_test() -> None:
        transport = httpx.MockTransport(
            lambda _request: httpx.Response(status_code, json=payload)
        )
        async with httpx.AsyncClient(transport=transport) as client:
            provider = (
                SarvamTranscriptionProvider(api_key="key", client=client)
                if provider_name == "sarvam"
                else ElevenLabsTranscriptionProvider(api_key="key", client=client)
            )

            with pytest.raises(TranscriptionProviderError) as exc_info:
                await provider.transcribe(make_request())

        assert exc_info.value.provider == provider_name
        assert exc_info.value.code == expected_code
        assert exc_info.value.retryable is retryable

    asyncio.run(run_test())


@pytest.mark.parametrize(
    "response",
    [
        httpx.Response(200, content=b"not-json"),
        httpx.Response(200, json=["not", "an", "object"]),
        httpx.Response(200, json={"language_code": "hi-IN"}),
        httpx.Response(
            200,
            json={
                "transcript": "question",
                "timestamps": {
                    "words": ["question"],
                    "start_time_seconds": [0],
                    "end_time_seconds": [],
                },
            },
        ),
    ],
)
def test_sarvam_rejects_malformed_success_responses(response: httpx.Response) -> None:
    async def run_test() -> None:
        transport = httpx.MockTransport(lambda _request: response)
        async with httpx.AsyncClient(transport=transport) as client:
            provider = SarvamTranscriptionProvider(api_key="key", client=client)
            with pytest.raises(TranscriptionProviderError) as exc_info:
                await provider.transcribe(make_request())

        assert exc_info.value.code == "invalid_response"

    asyncio.run(run_test())


def test_elevenlabs_rejects_missing_transcript() -> None:
    async def run_test() -> None:
        transport = httpx.MockTransport(
            lambda _request: httpx.Response(200, json={"language_code": "hin"})
        )
        async with httpx.AsyncClient(transport=transport) as client:
            provider = ElevenLabsTranscriptionProvider(api_key="key", client=client)
            with pytest.raises(TranscriptionProviderError) as exc_info:
                await provider.transcribe(make_request())

        assert exc_info.value.code == "invalid_response"

    asyncio.run(run_test())


@pytest.mark.parametrize(
    ("error", "expected_code"),
    [
        (httpx.ReadTimeout("slow provider"), "timeout"),
        (httpx.ConnectError("provider unavailable"), "network_error"),
    ],
)
def test_transport_errors_are_normalized(
    error: httpx.RequestError,
    expected_code: str,
) -> None:
    async def run_test() -> None:
        def handler(request: httpx.Request) -> httpx.Response:
            error.request = request
            raise error

        async with httpx.AsyncClient(transport=httpx.MockTransport(handler)) as client:
            provider = ElevenLabsTranscriptionProvider(api_key="key", client=client)
            with pytest.raises(TranscriptionProviderError) as exc_info:
                await provider.transcribe(make_request())

        assert exc_info.value.code == expected_code
        assert exc_info.value.retryable is True

    asyncio.run(run_test())


def test_production_client_uses_transcription_timeouts() -> None:
    async def run_test() -> None:
        async with create_transcription_http_client() as client:
            assert client.timeout.connect == 5
            assert client.timeout.read == 60
            assert client.timeout.write == 30
            assert client.timeout.pool == 5

    asyncio.run(run_test())
