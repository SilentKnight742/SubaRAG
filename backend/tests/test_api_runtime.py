import asyncio
from pathlib import Path

import httpx
import pytest
from pydantic import ValidationError

from subarag.api.app import create_app
from subarag.api.settings import Settings
from subarag.transcription import UnknownTranscriptionProviderError


@pytest.mark.parametrize("value", [0, -1, float("nan"), float("inf")])
def test_settings_reject_invalid_audio_limits(value: float) -> None:
    with pytest.raises(ValidationError):
        Settings(_env_file=None, max_audio_seconds=value)


def test_settings_mask_provider_credentials() -> None:
    settings = Settings(
        _env_file=None,
        sarvam_api_key="sarvam-secret",
        elevenlabs_api_key="eleven-secret",
        gemini_api_key="gemini-secret",
    )

    rendered = repr(settings)
    assert "sarvam-secret" not in rendered
    assert "eleven-secret" not in rendered
    assert "gemini-secret" not in rendered


def test_remote_compact_configuration_is_complete_and_exclusive() -> None:
    common = dict(
        _env_file=None,
        hf_token="hf-secret",
        gemini_api_key="gemini-secret",
        compact_repo="silentkirmada/SubaRAG-data",
        compact_revision="3f0ef0a35e7fd2cf733c6ff6e1ac9a9e49a539d6",
        compact_path="serving/compact/" + "a" * 64,
    )
    settings = Settings(**common)
    assert settings.compact_repo == "silentkirmada/SubaRAG-data"
    with pytest.raises(ValidationError):
        Settings(**{**common, "hf_token": None})
    with pytest.raises(ValidationError):
        Settings(**{**common, "compact_manifest": Path("/bundle/compact-bundle.json")})


def test_lifespan_builds_providers_and_closes_shared_client() -> None:
    client = httpx.AsyncClient(transport=httpx.MockTransport(lambda _request: None))
    application = create_app(
        settings=Settings(
            _env_file=None,
            sarvam_api_key="sarvam-secret",
            elevenlabs_api_key="eleven-secret",
        ),
        client_factory=lambda: client,
    )

    async def run_test() -> None:
        async with application.router.lifespan_context(application):
            assert client.is_closed is False
            assert application.state.transcription_service.available_providers == (
                "sarvam",
                "elevenlabs",
            )
        assert client.is_closed is True

    asyncio.run(run_test())


def test_lifespan_fails_when_default_provider_has_no_key() -> None:
    client = httpx.AsyncClient(transport=httpx.MockTransport(lambda _request: None))
    application = create_app(
        settings=Settings(
            _env_file=None,
            elevenlabs_api_key="eleven-secret",
            transcription_provider="sarvam",
        ),
        client_factory=lambda: client,
    )

    async def run_test() -> None:
        with pytest.raises(UnknownTranscriptionProviderError):
            async with application.router.lifespan_context(application):
                pass
        assert client.is_closed is True

    asyncio.run(run_test())


def test_lifespan_builds_grounded_rag_only_with_complete_configuration() -> None:
    transcription_client = httpx.AsyncClient(
        transport=httpx.MockTransport(lambda _request: None)
    )
    generation_client = httpx.AsyncClient(
        transport=httpx.MockTransport(lambda _request: None)
    )
    calls = []

    class Engine:
        status = "ready"

        async def answer(self, query):
            raise NotImplementedError

    def builder(**kwargs):
        calls.append(kwargs)
        return Engine()

    application = create_app(
        settings=Settings(
            _env_file=None,
            elevenlabs_api_key="eleven-secret",
            gemini_api_key="gemini-secret",
            retrieval_manifest_en=Path("/indexes/en/retrieval-manifest.json"),
            retrieval_manifest_hi=Path("/indexes/hi/retrieval-manifest.json"),
            retrieval_manifest_mr=Path("/indexes/mr/retrieval-manifest.json"),
        ),
        client_factory=lambda: transcription_client,
        generation_client_factory=lambda: generation_client,
        rag_engine_builder=builder,
    )

    async def run_test() -> None:
        async with application.router.lifespan_context(application):
            assert application.state.rag_engine.status == "ready"
            assert transcription_client.is_closed is False
            assert generation_client.is_closed is False
        assert transcription_client.is_closed is True
        assert generation_client.is_closed is True

    asyncio.run(run_test())
    assert calls[0]["artifact_manifests"] == {
        "en": Path("/indexes/en/retrieval-manifest.json"),
        "hi": Path("/indexes/hi/retrieval-manifest.json"),
        "mr": Path("/indexes/mr/retrieval-manifest.json"),
    }
    assert calls[0]["gemini_api_key"] == "gemini-secret"
    assert calls[0]["gemini_model"] == "gemini-3.5-flash-lite"


def test_lifespan_rejects_partial_retrieval_configuration() -> None:
    client = httpx.AsyncClient(transport=httpx.MockTransport(lambda _request: None))
    application = create_app(
        settings=Settings(
            _env_file=None,
            elevenlabs_api_key="eleven-secret",
            gemini_api_key="gemini-secret",
            retrieval_manifest_en=Path("/indexes/en/retrieval-manifest.json"),
        ),
        client_factory=lambda: client,
    )

    async def run_test() -> None:
        with pytest.raises(RuntimeError, match="configured together"):
            async with application.router.lifespan_context(application):
                pass
        assert client.is_closed is True

    asyncio.run(run_test())


def test_gemini_key_alone_keeps_rag_unavailable_without_opening_client() -> None:
    transcription_client = httpx.AsyncClient(
        transport=httpx.MockTransport(lambda _request: None)
    )
    generation_clients = 0

    def generation_client_factory():
        nonlocal generation_clients
        generation_clients += 1
        return httpx.AsyncClient()

    application = create_app(
        settings=Settings(
            _env_file=None,
            elevenlabs_api_key="eleven-secret",
            gemini_api_key="gemini-secret",
        ),
        client_factory=lambda: transcription_client,
        generation_client_factory=generation_client_factory,
    )

    async def run_test() -> None:
        async with application.router.lifespan_context(application):
            assert application.state.rag_engine.status == "unavailable"

    asyncio.run(run_test())
    assert generation_clients == 0
