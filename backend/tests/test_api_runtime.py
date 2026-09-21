import asyncio

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
    )

    rendered = repr(settings)
    assert "sarvam-secret" not in rendered
    assert "eleven-secret" not in rendered


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
