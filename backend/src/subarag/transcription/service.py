"""Provider selection and normalized timing for speech transcription."""

from collections.abc import Iterable
from time import perf_counter

from .errors import (
    DuplicateTranscriptionProviderError,
    TranscriptionProviderError,
    UnknownTranscriptionProviderError,
)
from .models import TranscriptionRequest, TranscriptionResult
from .provider import TranscriptionProvider


def _normalize_provider_name(name: str) -> str:
    normalized_name = name.strip().lower()
    if not normalized_name:
        raise ValueError("Provider name cannot be empty.")
    return normalized_name


class TranscriptionService:
    def __init__(
        self,
        providers: Iterable[TranscriptionProvider],
        *,
        default_provider: str,
    ) -> None:
        self._providers: dict[str, TranscriptionProvider] = {}

        for provider in providers:
            provider_name = _normalize_provider_name(provider.name)
            if provider_name in self._providers:
                raise DuplicateTranscriptionProviderError(
                    f"Transcription provider is already registered: {provider_name}"
                )
            self._providers[provider_name] = provider

        self._default_provider = _normalize_provider_name(default_provider)
        if self._default_provider not in self._providers:
            raise UnknownTranscriptionProviderError(
                f"Default transcription provider is not registered: {self._default_provider}"
            )

    @property
    def available_providers(self) -> tuple[str, ...]:
        return tuple(self._providers)

    async def transcribe(
        self,
        request: TranscriptionRequest,
        *,
        provider: str | None = None,
    ) -> TranscriptionResult:
        provider_name = (
            self._default_provider if provider is None else _normalize_provider_name(provider)
        )
        selected_provider = self._providers.get(provider_name)
        if selected_provider is None:
            raise UnknownTranscriptionProviderError(
                f"Transcription provider is not registered: {provider_name}"
            )

        duration_seconds = request.audio.duration_seconds
        max_audio_seconds = selected_provider.capabilities.max_audio_seconds
        if (
            duration_seconds is not None
            and max_audio_seconds is not None
            and duration_seconds > max_audio_seconds
        ):
            raise TranscriptionProviderError(
                provider_name,
                (
                    f"Audio duration {duration_seconds:.1f}s exceeds the "
                    f"{provider_name} limit of {max_audio_seconds:.1f}s."
                ),
                code="audio_too_long",
            )

        started_at = perf_counter()
        transcript = await selected_provider.transcribe(request)
        latency_ms = (perf_counter() - started_at) * 1_000

        return TranscriptionResult(
            text=transcript.text,
            provider=provider_name,
            latency_ms=latency_ms,
            language=transcript.language,
            language_confidence=transcript.language_confidence,
            segments=transcript.segments,
            provider_request_id=transcript.provider_request_id,
        )
