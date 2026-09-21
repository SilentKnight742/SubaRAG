"""Protocol implemented by concrete speech-to-text providers."""

from typing import Protocol, runtime_checkable

from .models import ProviderTranscript, TranscriptionCapabilities, TranscriptionRequest


@runtime_checkable
class TranscriptionProvider(Protocol):
    @property
    def name(self) -> str:
        """Stable provider identifier used for routing and metrics."""

    @property
    def capabilities(self) -> TranscriptionCapabilities:
        """Limits and modes used to validate and route requests."""

    async def transcribe(self, request: TranscriptionRequest) -> ProviderTranscript:
        """Transcribe one complete audio payload."""
