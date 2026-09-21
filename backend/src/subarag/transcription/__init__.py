"""Provider-neutral speech transcription contracts."""

from .errors import (
    DuplicateTranscriptionProviderError,
    TranscriptionProviderError,
    UnknownTranscriptionProviderError,
)
from .models import (
    AudioPayload,
    ProviderTranscript,
    TranscriptSegment,
    TranscriptionCapabilities,
    TranscriptionRequest,
    TranscriptionResult,
)
from .provider import TranscriptionProvider
from .service import TranscriptionService

__all__ = [
    "AudioPayload",
    "DuplicateTranscriptionProviderError",
    "ProviderTranscript",
    "TranscriptSegment",
    "TranscriptionCapabilities",
    "TranscriptionProvider",
    "TranscriptionProviderError",
    "TranscriptionRequest",
    "TranscriptionResult",
    "TranscriptionService",
    "UnknownTranscriptionProviderError",
]
