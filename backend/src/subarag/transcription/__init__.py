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
from .providers import (
    ElevenLabsTranscriptionProvider,
    SarvamTranscriptionProvider,
    create_transcription_http_client,
)
from .service import TranscriptionService

__all__ = [
    "AudioPayload",
    "DuplicateTranscriptionProviderError",
    "ElevenLabsTranscriptionProvider",
    "ProviderTranscript",
    "SarvamTranscriptionProvider",
    "TranscriptSegment",
    "TranscriptionCapabilities",
    "TranscriptionProvider",
    "TranscriptionProviderError",
    "TranscriptionRequest",
    "TranscriptionResult",
    "TranscriptionService",
    "UnknownTranscriptionProviderError",
    "create_transcription_http_client",
]
