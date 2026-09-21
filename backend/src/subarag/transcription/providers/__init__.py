"""Concrete speech-to-text provider adapters."""

from .elevenlabs import ElevenLabsTranscriptionProvider
from .http_client import create_transcription_http_client
from .sarvam import SarvamTranscriptionProvider

__all__ = [
    "ElevenLabsTranscriptionProvider",
    "SarvamTranscriptionProvider",
    "create_transcription_http_client",
]
