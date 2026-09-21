"""Errors exposed by the transcription service boundary."""


class TranscriptionServiceError(Exception):
    """Base exception for transcription service failures."""


class DuplicateTranscriptionProviderError(TranscriptionServiceError):
    """Raised when two providers register the same normalized name."""


class UnknownTranscriptionProviderError(TranscriptionServiceError):
    """Raised when a request selects a provider that is not registered."""


class TranscriptionProviderError(TranscriptionServiceError):
    """Normalized provider failure safe for the API layer to inspect."""

    def __init__(
        self,
        provider: str,
        message: str,
        *,
        code: str = "provider_error",
        retryable: bool = False,
    ) -> None:
        super().__init__(message)
        self.provider = provider
        self.code = code
        self.retryable = retryable
