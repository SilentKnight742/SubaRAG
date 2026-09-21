"""Shared request and response models for every speech provider."""

from dataclasses import dataclass, field


@dataclass(frozen=True, slots=True)
class AudioPayload:
    content: bytes
    content_type: str
    filename: str = "recording"
    duration_seconds: float | None = None

    def __post_init__(self) -> None:
        if not self.content:
            raise ValueError("Audio content cannot be empty.")

        normalized_content_type = self.content_type.partition(";")[0].strip().lower()
        if not normalized_content_type:
            raise ValueError("Audio content type is required.")

        if not normalized_content_type.startswith("audio/") and normalized_content_type not in {
            "video/mp4",
            "video/webm",
        }:
            raise ValueError(f"Unsupported audio content type: {normalized_content_type}")

        normalized_filename = self.filename.strip()
        if not normalized_filename:
            raise ValueError("Audio filename is required.")

        if self.duration_seconds is not None and self.duration_seconds <= 0:
            raise ValueError("Audio duration must be greater than zero when provided.")

        object.__setattr__(self, "content_type", normalized_content_type)
        object.__setattr__(self, "filename", normalized_filename)


@dataclass(frozen=True, slots=True)
class TranscriptionRequest:
    audio: AudioPayload
    language: str | None = None

    def __post_init__(self) -> None:
        if self.language is None:
            return

        normalized_language = self.language.strip().lower()
        object.__setattr__(
            self,
            "language",
            None if normalized_language in {"", "auto"} else normalized_language,
        )


@dataclass(frozen=True, slots=True)
class TranscriptSegment:
    text: str
    start_seconds: float
    end_seconds: float
    confidence: float | None = None
    speaker: str | None = None

    def __post_init__(self) -> None:
        if self.start_seconds < 0 or self.end_seconds < self.start_seconds:
            raise ValueError("Transcript word timestamps are invalid.")

        if self.confidence is not None and not 0 <= self.confidence <= 1:
            raise ValueError("Transcript segment confidence must be between zero and one.")


@dataclass(frozen=True, slots=True)
class TranscriptionCapabilities:
    max_audio_seconds: float | None = None
    supports_realtime: bool = False

    def __post_init__(self) -> None:
        if self.max_audio_seconds is not None and self.max_audio_seconds <= 0:
            raise ValueError("Maximum audio duration must be greater than zero.")


@dataclass(frozen=True, slots=True)
class ProviderTranscript:
    text: str
    language: str | None = None
    language_confidence: float | None = None
    segments: tuple[TranscriptSegment, ...] = field(default_factory=tuple)
    provider_request_id: str | None = None

    def __post_init__(self) -> None:
        if self.language_confidence is not None and not 0 <= self.language_confidence <= 1:
            raise ValueError("Language confidence must be between zero and one.")


@dataclass(frozen=True, slots=True)
class TranscriptionResult:
    text: str
    provider: str
    latency_ms: float
    language: str | None = None
    language_confidence: float | None = None
    segments: tuple[TranscriptSegment, ...] = field(default_factory=tuple)
    provider_request_id: str | None = None
