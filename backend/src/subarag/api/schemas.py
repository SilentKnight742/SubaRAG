"""Stable JSON response models for the HTTP API."""

from typing import Literal

from pydantic import BaseModel, ConfigDict

from ..rag import RagResult


class TranscriptSegmentResponse(BaseModel):
    model_config = ConfigDict(from_attributes=True)

    text: str
    start_seconds: float
    end_seconds: float
    confidence: float | None = None
    speaker: str | None = None


class TranscriptionResponse(BaseModel):
    text: str
    provider: str
    latency_ms: float
    duration_seconds: float
    language: str | None = None
    language_confidence: float | None = None
    segments: list[TranscriptSegmentResponse]
    provider_request_id: str | None = None


class ErrorDetail(BaseModel):
    code: str
    message: str
    retryable: bool = False


class ErrorResponse(BaseModel):
    error: ErrorDetail


class HealthResponse(BaseModel):
    status: str
    providers: list[str]


class QueryResponse(RagResult):
    request_id: str
    question: str
    latency_ms: float


class TranscriptionStatus(BaseModel):
    provider: Literal["elevenlabs", "sarvam"]


class RagStatus(BaseModel):
    status: Literal["unavailable", "ready"]


class DataStatus(BaseModel):
    status: Literal["not_configured", "ready"]
    revision: str | None = None


class ApplicationStatus(BaseModel):
    transcription: TranscriptionStatus
    rag: RagStatus
    data: DataStatus
