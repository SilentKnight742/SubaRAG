"""Validated runtime configuration for the transcription API."""

from math import isfinite
from pathlib import Path
from typing import Literal

from pydantic import Field, SecretStr, field_validator
from pydantic_settings import BaseSettings, SettingsConfigDict


BACKEND_ROOT = Path(__file__).resolve().parents[3]


class Settings(BaseSettings):
    model_config = SettingsConfigDict(
        env_file=BACKEND_ROOT / ".env",
        env_file_encoding="utf-8",
        extra="ignore",
        populate_by_name=True,
    )

    sarvam_api_key: SecretStr | None = Field(
        default=None,
        validation_alias="SARVAM_API_KEY",
    )
    elevenlabs_api_key: SecretStr | None = Field(
        default=None,
        validation_alias="ELEVENLABS_API_KEY",
    )
    gemini_api_key: SecretStr | None = Field(
        default=None,
        validation_alias="GEMINI_API_KEY",
    )
    gemini_model: str = Field(
        default="gemini-2.5-flash-lite",
        validation_alias="SUBARAG_GEMINI_MODEL",
    )
    transcription_provider: Literal["sarvam", "elevenlabs"] = Field(
        default="elevenlabs",
        validation_alias="SUBARAG_TRANSCRIPTION_PROVIDER",
    )
    allow_provider_selection: bool = Field(
        default=False,
        validation_alias="SUBARAG_ALLOW_PROVIDER_SELECTION",
    )
    frontend_dist: Path | None = Field(
        default=None,
        validation_alias="SUBARAG_FRONTEND_DIST",
    )
    data_manifest: Path | None = Field(
        default=None,
        validation_alias="SUBARAG_DATA_MANIFEST",
    )
    preview_password: SecretStr | None = Field(
        default=None,
        validation_alias="SUBARAG_PREVIEW_PASSWORD",
    )
    query_timeout_seconds: float = Field(default=30, gt=0, le=120, allow_inf_nan=False)
    max_concurrent_transcriptions: int = Field(default=2, ge=1, le=8)
    max_audio_seconds: float = 30
    max_upload_bytes: int = 8 * 1024 * 1024

    @field_validator(
        "sarvam_api_key",
        "elevenlabs_api_key",
        "gemini_api_key",
        "preview_password",
        mode="before",
    )
    @classmethod
    def empty_secrets_are_unset(cls, value: object) -> object:
        return None if value == "" else value

    @field_validator("max_audio_seconds")
    @classmethod
    def positive_audio_limit(cls, value: float) -> float:
        if not isfinite(value) or value <= 0:
            raise ValueError("Maximum audio duration must be greater than zero.")
        return value

    @field_validator("max_upload_bytes")
    @classmethod
    def positive_upload_limit(cls, value: int) -> int:
        if value <= 0:
            raise ValueError("Maximum upload size must be greater than zero.")
        return value
