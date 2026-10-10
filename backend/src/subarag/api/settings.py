"""Validated runtime configuration for the transcription API."""

from math import isfinite
from pathlib import Path
from typing import Literal

from pydantic import Field, SecretStr, field_validator, model_validator
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
        default="gemini-3.5-flash-lite",
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
    hf_token: SecretStr | None = Field(default=None, validation_alias="HF_TOKEN")
    data_repo: str | None = Field(
        default=None, validation_alias="SUBARAG_DATA_REPO",
        pattern=r"^[A-Za-z0-9_-]+/[A-Za-z0-9_.-]+$",
    )
    data_revision: str | None = Field(
        default=None, validation_alias="SUBARAG_DATA_REVISION", pattern=r"^[0-9a-f]{40}$",
    )
    data_cache_dir: Path = Field(
        default=BACKEND_ROOT / ".cache" / "corpus",
        validation_alias="SUBARAG_DATA_CACHE_DIR",
    )
    retrieval_manifest_en: Path | None = Field(
        default=None,
        validation_alias="SUBARAG_RETRIEVAL_MANIFEST_EN",
    )
    retrieval_manifest_hi: Path | None = Field(
        default=None,
        validation_alias="SUBARAG_RETRIEVAL_MANIFEST_HI",
    )
    retrieval_manifest_mr: Path | None = Field(
        default=None,
        validation_alias="SUBARAG_RETRIEVAL_MANIFEST_MR",
    )
    retrieval_repo: str | None = Field(
        default=None, validation_alias="SUBARAG_RETRIEVAL_REPO",
        pattern=r"^[A-Za-z0-9_-]+/[A-Za-z0-9_.-]+$",
    )
    retrieval_revision: str | None = Field(
        default=None, validation_alias="SUBARAG_RETRIEVAL_REVISION", pattern=r"^[0-9a-f]{40}$",
    )
    retrieval_path: str | None = Field(
        default=None, validation_alias="SUBARAG_RETRIEVAL_PATH", pattern=r"^retrieval/[0-9a-f]{64}$",
    )
    retrieval_cache_dir: Path = Field(
        default=BACKEND_ROOT / ".cache" / "retrieval",
        validation_alias="SUBARAG_RETRIEVAL_CACHE_DIR",
    )
    compact_manifest: Path | None = Field(default=None, validation_alias="SUBARAG_COMPACT_MANIFEST")
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
        "hf_token",
        mode="before",
    )
    @classmethod
    def empty_secrets_are_unset(cls, value: object) -> object:
        return None if value == "" else value

    @model_validator(mode="after")
    def complete_remote_data_configuration(self) -> "Settings":
        if (self.data_repo is None) != (self.data_revision is None):
            raise ValueError("Set both SUBARAG_DATA_REPO and SUBARAG_DATA_REVISION")
        if self.data_repo is not None:
            if self.data_manifest is not None:
                raise ValueError("Choose a local data manifest or a remote dataset, not both")
            if self.hf_token is None:
                raise ValueError("HF_TOKEN is required for the private dataset")
        return self

    @model_validator(mode="after")
    def complete_remote_retrieval_configuration(self) -> "Settings":
        remote = (self.retrieval_repo, self.retrieval_revision, self.retrieval_path)
        if any(value is not None for value in remote):
            if any(value is None for value in remote):
                raise ValueError("Set retrieval repository, revision, and path together")
            if any(value is not None for value in (
                self.retrieval_manifest_en, self.retrieval_manifest_hi, self.retrieval_manifest_mr,
            )):
                raise ValueError("Choose local retrieval manifests or a remote bundle, not both")
            if self.hf_token is None or self.gemini_api_key is None:
                raise ValueError("Remote RAG requires HF_TOKEN and GEMINI_API_KEY")
        return self

    @model_validator(mode="after")
    def complete_compact_configuration(self) -> "Settings":
        if self.compact_manifest is not None:
            if self.gemini_api_key is None:
                raise ValueError("Compact RAG requires GEMINI_API_KEY")
            if self.retrieval_repo is not None or any(path is not None for path in (
                self.retrieval_manifest_en, self.retrieval_manifest_hi, self.retrieval_manifest_mr,
            )):
                raise ValueError("Choose a compact bundle or full-size retrieval configuration, not both")
        return self

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
