"""Contracts shared by the application and its future retrieval engine."""

from typing import Literal, Self

from pydantic import BaseModel, ConfigDict, Field, field_validator, model_validator


class RagQuery(BaseModel):
    model_config = ConfigDict(extra="forbid", str_strip_whitespace=True)

    question: str = Field(min_length=1, max_length=2000)
    language: str | None = Field(default=None, max_length=32)

    @field_validator("language")
    @classmethod
    def normalize_language(cls, value: str | None) -> str | None:
        if value is None or value.lower() in {"", "auto"}:
            return None
        return value.lower()


class Citation(BaseModel):
    document_id: str = Field(min_length=1)
    text: str = Field(min_length=1)
    source: str | None = None


class RagResult(BaseModel):
    status: Literal["unavailable", "answered", "no_answer"]
    answer: str | None = None
    citations: list[Citation] = Field(default_factory=list)
    message: str | None = None

    @model_validator(mode="after")
    def validate_answer(self) -> Self:
        if self.status == "answered":
            if not self.answer or not self.answer.strip() or not self.citations:
                raise ValueError("An answered result requires text and supporting citations.")
        elif self.answer is not None or self.citations:
            raise ValueError("Unavailable and no-answer results cannot invent an answer or citations.")
        return self
