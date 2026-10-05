"""Strict response validation shared by transcription adapters."""

from collections.abc import Mapping
from typing import Any

import httpx

from ..errors import TranscriptionProviderError


def invalid_response(provider: str, detail: str) -> TranscriptionProviderError:
    return TranscriptionProviderError(
        provider,
        f"{provider} returned an invalid response: {detail}",
        code="invalid_response",
    )


def json_object(response: httpx.Response, *, provider: str) -> Mapping[str, Any]:
    try:
        payload: Any = response.json()
    except ValueError as error:
        raise invalid_response(provider, "the body was not valid JSON") from error

    if not isinstance(payload, Mapping):
        raise invalid_response(provider, "the JSON body was not an object")

    return payload


def required_text(payload: Mapping[str, Any], field: str, *, provider: str) -> str:
    value = payload.get(field)
    if not isinstance(value, str) or not value.strip():
        raise invalid_response(provider, f"{field!r} was missing or empty")
    return value.strip()


def optional_text(payload: Mapping[str, Any], field: str, *, provider: str) -> str | None:
    value = payload.get(field)
    if value is None:
        return None
    if not isinstance(value, str):
        raise invalid_response(provider, f"{field!r} was not text")
    return value


def optional_probability(
    payload: Mapping[str, Any],
    field: str,
    *,
    provider: str,
) -> float | None:
    value = payload.get(field)
    if value is None:
        return None
    if isinstance(value, bool) or not isinstance(value, int | float):
        raise invalid_response(provider, f"{field!r} was not numeric")

    probability = float(value)
    if not 0 <= probability <= 1:
        raise invalid_response(provider, f"{field!r} was outside zero to one")
    return probability
