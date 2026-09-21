"""Shared HTTP error normalization for provider adapters."""

from collections.abc import Mapping
from typing import Any

import httpx

from ..errors import TranscriptionProviderError


def _error_details(response: httpx.Response) -> tuple[str, str]:
    fallback_code = f"http_{response.status_code}"
    fallback_message = f"Transcription provider returned HTTP {response.status_code}."

    try:
        payload: Any = response.json()
    except ValueError:
        return fallback_code, fallback_message

    if not isinstance(payload, Mapping):
        return fallback_code, fallback_message

    detail = payload.get("error", payload.get("detail", payload))
    if isinstance(detail, Mapping):
        code = detail.get("code", detail.get("status", fallback_code))
        message = detail.get("message", fallback_message)
        return str(code), str(message)

    if isinstance(detail, str) and detail.strip():
        return fallback_code, detail.strip()

    return fallback_code, fallback_message


async def post_multipart(
    client: httpx.AsyncClient,
    *,
    provider: str,
    url: str,
    headers: Mapping[str, str],
    data: Mapping[str, str],
    files: Mapping[str, tuple[str, bytes, str]],
) -> httpx.Response:
    try:
        response = await client.post(url, headers=headers, data=data, files=files)
    except httpx.TimeoutException as error:
        raise TranscriptionProviderError(
            provider,
            f"{provider} transcription timed out.",
            code="timeout",
            retryable=True,
        ) from error
    except httpx.RequestError as error:
        raise TranscriptionProviderError(
            provider,
            f"{provider} transcription could not reach the provider.",
            code="network_error",
            retryable=True,
        ) from error

    if response.is_error:
        code, message = _error_details(response)
        raise TranscriptionProviderError(
            provider,
            message,
            code=code,
            retryable=response.status_code == 429 or response.status_code >= 500,
        )

    return response
