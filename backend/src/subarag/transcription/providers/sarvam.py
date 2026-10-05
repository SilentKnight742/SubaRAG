"""Sarvam synchronous speech-to-text adapter."""

from collections.abc import Mapping
from typing import Any

import httpx

from ..errors import TranscriptionProviderError
from ..models import (
    ProviderTranscript,
    TranscriptSegment,
    TranscriptionCapabilities,
    TranscriptionRequest,
)
from ._http import post_multipart
from ._response import (
    invalid_response,
    json_object,
    optional_probability,
    optional_text,
    required_text,
)


_SARVAM_LANGUAGE_CODES = {
    code: f"{code}-IN"
    for code in (
        "as",
        "bn",
        "brx",
        "doi",
        "en",
        "gu",
        "hi",
        "kn",
        "kok",
        "ks",
        "mai",
        "ml",
        "mni",
        "mr",
        "ne",
        "od",
        "pa",
        "sa",
        "sat",
        "sd",
        "ta",
        "te",
        "ur",
    )
}
_SARVAM_LANGUAGE_CODES["or"] = "od-IN"


def _sarvam_language_code(language: str) -> str:
    base_language = language.lower().partition("-")[0]
    if base_language == "unknown":
        return "unknown"

    try:
        return _SARVAM_LANGUAGE_CODES[base_language]
    except KeyError as error:
        raise TranscriptionProviderError(
            "sarvam",
            f"Sarvam does not support the requested language: {language}",
            code="unsupported_language",
        ) from error


def _segments(payload: Any) -> tuple[TranscriptSegment, ...]:
    if payload is None:
        return ()
    if not isinstance(payload, Mapping):
        raise invalid_response("sarvam", "'timestamps' was not an object")

    texts = payload.get("words")
    starts = payload.get("start_time_seconds")
    ends = payload.get("end_time_seconds")
    if not isinstance(texts, list) or not isinstance(starts, list) or not isinstance(ends, list):
        raise invalid_response("sarvam", "timestamp arrays were missing")
    if not (len(texts) == len(starts) == len(ends)):
        raise invalid_response("sarvam", "timestamp arrays had different lengths")

    segments: list[TranscriptSegment] = []
    try:
        for text, start, end in zip(texts, starts, ends, strict=True):
            if (
                not isinstance(text, str)
                or isinstance(start, bool)
                or not isinstance(start, int | float)
                or isinstance(end, bool)
                or not isinstance(end, int | float)
            ):
                raise invalid_response("sarvam", "a timestamp entry had an invalid type")
            segments.append(
                TranscriptSegment(
                    text=text,
                    start_seconds=float(start),
                    end_seconds=float(end),
                )
            )
    except ValueError as error:
        raise invalid_response("sarvam", "a timestamp range was invalid") from error

    return tuple(segments)


class SarvamTranscriptionProvider:
    name = "sarvam"
    capabilities = TranscriptionCapabilities(max_audio_seconds=30, supports_realtime=True)
    endpoint = "https://api.sarvam.ai/speech-to-text"

    def __init__(
        self,
        *,
        api_key: str,
        client: httpx.AsyncClient,
        model: str = "saaras:v3",
    ) -> None:
        if not api_key.strip():
            raise ValueError("Sarvam API key is required.")
        self._api_key = api_key
        self._client = client
        self._model = model

    async def transcribe(self, request: TranscriptionRequest) -> ProviderTranscript:
        data = {
            "model": self._model,
            "mode": "transcribe",
            "with_timestamps": "false",
        }
        if request.language:
            data["language_code"] = _sarvam_language_code(request.language)

        audio = request.audio
        response = await post_multipart(
            self._client,
            provider=self.name,
            url=self.endpoint,
            headers={"api-subscription-key": self._api_key},
            data=data,
            files={"file": (audio.filename, audio.content, audio.content_type)},
        )
        payload = json_object(response, provider=self.name)

        return ProviderTranscript(
            text=required_text(payload, "transcript", provider=self.name),
            language=optional_text(payload, "language_code", provider=self.name),
            language_confidence=optional_probability(
                payload,
                "language_probability",
                provider=self.name,
            ),
            segments=_segments(payload.get("timestamps")),
            provider_request_id=optional_text(payload, "request_id", provider=self.name),
        )
