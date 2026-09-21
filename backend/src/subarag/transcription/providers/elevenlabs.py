"""ElevenLabs batch speech-to-text adapter."""

import httpx

from ..models import ProviderTranscript, TranscriptionCapabilities, TranscriptionRequest
from ._http import post_multipart
from ._response import json_object, optional_probability, optional_text, required_text


class ElevenLabsTranscriptionProvider:
    name = "elevenlabs"
    capabilities = TranscriptionCapabilities(
        max_audio_seconds=10 * 60 * 60,
        supports_realtime=True,
    )
    endpoint = "https://api.elevenlabs.io/v1/speech-to-text"

    def __init__(
        self,
        *,
        api_key: str,
        client: httpx.AsyncClient,
        model: str = "scribe_v2",
    ) -> None:
        if not api_key.strip():
            raise ValueError("ElevenLabs API key is required.")
        self._api_key = api_key
        self._client = client
        self._model = model

    async def transcribe(self, request: TranscriptionRequest) -> ProviderTranscript:
        data = {
            "model_id": self._model,
            "tag_audio_events": "false",
            "timestamps_granularity": "none",
            "diarize": "false",
        }
        if request.language:
            data["language_code"] = request.language.partition("-")[0]

        audio = request.audio
        response = await post_multipart(
            self._client,
            provider=self.name,
            url=self.endpoint,
            headers={"xi-api-key": self._api_key},
            data=data,
            files={"file": (audio.filename, audio.content, audio.content_type)},
        )
        payload = json_object(response, provider=self.name)

        return ProviderTranscript(
            text=required_text(payload, "text", provider=self.name),
            language=optional_text(payload, "language_code", provider=self.name),
            language_confidence=optional_probability(
                payload,
                "language_probability",
                provider=self.name,
            ),
            provider_request_id=optional_text(
                payload,
                "transcription_id",
                provider=self.name,
            ),
        )
