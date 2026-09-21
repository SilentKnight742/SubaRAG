# SubaRAG transcription service

This package defines the provider-neutral boundary between captured browser audio and speech-to-text vendors.

The first implementation uses completed WebM recordings because both providers accept that browser-native format:

- [Sarvam Speech-to-Text](https://docs.sarvam.ai/api-reference/speech-to-text/transcribe) accepts synchronous recordings up to 30 seconds.
- [ElevenLabs Speech-to-Text](https://elevenlabs.io/docs/api-reference/speech-to-text/convert) accepts the interface's full 60-second safety limit.

`TranscriptionService` validates provider capabilities before upload, routes to an explicitly selected provider or the configured default, and returns one normalized result with timing data for later A/B evaluation. Provider credentials will remain server-side.

Realtime partial transcripts are intentionally deferred. Both vendors' realtime APIs require PCM-style streams rather than the current MediaRecorder WebM output, so that path needs a separate browser audio-worklet design.

## Run tests

```bash
cd backend
PYTHONDONTWRITEBYTECODE=1 python3 -m pytest
```
