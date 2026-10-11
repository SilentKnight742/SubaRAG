"""Production HTTP client defaults shared by transcription adapters."""

import httpx


def create_transcription_http_client() -> httpx.AsyncClient:
    """Create a client with timeouts suited to short speech-to-text requests."""

    return httpx.AsyncClient(
        timeout=httpx.Timeout(
            connect=5,
            read=60,
            write=30,
            pool=5,
        )
    )
