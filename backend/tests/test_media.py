from io import BytesIO

import av
import pytest

from subarag.api.media import MediaValidationError, inspect_audio


def make_webm(*, frame_count: int = 10) -> bytes:
    output_buffer = BytesIO()
    with av.open(output_buffer, mode="w", format="webm") as container:
        stream = container.add_stream("libopus", rate=48_000)
        stream.layout = "mono"

        for _ in range(frame_count):
            frame = av.AudioFrame(format="s16", layout="mono", samples=960)
            frame.sample_rate = 48_000
            frame.planes[0].update(bytes(960 * 2))
            for packet in stream.encode(frame):
                container.mux(packet)

        for packet in stream.encode(None):
            container.mux(packet)

    return output_buffer.getvalue()


def make_mp4(*, frame_count: int = 10) -> bytes:
    output_buffer = BytesIO()
    with av.open(output_buffer, mode="w", format="mp4") as container:
        stream = container.add_stream("aac", rate=48_000)
        stream.layout = "mono"

        for _ in range(frame_count):
            frame = av.AudioFrame(format="fltp", layout="mono", samples=1024)
            frame.sample_rate = 48_000
            frame.planes[0].update(bytes(frame.planes[0].buffer_size))
            for packet in stream.encode(frame):
                container.mux(packet)

        for packet in stream.encode(None):
            container.mux(packet)

    return output_buffer.getvalue()


def test_inspector_measures_real_webm_audio() -> None:
    result = inspect_audio(
        make_webm(frame_count=10),
        content_type="audio/webm; codecs=opus",
        max_duration_seconds=30,
    )

    assert result.content_type == "audio/webm"
    assert result.codec == "opus"
    assert result.duration_seconds == pytest.approx(0.2)


def test_inspector_measures_real_mp4_audio() -> None:
    result = inspect_audio(
        make_mp4(frame_count=10),
        content_type="audio/mp4",
        max_duration_seconds=30,
    )

    assert result.content_type == "audio/mp4"
    assert result.codec == "aac"
    assert result.duration_seconds == pytest.approx(0.2346666667)


def test_inspector_stops_decoding_when_audio_exceeds_limit() -> None:
    with pytest.raises(MediaValidationError) as exc_info:
        inspect_audio(
            make_webm(frame_count=10),
            content_type="audio/webm",
            max_duration_seconds=0.1,
        )

    assert exc_info.value.code == "audio_too_long"
    assert exc_info.value.status_code == 413


@pytest.mark.parametrize(
    ("content", "content_type", "expected_code", "expected_status"),
    [
        (b"not-media", "audio/webm", "invalid_audio", 422),
        (make_webm(), "audio/mp4", "media_type_mismatch", 415),
        (make_webm(), "audio/ogg", "unsupported_media_type", 415),
        (b"", "audio/webm", "empty_audio", 422),
    ],
)
def test_inspector_rejects_invalid_media(
    content: bytes,
    content_type: str,
    expected_code: str,
    expected_status: int,
) -> None:
    with pytest.raises(MediaValidationError) as exc_info:
        inspect_audio(
            content,
            content_type=content_type,
            max_duration_seconds=30,
        )

    assert exc_info.value.code == expected_code
    assert exc_info.value.status_code == expected_status
