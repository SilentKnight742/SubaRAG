"""Server-side validation and duration measurement for browser audio."""

from dataclasses import dataclass
from io import BytesIO
from math import isfinite

import av


ALLOWED_MEDIA_TYPES = frozenset({"audio/mp4", "audio/webm"})


class MediaValidationError(ValueError):
    def __init__(self, code: str, message: str, *, status_code: int = 422) -> None:
        super().__init__(message)
        self.code = code
        self.status_code = status_code


@dataclass(frozen=True, slots=True)
class InspectedAudio:
    duration_seconds: float
    content_type: str
    codec: str


def normalize_media_type(content_type: str | None) -> str:
    if content_type is None:
        return ""
    return content_type.partition(";")[0].strip().lower()


def _validate_container(content_type: str, format_names: set[str], codec: str) -> None:
    if content_type == "audio/webm":
        if not format_names.intersection({"matroska", "webm"}) or codec not in {
            "opus",
            "vorbis",
        }:
            raise MediaValidationError(
                "media_type_mismatch",
                "The uploaded file is not valid WebM audio.",
                status_code=415,
            )
        return

    if content_type == "audio/mp4":
        if not format_names.intersection({"m4a", "mov", "mp4"}) or codec != "aac":
            raise MediaValidationError(
                "media_type_mismatch",
                "The uploaded file is not valid MP4 audio.",
                status_code=415,
            )


def inspect_audio(
    content: bytes,
    *,
    content_type: str,
    max_duration_seconds: float,
) -> InspectedAudio:
    """Decode one browser recording and measure duration from audio frames."""

    normalized_type = normalize_media_type(content_type)
    if normalized_type not in ALLOWED_MEDIA_TYPES:
        raise MediaValidationError(
            "unsupported_media_type",
            "Only WebM or MP4 audio recordings are supported.",
            status_code=415,
        )
    if not content:
        raise MediaValidationError("empty_audio", "The audio recording is empty.")

    try:
        with av.open(BytesIO(content), mode="r") as container:
            if len(container.streams.video) > 0:
                raise MediaValidationError(
                    "video_not_allowed",
                    "Video uploads are not supported.",
                    status_code=415,
                )
            if len(container.streams.audio) != 1:
                raise MediaValidationError(
                    "invalid_audio_stream",
                    "The recording must contain exactly one audio stream.",
                )

            stream = container.streams.audio[0]
            codec = stream.codec_context.name
            format_names = set(container.format.name.lower().split(","))
            _validate_container(normalized_type, format_names, codec)

            sample_duration = 0.0
            measured_duration = 0.0
            earliest_start: float | None = None
            latest_end = 0.0
            decoded_frames = 0

            for frame in container.decode(audio=0):
                if frame.sample_rate is None or frame.sample_rate <= 0 or frame.samples <= 0:
                    continue

                decoded_frames += 1
                frame_duration = frame.samples / frame.sample_rate
                sample_duration += frame_duration

                if frame.pts is not None and frame.time_base is not None:
                    frame_start = float(frame.pts * frame.time_base)
                    if isfinite(frame_start):
                        earliest_start = (
                            frame_start
                            if earliest_start is None
                            else min(earliest_start, frame_start)
                        )
                        latest_end = max(latest_end, frame_start + frame_duration)

                measured_duration = sample_duration
                if earliest_start is not None:
                    measured_duration = max(
                        measured_duration,
                        latest_end - earliest_start,
                    )
                if measured_duration > max_duration_seconds:
                    raise MediaValidationError(
                        "audio_too_long",
                        (
                            f"Audio exceeds the SubaRAG limit of "
                            f"{max_duration_seconds:.1f} seconds."
                        ),
                        status_code=413,
                    )

    except MediaValidationError:
        raise
    except (av.FFmpegError, EOFError, ValueError) as error:
        raise MediaValidationError(
            "invalid_audio",
            "The uploaded recording could not be decoded.",
        ) from error

    if decoded_frames == 0 or not isfinite(measured_duration) or measured_duration <= 0:
        raise MediaValidationError(
            "invalid_audio",
            "The recording did not contain usable audio.",
        )

    return InspectedAudio(
        duration_seconds=measured_duration,
        content_type=normalized_type,
        codec=codec,
    )
