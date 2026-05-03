"""Utilities for FFmpeg and ffprobe commands."""

from __future__ import annotations

import json
import logging
import shlex
import subprocess
from pathlib import Path

LOGGER = logging.getLogger(__name__)


def log_command(command: list[str]) -> None:
    LOGGER.info("Command: %s", shlex.join(command))


def run_command(command: list[str]) -> subprocess.CompletedProcess[str]:
    log_command(command)
    completed = subprocess.run(command, capture_output=True, text=True, check=False)
    if completed.returncode != 0:
        raise RuntimeError(
            f"Command failed with exit code {completed.returncode}: {shlex.join(command)}\n"
            f"STDOUT:\n{completed.stdout}\nSTDERR:\n{completed.stderr}"
        )
    return completed


def ffmpeg_image_sequence_to_video(
    image_pattern: Path,
    output_path: Path,
    fps: float,
    overwrite: bool,
) -> None:
    command = [
        "ffmpeg",
        "-y" if overwrite else "-n",
        "-framerate",
        str(fps),
        "-i",
        str(image_pattern),
        "-c:v",
        "libx264",
        "-pix_fmt",
        "yuv420p",
        str(output_path),
    ]
    run_command(command)


def ffmpeg_compress_video(
    input_path: Path,
    output_path: Path,
    codec: str,
    crf: int,
    preset: str,
    gop: int,
    overwrite: bool,
) -> None:
    command = [
        "ffmpeg",
        "-y" if overwrite else "-n",
        "-i",
        str(input_path),
        "-c:v",
        codec,
        "-crf",
        str(crf),
        "-preset",
        preset,
        "-an",
    ]
    if codec == "libx264":
        command += ["-g", str(gop), "-keyint_min", str(gop), "-sc_threshold", "0"]
    elif codec == "libx265":
        command += ["-x265-params", f"keyint={gop}:min-keyint={gop}:scenecut=0"]
    else:
        raise ValueError(f"Unsupported codec: {codec}")
    command.append(str(output_path))
    run_command(command)


def ffprobe_video_metadata(video_path: Path) -> dict:
    command = [
        "ffprobe",
        "-v",
        "error",
        "-print_format",
        "json",
        "-show_streams",
        "-show_format",
        str(video_path),
    ]
    completed = run_command(command)
    payload = json.loads(completed.stdout or "{}")
    video_stream = next((stream for stream in payload.get("streams", []) if stream.get("codec_type") == "video"), {})
    format_info = payload.get("format", {})
    return {
        "codec_name": video_stream.get("codec_name"),
        "width": _safe_int(video_stream.get("width")),
        "height": _safe_int(video_stream.get("height")),
        "avg_frame_rate": _parse_frame_rate(video_stream.get("avg_frame_rate")),
        "bit_rate": _safe_float(video_stream.get("bit_rate") or format_info.get("bit_rate")),
        "duration": _safe_float(video_stream.get("duration") or format_info.get("duration")),
        "nb_frames": _safe_int(video_stream.get("nb_frames")),
        "file_size_bytes": _safe_int(format_info.get("size")),
    }


def _parse_frame_rate(value: str | None) -> float | None:
    if not value or value == "0/0":
        return None
    if "/" in value:
        numerator, denominator = value.split("/", maxsplit=1)
        if float(denominator) == 0:
            return None
        return float(numerator) / float(denominator)
    return _safe_float(value)


def _safe_int(value: object) -> int | None:
    try:
        return int(float(value)) if value is not None else None
    except (TypeError, ValueError):
        return None


def _safe_float(value: object) -> float | None:
    try:
        return float(value) if value is not None else None
    except (TypeError, ValueError):
        return None
