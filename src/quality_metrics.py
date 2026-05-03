"""FFmpeg-based PSNR and SSIM helpers."""

from __future__ import annotations

import logging
import re
import subprocess
from pathlib import Path

from .ffmpeg_utils import log_command

LOGGER = logging.getLogger(__name__)


def compute_psnr_ssim(source_video: str | Path, compressed_video: str | Path) -> dict[str, float]:
    return {
        "psnr_avg": _run_metric(source_video, compressed_video, "psnr", r"average:([0-9eE+.\-]+)"),
        "ssim_avg": _run_metric(source_video, compressed_video, "ssim", r"All:([0-9eE+.\-]+)"),
    }


def _run_metric(source_video: str | Path, compressed_video: str | Path, filter_name: str, pattern: str) -> float:
    command = [
        "ffmpeg",
        "-i",
        str(source_video),
        "-i",
        str(compressed_video),
        "-lavfi",
        filter_name,
        "-f",
        "null",
        "-",
    ]
    log_command(command)
    completed = subprocess.run(command, capture_output=True, text=True, check=False)
    text = f"{completed.stdout}\n{completed.stderr}"
    match = re.search(pattern, text)
    if match:
        return float(match.group(1))
    LOGGER.warning("Unable to parse %s from FFmpeg output for %s", filter_name, compressed_video)
    return float("nan")
