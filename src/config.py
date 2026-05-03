"""Configuration models and loading helpers."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any

import yaml


@dataclass(slots=True)
class CompressionProfile:
    name: str
    codec: str
    crf: int
    preset: str
    gop: int


@dataclass(slots=True)
class AppConfig:
    dataset_root: Path
    source_video_dir: Path
    compressed_video_dir: Path
    detections_dir: Path
    metrics_dir: Path
    quality_dir: Path
    tables_dir: Path
    plots_dir: Path
    sequences: list[str]
    frame_limit: int | None
    fps_override: float | None
    model_name: str
    confidence_threshold: float
    iou_threshold_eval: float
    imgsz: int
    compression_profiles: list[CompressionProfile]


def _to_path(value: str | Path) -> Path:
    return Path(value)


def load_config(config_path: str | Path) -> AppConfig:
    path = Path(config_path)
    with path.open("r", encoding="utf-8") as handle:
        raw: dict[str, Any] = yaml.safe_load(handle) or {}

    profiles = [CompressionProfile(**profile) for profile in raw.get("compression_profiles", [])]

    return AppConfig(
        dataset_root=_to_path(raw["dataset_root"]),
        source_video_dir=_to_path(raw["source_video_dir"]),
        compressed_video_dir=_to_path(raw["compressed_video_dir"]),
        detections_dir=_to_path(raw["detections_dir"]),
        metrics_dir=_to_path(raw["metrics_dir"]),
        quality_dir=_to_path(raw["quality_dir"]),
        tables_dir=_to_path(raw["tables_dir"]),
        plots_dir=_to_path(raw["plots_dir"]),
        sequences=list(raw.get("sequences", [])),
        frame_limit=raw.get("frame_limit"),
        fps_override=raw.get("fps_override"),
        model_name=str(raw.get("model_name", "yolov8n.pt")),
        confidence_threshold=float(raw.get("confidence_threshold", 0.25)),
        iou_threshold_eval=float(raw.get("iou_threshold_eval", 0.5)),
        imgsz=int(raw.get("imgsz", 640)),
        compression_profiles=profiles,
    )
