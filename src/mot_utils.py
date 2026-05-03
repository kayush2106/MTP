"""Helpers for reading MOT17 metadata and ground truth."""

from __future__ import annotations

import configparser
import logging
from pathlib import Path

LOGGER = logging.getLogger(__name__)


def read_seqinfo(seqinfo_path: str | Path) -> dict:
    path = Path(seqinfo_path)
    parser = configparser.ConfigParser()
    parser.read(path, encoding="utf-8")
    if "Sequence" not in parser:
        raise ValueError(f"Missing [Sequence] section in {path}")
    section = parser["Sequence"]
    return {
        "name": section.get("name", path.parent.name),
        "imDir": section.get("imDir", "img1"),
        "frameRate": section.getint("frameRate", fallback=30),
        "seqLength": section.getint("seqLength", fallback=0),
        "imWidth": section.getint("imWidth", fallback=0),
        "imHeight": section.getint("imHeight", fallback=0),
        "imExt": section.get("imExt", ".jpg"),
    }


def load_mot_ground_truth(gt_path: str | Path, frame_limit: int | None = None) -> dict[int, list[dict]]:
    path = Path(gt_path)
    output: dict[int, list[dict]] = {}
    with path.open("r", encoding="utf-8") as handle:
        for line_number, line in enumerate(handle, start=1):
            raw = line.strip()
            if not raw:
                continue
            parts = raw.split(",")
            if len(parts) < 6:
                LOGGER.warning("Skipping malformed GT row %s in %s", line_number, path)
                continue
            try:
                frame = int(float(parts[0]))
                obj_id = int(float(parts[1]))
                x = float(parts[2])
                y = float(parts[3])
                w = float(parts[4])
                h = float(parts[5])
                conf = float(parts[6]) if len(parts) > 6 else 1.0
                class_id = int(float(parts[7])) if len(parts) > 7 else 1
                visibility = float(parts[8]) if len(parts) > 8 else 1.0
            except (TypeError, ValueError):
                LOGGER.warning("Skipping invalid GT row %s in %s", line_number, path)
                continue

            if frame_limit is not None and frame > frame_limit:
                continue
            if conf <= 0:
                continue
            if len(parts) > 7 and class_id != 1:
                continue

            output.setdefault(frame, []).append(
                {
                    "id": obj_id,
                    "x1": x,
                    "y1": y,
                    "x2": x + w,
                    "y2": y + h,
                    "conf": conf,
                    "class": class_id,
                    "visibility": visibility,
                }
            )
    return output
