"""YOLOv8 detection helpers."""

from __future__ import annotations

import logging
from pathlib import Path

import cv2
import pandas as pd
from tqdm import tqdm
from ultralytics import YOLO

LOGGER = logging.getLogger(__name__)


def run_detection_on_video(
    model: YOLO,
    video_path: str | Path,
    confidence_threshold: float,
    imgsz: int,
    frame_limit: int | None = None,
    every_nth_frame: int = 1,
    classes: list[int] | None = None,
) -> pd.DataFrame:
    capture = cv2.VideoCapture(str(video_path))
    if not capture.isOpened():
        raise RuntimeError(f"Unable to open video: {video_path}")

    total_frames = int(capture.get(cv2.CAP_PROP_FRAME_COUNT)) or 0
    if frame_limit is not None and frame_limit > 0:
        total_frames = min(total_frames or frame_limit, frame_limit)

    rows: list[dict] = []
    frame_number = 0
    processed_frames = 0
    with tqdm(total=total_frames if total_frames > 0 else None, desc=Path(video_path).stem, unit="frame") as progress:
        while True:
            ok, frame = capture.read()
            if not ok:
                break
            frame_number += 1
            if frame_limit is not None and frame_number > frame_limit:
                break
            progress.update(1)
            if every_nth_frame > 1 and (frame_number - 1) % every_nth_frame != 0:
                continue

            results = model.predict(
                frame,
                conf=confidence_threshold,
                imgsz=imgsz,
                device="cpu",
                verbose=False,
            )
            boxes = results[0].boxes
            det_id = 0
            if boxes is not None:
                for box in boxes:
                    cls_id = int(box.cls.item())
                    if cls_id != 0:
                        continue
                    coords = box.xyxy[0].tolist()
                    rows.append(
                        {
                            "frame": frame_number,
                            "det_id": det_id,
                            "x1": float(coords[0]),
                            "y1": float(coords[1]),
                            "x2": float(coords[2]),
                            "y2": float(coords[3]),
                            "conf": float(box.conf.item()),
                            "cls": cls_id,
                        }
                    )
                    det_id += 1
            processed_frames += 1

    capture.release()
    LOGGER.info("Processed %s frames for %s", processed_frames, video_path)
    return pd.DataFrame(rows, columns=["frame", "det_id", "x1", "y1", "x2", "y2", "conf", "cls"])
