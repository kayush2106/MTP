from __future__ import annotations

import argparse
import logging
import sys
from pathlib import Path

from ultralytics import YOLO

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from src.config import load_config
from src.detection import run_detection_on_video
from src.io_utils import ensure_dir, setup_logging, should_skip_output

LOGGER = logging.getLogger(__name__)


def iter_videos(config) -> list[tuple[str, Path]]:
    items: list[tuple[str, Path]] = []
    for sequence in config.sequences:
        items.append((sequence, config.source_video_dir / f"{sequence}_source.mp4"))
        sequence_dir = config.compressed_video_dir / sequence
        for profile in config.compression_profiles:
            items.append((sequence, sequence_dir / f"{sequence}_{profile.name}.mp4"))
    return items


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", required=True)
    parser.add_argument("--frame-limit", type=int, default=None)
    parser.add_argument("--every-nth-frame", type=int, default=1)
    parser.add_argument("--overwrite", action="store_true")
    args = parser.parse_args()

    setup_logging()
    config = load_config(args.config)
    model = YOLO(config.model_name)
    frame_limit = args.frame_limit if args.frame_limit is not None else config.frame_limit

    for sequence, video_path in iter_videos(config):
        if not video_path.exists():
            LOGGER.warning("Skipping missing video %s", video_path)
            continue
        output_dir = ensure_dir(config.detections_dir / sequence)
        output_path = output_dir / f"{video_path.stem}_detections.csv"
        if should_skip_output(output_path, args.overwrite):
            LOGGER.info("Skipping existing detections %s", output_path)
            continue
        detections_df = run_detection_on_video(
            model=model,
            video_path=video_path,
            confidence_threshold=config.confidence_threshold,
            imgsz=config.imgsz,
            frame_limit=frame_limit,
            every_nth_frame=args.every_nth_frame,
            classes=[0],
        )
        detections_df.to_csv(output_path, index=False)


if __name__ == "__main__":
    main()
