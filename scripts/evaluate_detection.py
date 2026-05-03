from __future__ import annotations

import argparse
import logging
import sys
from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from src.config import load_config
from src.evaluation import evaluate_video_detections
from src.io_utils import ensure_dir, setup_logging
from src.mot_utils import load_mot_ground_truth

LOGGER = logging.getLogger(__name__)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", required=True)
    args = parser.parse_args()

    setup_logging()
    config = load_config(args.config)
    per_frame_dir = ensure_dir(config.metrics_dir / "per_frame")
    ensure_dir(config.metrics_dir)

    rows: list[dict] = []
    for sequence in config.sequences:
        gt_dict = load_mot_ground_truth(config.dataset_root / sequence / "gt" / "gt.txt", frame_limit=config.frame_limit)
        detection_dir = config.detections_dir / sequence
        if not detection_dir.exists():
            LOGGER.warning("Missing detection directory for %s", sequence)
            continue
        for detection_csv in detection_dir.glob("*_detections.csv"):
            pred_df = pd.read_csv(detection_csv) if detection_csv.stat().st_size else pd.DataFrame()
            metrics = evaluate_video_detections(pred_df, gt_dict, config.iou_threshold_eval)
            video_name = detection_csv.stem.replace("_detections", "")
            is_source = video_name.endswith("_source")
            profile_name = "source" if is_source else video_name.replace(f"{sequence}_", "", 1)
            rows.append(
                {
                    "sequence": sequence,
                    "video_name": video_name,
                    "profile_name": profile_name,
                    "is_source": is_source,
                    "tp": metrics["tp"],
                    "fp": metrics["fp"],
                    "fn": metrics["fn"],
                    "precision": metrics["precision"],
                    "recall": metrics["recall"],
                    "f1": metrics["f1"],
                    "mean_matched_iou": metrics["mean_matched_iou"],
                    "frames_evaluated": metrics["frames_evaluated"],
                    "total_gt_boxes": metrics["total_gt_boxes"],
                    "total_pred_boxes": metrics["total_pred_boxes"],
                }
            )
            metrics["per_frame"].to_csv(per_frame_dir / f"{video_name}_per_frame_metrics.csv", index=False)

    pd.DataFrame(rows).to_csv(config.metrics_dir / "detection_metrics.csv", index=False)


if __name__ == "__main__":
    main()
