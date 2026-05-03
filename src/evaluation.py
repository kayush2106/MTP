"""Detection evaluation helpers."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd


def compute_iou(box_a: dict, box_b: dict) -> float:
    x_left = max(box_a["x1"], box_b["x1"])
    y_top = max(box_a["y1"], box_b["y1"])
    x_right = min(box_a["x2"], box_b["x2"])
    y_bottom = min(box_a["y2"], box_b["y2"])

    if x_right <= x_left or y_bottom <= y_top:
        return 0.0

    intersection = (x_right - x_left) * (y_bottom - y_top)
    area_a = max(0.0, box_a["x2"] - box_a["x1"]) * max(0.0, box_a["y2"] - box_a["y1"])
    area_b = max(0.0, box_b["x2"] - box_b["x1"]) * max(0.0, box_b["y2"] - box_b["y1"])
    union = area_a + area_b - intersection
    return float(intersection / union) if union > 0 else 0.0


def greedy_match_predictions_to_gt(pred_boxes: list[dict], gt_boxes: list[dict], iou_threshold: float) -> dict:
    candidates: list[tuple[float, int, int]] = []
    for pred_index, pred in enumerate(pred_boxes):
        for gt_index, gt in enumerate(gt_boxes):
            iou = compute_iou(pred, gt)
            if iou >= iou_threshold:
                candidates.append((iou, pred_index, gt_index))
    candidates.sort(reverse=True, key=lambda item: item[0])

    matched_preds: set[int] = set()
    matched_gt: set[int] = set()
    matches: list[dict] = []

    for iou, pred_index, gt_index in candidates:
        if pred_index in matched_preds or gt_index in matched_gt:
            continue
        matched_preds.add(pred_index)
        matched_gt.add(gt_index)
        matches.append({"pred_index": pred_index, "gt_index": gt_index, "iou": iou})

    return {
        "matches": matches,
        "matched_pred_indices": matched_preds,
        "matched_gt_indices": matched_gt,
    }


def evaluate_video_detections(pred_df: pd.DataFrame, gt_dict: dict[int, list[dict]], iou_threshold: float) -> dict:
    if pred_df.empty:
        pred_df = pd.DataFrame(columns=["frame", "x1", "y1", "x2", "y2"])

    frame_numbers = sorted(set(gt_dict.keys()) | set(pred_df["frame"].astype(int).tolist()))
    per_frame_rows: list[dict] = []
    total_tp = total_fp = total_fn = 0
    matched_ious: list[float] = []
    total_gt_boxes = sum(len(boxes) for boxes in gt_dict.values())
    total_pred_boxes = int(len(pred_df))

    for frame in frame_numbers:
        frame_pred_df = pred_df[pred_df["frame"] == frame]
        pred_boxes = frame_pred_df[["x1", "y1", "x2", "y2"]].to_dict(orient="records")
        gt_boxes = gt_dict.get(frame, [])
        match_info = greedy_match_predictions_to_gt(pred_boxes, gt_boxes, iou_threshold)
        tp = len(match_info["matches"])
        fp = len(pred_boxes) - tp
        fn = len(gt_boxes) - tp
        total_tp += tp
        total_fp += fp
        total_fn += fn
        matched_ious.extend(match["iou"] for match in match_info["matches"])
        per_frame_rows.append(
            {
                "frame": frame,
                "tp": tp,
                "fp": fp,
                "fn": fn,
                "pred_boxes": len(pred_boxes),
                "gt_boxes": len(gt_boxes),
                "mean_matched_iou": float(np.mean([m["iou"] for m in match_info["matches"]])) if match_info["matches"] else np.nan,
            }
        )

    precision = total_tp / (total_tp + total_fp) if (total_tp + total_fp) else 0.0
    recall = total_tp / (total_tp + total_fn) if (total_tp + total_fn) else 0.0
    f1 = 2 * precision * recall / (precision + recall) if (precision + recall) else 0.0
    return {
        "tp": total_tp,
        "fp": total_fp,
        "fn": total_fn,
        "precision": precision,
        "recall": recall,
        "f1": f1,
        "mean_matched_iou": float(np.mean(matched_ious)) if matched_ious else np.nan,
        "frames_evaluated": len(frame_numbers),
        "total_gt_boxes": total_gt_boxes,
        "total_pred_boxes": total_pred_boxes,
        "per_frame": pd.DataFrame(per_frame_rows),
    }
