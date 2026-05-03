"""Result aggregation helpers."""

from __future__ import annotations

from pathlib import Path

import pandas as pd


def load_csv_if_exists(path: str | Path) -> pd.DataFrame:
    csv_path = Path(path)
    if not csv_path.exists():
        return pd.DataFrame()
    return pd.read_csv(csv_path)


def aggregate_final_results(
    metadata_df: pd.DataFrame,
    detection_df: pd.DataFrame,
    quality_df: pd.DataFrame,
) -> pd.DataFrame:
    if metadata_df.empty:
        return pd.DataFrame(
            columns=[
                "sequence",
                "profile_name",
                "codec",
                "crf",
                "preset",
                "gop",
                "bitrate",
                "file_size_bytes",
                "psnr_avg",
                "ssim_avg",
                "precision",
                "recall",
                "f1",
                "mean_matched_iou",
                "frames_evaluated",
            ]
        )

    if detection_df.empty:
        detection_df = pd.DataFrame(columns=["sequence", "profile_name", "is_source"])
    if quality_df.empty:
        quality_df = pd.DataFrame(columns=["sequence", "profile_name", "psnr_avg", "ssim_avg"])

    compressed_metadata = metadata_df[metadata_df["source_or_compressed"] == "compressed"].copy()
    source_metadata = metadata_df[metadata_df["source_or_compressed"] == "source"].copy()

    result = compressed_metadata.merge(
        detection_df,
        on=["sequence", "profile_name"],
        how="left",
    ).merge(
        quality_df,
        on=["sequence", "profile_name"],
        how="left",
    )

    if not source_metadata.empty:
        source_rows = source_metadata.merge(
            detection_df[detection_df["is_source"] == True],
            on=["sequence", "profile_name"],
            how="left",
        )
        source_rows["psnr_avg"] = pd.NA
        source_rows["ssim_avg"] = pd.NA
        result = pd.concat([result, source_rows], ignore_index=True, sort=False)

    ordered_columns = [
        "sequence",
        "profile_name",
        "codec",
        "crf",
        "preset",
        "gop",
        "bitrate",
        "file_size_bytes",
        "psnr_avg",
        "ssim_avg",
        "precision",
        "recall",
        "f1",
        "mean_matched_iou",
        "frames_evaluated",
    ]
    for column in ordered_columns:
        if column not in result.columns:
            result[column] = pd.NA
    return result[ordered_columns].sort_values(["sequence", "profile_name"], na_position="last").reset_index(drop=True)

def aggregate_weighted_crf_results(final_df: pd.DataFrame) -> pd.DataFrame:
    """Aggregate weighted averages for each CRF level."""
    columns = [
        "crf",
        "num_rows",
        "weighted_precision",
        "weighted_recall",
        "weighted_f1",
        "weighted_psnr",
        "weighted_ssim",
        "weighted_bitrate",
        "total_weight",
    ]
    if final_df.empty or "crf" not in final_df.columns:
        return pd.DataFrame(columns=columns)

    df = final_df.copy()
    df = df[df["crf"].notna()].copy()
    if df.empty:
        return pd.DataFrame(columns=columns)

    df["weight"] = pd.to_numeric(df.get("frames_evaluated"), errors="coerce").fillna(1.0)
    df.loc[df["weight"] <= 0, "weight"] = 1.0

    metric_columns = {
        "weighted_precision": "precision",
        "weighted_recall": "recall",
        "weighted_f1": "f1",
        "weighted_psnr": "psnr_avg",
        "weighted_ssim": "ssim_avg",
        "weighted_bitrate": "bitrate",
    }

    rows: list[dict] = []
    for crf_value, group in df.groupby("crf", dropna=True):
        row = {
            "crf": int(float(crf_value)),
            "num_rows": int(len(group)),
            "total_weight": float(group["weight"].sum()),
        }
        for output_column, source_column in metric_columns.items():
            row[output_column] = _weighted_average(group, source_column, "weight")
        rows.append(row)

    return pd.DataFrame(rows, columns=columns).sort_values("crf").reset_index(drop=True)


def _weighted_average(df: pd.DataFrame, value_column: str, weight_column: str) -> float | object:
    values = pd.to_numeric(df.get(value_column), errors="coerce")
    weights = pd.to_numeric(df.get(weight_column), errors="coerce")
    valid = values.notna() & weights.notna() & (weights > 0)
    if not valid.any():
        return pd.NA
    return float((values[valid] * weights[valid]).sum() / weights[valid].sum())
