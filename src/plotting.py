"""Plot creation helpers."""

from __future__ import annotations

from pathlib import Path

import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns


def save_required_plots(
    results_df: pd.DataFrame,
    crf_summary_df: pd.DataFrame,
    output_dir: str | Path,
) -> None:
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    sns.set_theme(style="whitegrid")

    if crf_summary_df.empty:
        return

    crf_df = crf_summary_df.dropna(subset=["crf"]).copy()
    if crf_df.empty:
        return

    crf_df = crf_df.sort_values("crf").reset_index(drop=True)
    _save_precision_recall_by_crf(crf_df, output_dir / "precision_recall_by_crf.png")
    _save_metric_vs_crf_plot(crf_df, "weighted_f1", "Weighted F1", output_dir / "weighted_f1_vs_crf.png")
    _save_metric_vs_crf_plot(crf_df, "weighted_psnr", "Weighted PSNR", output_dir / "weighted_psnr_vs_crf.png")
    _save_metric_vs_crf_plot(crf_df, "weighted_ssim", "Weighted SSIM", output_dir / "weighted_ssim_vs_crf.png")
    _save_metric_vs_crf_plot(crf_df, "weighted_bitrate", "Weighted Bitrate", output_dir / "weighted_bitrate_vs_crf.png")
    _save_weighted_metric_pair_plot(
        crf_df,
        "weighted_psnr",
        "weighted_f1",
        "Weighted PSNR",
        "Weighted F1",
        output_dir / "weighted_f1_vs_weighted_psnr.png",
    )
    _save_weighted_metric_pair_plot(
        crf_df,
        "weighted_ssim",
        "weighted_f1",
        "Weighted SSIM",
        "Weighted F1",
        output_dir / "weighted_f1_vs_weighted_ssim.png",
    )


    if not results_df.empty:
        _save_metric_vs_metric_plot(results_df, "bitrate", "f1", "Bitrate", "F1", output_dir / "f1_vs_bitrate_raw.png")
        _save_metric_vs_metric_plot(results_df, "psnr_avg", "f1", "PSNR", "F1", output_dir / "f1_vs_psnr_raw.png")
        _save_metric_vs_metric_plot(results_df, "ssim_avg", "f1", "SSIM", "F1", output_dir / "f1_vs_ssim_raw.png")


def _save_precision_recall_by_crf(crf_df: pd.DataFrame, output_path: Path) -> None:
    plt.figure(figsize=(8, 6))
    palette = sns.color_palette("deep", n_colors=len(crf_df))

    for idx, row in crf_df.iterrows():
        precision = row["weighted_precision"]
        recall = row["weighted_recall"]
        f1 = row["weighted_f1"]
        if pd.isna(precision) or pd.isna(recall):
            continue
        plt.plot(
            [0.0, recall],
            [1.0, precision],
            marker="o",
            linewidth=2,
            color=palette[idx],
            label=f"CRF level = {int(row['crf'])}, f1 = {f1:.2f}" if pd.notna(f1) else f"CRF level = {int(row['crf'])}",
        )

    plt.title("Class: Person")
    plt.xlabel("Recall")
    plt.ylabel("Precision")
    plt.xlim(0.0, 1.0)
    plt.ylim(0.0, 1.0)
    plt.legend(loc="best", frameon=True)
    plt.tight_layout()
    plt.savefig(output_path, dpi=150)
    plt.close()


def _save_metric_vs_crf_plot(crf_df: pd.DataFrame, metric_col: str, y_label: str, output_path: Path) -> None:
    plot_df = crf_df.dropna(subset=["crf", metric_col]).copy()
    if plot_df.empty:
        return
    plt.figure(figsize=(8, 6))
    sns.lineplot(data=plot_df, x="crf", y=metric_col, marker="o", linewidth=2)
    plt.xlabel("CRF Level")
    plt.ylabel(y_label)
    plt.tight_layout()
    plt.savefig(output_path, dpi=150)
    plt.close()


def _save_metric_vs_metric_plot(
    results_df: pd.DataFrame,
    x_col: str,
    y_col: str,
    x_label: str,
    y_label: str,
    output_path: Path,
) -> None:
    plot_df = results_df.dropna(subset=[x_col, y_col, "crf"]).copy()
    if plot_df.empty:
        return
    plt.figure(figsize=(8, 6))
    sns.scatterplot(data=plot_df, x=x_col, y=y_col, hue="crf", palette="viridis")
    plt.xlabel(x_label)
    plt.ylabel(y_label)
    plt.tight_layout()
    plt.savefig(output_path, dpi=150)
    plt.close()


def _save_weighted_metric_pair_plot(
    crf_df: pd.DataFrame,
    x_col: str,
    y_col: str,
    x_label: str,
    y_label: str,
    output_path: Path,
) -> None:
    plot_df = crf_df.dropna(subset=[x_col, y_col, "crf"]).copy()
    if plot_df.empty:
        return
    plt.figure(figsize=(8, 6))
    ax = sns.lineplot(data=plot_df, x=x_col, y=y_col, marker="o", linewidth=2)
    for _, row in plot_df.iterrows():
        ax.annotate(
            f"CRF {int(row['crf'])}",
            (row[x_col], row[y_col]),
            textcoords="offset points",
            xytext=(6, 6),
            ha="left",
            fontsize=8,
        )
    plt.xlabel(x_label)
    plt.ylabel(y_label)
    plt.tight_layout()
    plt.savefig(output_path, dpi=150)
    plt.close()
