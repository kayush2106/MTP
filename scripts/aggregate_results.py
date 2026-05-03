from __future__ import annotations

import argparse
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from src.aggregation import aggregate_final_results, aggregate_weighted_crf_results, load_csv_if_exists
from src.config import load_config
from src.io_utils import ensure_dir, setup_logging


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", required=True)
    args = parser.parse_args()

    setup_logging()
    config = load_config(args.config)
    ensure_dir(config.tables_dir)

    metadata_df = load_csv_if_exists(config.tables_dir / "video_metadata.csv")
    detection_df = load_csv_if_exists(config.metrics_dir / "detection_metrics.csv")
    quality_df = load_csv_if_exists(config.quality_dir / "quality_metrics.csv")
    final_df = aggregate_final_results(metadata_df, detection_df, quality_df)
    final_df.to_csv(config.tables_dir / "final_results.csv", index=False)
    crf_summary_df = aggregate_weighted_crf_results(final_df)
    crf_summary_df.to_csv(config.tables_dir / "crf_level_summary.csv", index=False)


if __name__ == "__main__":
    main()
