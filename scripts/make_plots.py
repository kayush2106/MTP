from __future__ import annotations

import argparse
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from src.aggregation import load_csv_if_exists
from src.config import load_config
from src.io_utils import ensure_dir, setup_logging
from src.plotting import save_required_plots


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", required=True)
    args = parser.parse_args()

    setup_logging()
    config = load_config(args.config)
    ensure_dir(config.plots_dir)
    results_df = load_csv_if_exists(config.tables_dir / "final_results.csv")
    crf_summary_df = load_csv_if_exists(config.tables_dir / "crf_level_summary.csv")
    save_required_plots(results_df, crf_summary_df, config.plots_dir)


if __name__ == "__main__":
    main()
