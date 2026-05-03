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
from src.io_utils import ensure_dir, setup_logging
from src.quality_metrics import compute_psnr_ssim

LOGGER = logging.getLogger(__name__)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", required=True)
    args = parser.parse_args()

    setup_logging()
    config = load_config(args.config)
    ensure_dir(config.quality_dir)

    rows: list[dict] = []
    for sequence in config.sequences:
        source_video = config.source_video_dir / f"{sequence}_source.mp4"
        for profile in config.compression_profiles:
            compressed_video = config.compressed_video_dir / sequence / f"{sequence}_{profile.name}.mp4"
            if not source_video.exists() or not compressed_video.exists():
                LOGGER.warning("Skipping quality metrics for missing pair %s / %s", source_video, compressed_video)
                continue
            metrics = compute_psnr_ssim(source_video, compressed_video)
            rows.append(
                {
                    "sequence": sequence,
                    "profile_name": profile.name,
                    "compressed_video": str(compressed_video),
                    "source_video": str(source_video),
                    "psnr_avg": metrics["psnr_avg"],
                    "ssim_avg": metrics["ssim_avg"],
                }
            )

    pd.DataFrame(rows).to_csv(config.quality_dir / "quality_metrics.csv", index=False)


if __name__ == "__main__":
    main()
