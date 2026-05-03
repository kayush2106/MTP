from __future__ import annotations

import argparse
import logging
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from src.config import load_config
from src.ffmpeg_utils import ffmpeg_image_sequence_to_video
from src.io_utils import ensure_dir, setup_logging, should_skip_output
from src.mot_utils import read_seqinfo

LOGGER = logging.getLogger(__name__)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", required=True)
    parser.add_argument("--overwrite", action="store_true")
    args = parser.parse_args()

    setup_logging()
    config = load_config(args.config)
    ensure_dir(config.source_video_dir)

    for sequence in config.sequences:
        sequence_dir = config.dataset_root / sequence
        seqinfo = read_seqinfo(sequence_dir / "seqinfo.ini")
        fps = config.fps_override or seqinfo["frameRate"]
        image_pattern = sequence_dir / seqinfo["imDir"] / "%06d.jpg"
        output_path = config.source_video_dir / f"{sequence}_source.mp4"
        if should_skip_output(output_path, args.overwrite):
            LOGGER.info("Skipping existing file %s", output_path)
            continue
        ffmpeg_image_sequence_to_video(image_pattern=image_pattern, output_path=output_path, fps=fps, overwrite=args.overwrite)


if __name__ == "__main__":
    main()
