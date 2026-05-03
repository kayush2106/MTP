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
from src.ffmpeg_utils import ffmpeg_compress_video, ffprobe_video_metadata
from src.io_utils import ensure_dir, setup_logging, should_skip_output

LOGGER = logging.getLogger(__name__)


def build_metadata_row(sequence: str, source_or_compressed: str, profile_name: str, file_path: Path, profile: dict | None) -> dict:
    metadata = ffprobe_video_metadata(file_path)
    return {
        "sequence": sequence,
        "source_or_compressed": source_or_compressed,
        "profile_name": profile_name,
        "file_path": str(file_path),
        "codec": profile["codec"] if profile else metadata.get("codec_name"),
        "crf": profile["crf"] if profile else None,
        "preset": profile["preset"] if profile else None,
        "gop": profile["gop"] if profile else None,
        "width": metadata.get("width"),
        "height": metadata.get("height"),
        "fps": metadata.get("avg_frame_rate"),
        "bitrate": metadata.get("bit_rate"),
        "duration": metadata.get("duration"),
        "nb_frames": metadata.get("nb_frames"),
        "file_size_bytes": metadata.get("file_size_bytes"),
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", required=True)
    parser.add_argument("--overwrite", action="store_true")
    args = parser.parse_args()

    setup_logging()
    config = load_config(args.config)
    ensure_dir(config.compressed_video_dir)
    ensure_dir(config.tables_dir)

    rows: list[dict] = []
    for sequence in config.sequences:
        source_video = config.source_video_dir / f"{sequence}_source.mp4"
        if not source_video.exists():
            raise FileNotFoundError(f"Missing source video: {source_video}")
        rows.append(build_metadata_row(sequence, "source", "source", source_video, None))

        output_sequence_dir = ensure_dir(config.compressed_video_dir / sequence)
        for profile in config.compression_profiles:
            profile_dict = {
                "codec": profile.codec,
                "crf": profile.crf,
                "preset": profile.preset,
                "gop": profile.gop,
            }
            output_path = output_sequence_dir / f"{sequence}_{profile.name}.mp4"
            if should_skip_output(output_path, args.overwrite):
                LOGGER.info("Skipping existing file %s", output_path)
            else:
                ffmpeg_compress_video(
                    input_path=source_video,
                    output_path=output_path,
                    codec=profile.codec,
                    crf=profile.crf,
                    preset=profile.preset,
                    gop=profile.gop,
                    overwrite=args.overwrite,
                )
            rows.append(build_metadata_row(sequence, "compressed", profile.name, output_path, profile_dict))

    pd.DataFrame(rows).to_csv(config.tables_dir / "video_metadata.csv", index=False)


if __name__ == "__main__":
    main()
