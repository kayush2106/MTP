# compression_cv_study

`compression_cv_study` is a CPU-only Windows-friendly study pipeline for measuring how video compression changes person detection performance on MOT17 sequences using Ultralytics YOLOv8n.

## What it does

The project:

- converts MOT17 image sequences into source MP4 videos
- compresses each source video with multiple H.264 and H.265 profiles
- runs YOLOv8n person detection on source and compressed videos
- evaluates detections against MOT17 ground truth
- computes PSNR and SSIM quality metrics
- aggregates metadata and metrics into CSV files
- generates plots for compression-vs-detection analysis

## Repository layout

```text
compression_cv_study/
├── README.md
├── requirements.txt
├── config/
│   └── default.yaml
├── data/
│   ├── raw/
│   │   └── MOT17/
│   ├── source_videos/
│   ├── compressed_videos/
│   └── intermediate/
├── results/
│   ├── detections/
│   ├── metrics/
│   ├── quality/
│   ├── tables/
│   └── plots/
├── scripts/
│   ├── aggregate_results.py
│   ├── compress_videos.py
│   ├── compute_quality_metrics.py
│   ├── convert_mot_to_video.py
│   ├── evaluate_detection.py
│   ├── make_plots.py
│   └── run_detection.py
└── src/
    ├── __init__.py
    ├── aggregation.py
    ├── config.py
    ├── detection.py
    ├── evaluation.py
    ├── ffmpeg_utils.py
    ├── io_utils.py
    ├── mot_utils.py
    ├── plotting.py
    └── quality_metrics.py
```

## Dataset expectations

Place the MOT17 sequences manually inside [data/raw/MOT17](/C:/Users/kumar/OneDrive%20-%20iitkgp.ac.in/Documents/New%20project/data/raw/MOT17). Each selected sequence should look like:

```text
data/raw/MOT17/MOT17-02/
├── img1/
├── gt/
│   └── gt.txt
└── seqinfo.ini
```

Ground-truth parsing keeps rows with `conf > 0`. If the MOT row includes class information, the parser keeps class `1` rows as person annotations. Malformed rows are skipped with warnings.

## Requirements

- Windows 11
- Python 3.10+
- FFmpeg and `ffprobe` available in `PATH`
- CPU-only execution

Install dependencies with:

```bash
python -m pip install -r requirements.txt
```

## Configuration

The default configuration lives at [config/default.yaml](/C:/Users/kumar/OneDrive%20-%20iitkgp.ac.in/Documents/New%20project/config/default.yaml). Update it to choose sequences, frame limits, or compression profiles.

## Run the pipeline

```bash
python scripts/convert_mot_to_video.py --config config/default.yaml
python scripts/compress_videos.py --config config/default.yaml
python scripts/run_detection.py --config config/default.yaml
python scripts/evaluate_detection.py --config config/default.yaml
python scripts/compute_quality_metrics.py --config config/default.yaml
python scripts/aggregate_results.py --config config/default.yaml
python scripts/make_plots.py --config config/default.yaml
```

Common options:

- `--overwrite` for stages that create files
- `--frame-limit` and `--every-nth-frame` on detection

## Outputs

The main outputs are:

- [results/tables/video_metadata.csv](/C:/Users/kumar/OneDrive%20-%20iitkgp.ac.in/Documents/New%20project/results/tables/video_metadata.csv)
- [results/metrics/detection_metrics.csv](/C:/Users/kumar/OneDrive%20-%20iitkgp.ac.in/Documents/New%20project/results/metrics/detection_metrics.csv)
- [results/quality/quality_metrics.csv](/C:/Users/kumar/OneDrive%20-%20iitkgp.ac.in/Documents/New%20project/results/quality/quality_metrics.csv)
- [results/tables/final_results.csv](/C:/Users/kumar/OneDrive%20-%20iitkgp.ac.in/Documents/New%20project/results/tables/final_results.csv)
- PNG plots under [results/plots](/C:/Users/kumar/OneDrive%20-%20iitkgp.ac.in/Documents/New%20project/results/plots)

## Notes

- The project uses pretrained `yolov8n.pt` weights from Ultralytics and does not train a model.
- FFmpeg metric parsing is best-effort. If PSNR or SSIM parsing fails, the pipeline stores `NaN` and logs a warning instead of failing the entire run.