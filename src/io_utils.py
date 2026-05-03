"""Common IO and logging helpers."""

from __future__ import annotations

import logging
import sys
from pathlib import Path
from typing import Iterable


def setup_logging(level: int = logging.INFO) -> None:
    logging.basicConfig(
        level=level,
        format="%(asctime)s | %(levelname)s | %(name)s | %(message)s",
        handlers=[logging.StreamHandler(sys.stdout)],
        force=True,
    )


def ensure_dir(path: str | Path) -> Path:
    target = Path(path)
    target.mkdir(parents=True, exist_ok=True)
    return target


def ensure_dirs(paths: Iterable[str | Path]) -> None:
    for path in paths:
        ensure_dir(path)


def should_skip_output(path: str | Path, overwrite: bool) -> bool:
    return Path(path).exists() and not overwrite
