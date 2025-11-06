from __future__ import annotations

import csv
import json
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, Iterable, Iterator, List, Tuple

import numpy as np
from PIL import Image


@dataclass(frozen=True)
class CaseData:
    """Container for a single mask case."""

    case_id: str
    mask_path: Path
    start: Tuple[int, int]
    end: Tuple[int, int]


def load_annotations(path: Path) -> Dict[str, List[List[int]]]:
    """Load the annotations JSON and return it as a dict."""
    with path.open("r", encoding="utf-8") as f:
        data = json.load(f)
    if not isinstance(data, dict):
        raise ValueError("annotations.json must contain a mapping of filenames to points")
    return data


def iter_cases(mask_dir: Path, annotations: Dict[str, List[List[int]]]) -> Iterator[CaseData]:
    """Yield cases that have both masks and point annotations."""
    for case_id, points in annotations.items():
        if len(points) != 2:
            continue
        mask_path = mask_dir / case_id
        if not mask_path.exists():
            continue
        start_xy, end_xy = points
        # annotations provide [x, y]; convert to (row, col) = (y, x)
        start = (int(start_xy[1]), int(start_xy[0]))
        end = (int(end_xy[1]), int(end_xy[0]))
        yield CaseData(
            case_id=case_id,
            mask_path=mask_path,
            start=start,
            end=end,
        )


def load_mask(path: Path) -> np.ndarray:
    """Read a grayscale mask image and return a boolean array."""
    mask = np.array(Image.open(path))
    if mask.ndim == 3:
        # Collapse RGB(A) masks to single channel via max.
        mask = mask.max(axis=2)
    # Treat any non-zero as vessel foreground.
    return mask > 0


def ensure_dir(path: Path) -> None:
    """Create path if missing."""
    path.mkdir(parents=True, exist_ok=True)


def save_json(path: Path, payload: Dict) -> None:
    with path.open("w", encoding="utf-8") as f:
        json.dump(payload, f, indent=2)


def save_csv(path: Path, rows: Iterable[Iterable]) -> None:
    with path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)
        writer.writerows(rows)
