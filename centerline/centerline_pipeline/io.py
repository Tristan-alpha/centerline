from __future__ import annotations

import csv
import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, Iterable, Iterator, List, Sequence, Tuple

import numpy as np
from PIL import Image


@dataclass(frozen=True)
class PointPair:
    """Hold a start/end annotation for a single centerline request."""

    index: int
    start: Tuple[int, int]
    end: Tuple[int, int]
    start_xy: Tuple[int, int]
    end_xy: Tuple[int, int]


@dataclass(frozen=True)
class CaseData:
    """Container for a mask case with one or more annotated paths."""

    case_id: str
    mask_path: Path
    pairs: Tuple[PointPair, ...]


def load_annotations(path: Path) -> Dict[str, Any]:
    """Load the annotations JSON and return it as a dict."""
    with path.open("r", encoding="utf-8") as f:
        data = json.load(f)
    if not isinstance(data, dict):
        raise ValueError("annotations.json must contain a mapping of filenames to points")
    return data


def iter_cases(mask_dir: Path, annotations: Dict[str, Any]) -> Iterator[CaseData]:
    """Yield cases that have masks plus at least one annotated point-pair."""
    for case_id, raw_points in annotations.items():
        mask_path = mask_dir / case_id
        if not mask_path.exists():
            continue
        normalized_pairs = _normalize_annotation_entry(raw_points)
        if not normalized_pairs:
            continue

        requests: List[PointPair] = []
        for pair_idx, (start_xy, end_xy) in enumerate(normalized_pairs):
            start = (int(start_xy[1]), int(start_xy[0]))
            end = (int(end_xy[1]), int(end_xy[0]))
            requests.append(
                PointPair(
                    index=pair_idx,
                    start=start,
                    end=end,
                    start_xy=(int(start_xy[0]), int(start_xy[1])),
                    end_xy=(int(end_xy[0]), int(end_xy[1])),
                )
            )

        if requests:
            yield CaseData(
                case_id=case_id,
                mask_path=mask_path,
                pairs=tuple(requests),
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


def _is_point(candidate: Sequence) -> bool:
    return (
        isinstance(candidate, (list, tuple))
        and len(candidate) == 2
        and all(isinstance(coord, (int, float)) for coord in candidate)
    )


def _normalize_annotation_entry(entry: Any) -> List[Tuple[Sequence[int], Sequence[int]]]:
    """Return a list of [[x, y], [x, y]] pairs."""
    if _is_single_pair(entry):
        return [(entry[0], entry[1])]

    normalized: List[Tuple[Sequence[int], Sequence[int]]] = []
    if isinstance(entry, list):
        for pair in entry:
            if _is_single_pair(pair):
                normalized.append((pair[0], pair[1]))
    return normalized


def _is_single_pair(entry: Any) -> bool:
    return isinstance(entry, list) and len(entry) == 2 and all(_is_point(point) for point in entry)
