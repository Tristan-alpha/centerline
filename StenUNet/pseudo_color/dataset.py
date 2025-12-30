"""Dataset utilities for pseudo color regression using raw images."""

from __future__ import annotations

from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence, Tuple

import imageio.v2 as imageio
import numpy as np
import torch
from torch.utils.data import Dataset

REQUIRED_TARGET = "feature_pseudo_color.npy"
SUPPORTED_EXTS: Tuple[str, ...] = (".png", ".jpg", ".jpeg", ".tif", ".tiff", ".npy")


def find_image(stem: str, image_root: Path) -> Optional[Path]:
    exact = []
    prefixed = []
    for ext in SUPPORTED_EXTS:
        path = image_root / f"{stem}{ext}"
        if path.exists():
            exact.append(path)
    if exact:
        return sorted(exact)[0]

    # fallback: first file whose stem starts with stem + "_"
    for path in sorted(image_root.iterdir()):
        if not path.is_file():
            continue
        if path.suffix.lower() not in SUPPORTED_EXTS:
            continue
        if path.stem.startswith(f"{stem}_"):
            prefixed.append(path)
    if prefixed:
        return prefixed[0]
    return None


def load_image(path: Path) -> np.ndarray:
    """Load a grayscale image and normalize to [0, 1]."""
    if path.suffix.lower() == ".npy":
        arr = np.load(path)
    else:
        arr = imageio.imread(path)
    if arr.ndim == 3:
        arr = arr[..., 0]
    arr = arr.astype(np.float32)
    if arr.max() > 1.0:
        arr = arr / 255.0
    return arr


def list_cases_with_targets(data_root: str | Path, image_root: str | Path) -> List[str]:
    """Cases that have targets in data_root and corresponding images in image_root."""
    root = Path(data_root)
    image_root = Path(image_root)
    cases: List[str] = []
    for case_dir in sorted(p for p in root.iterdir() if p.is_dir()):
        data_dir = case_dir / "data"
        if not (data_dir / REQUIRED_TARGET).exists():
            continue
        if find_image(case_dir.name, image_root) is not None:
            cases.append(case_dir.name)
    if not cases:
        raise ValueError(f"No valid cases with targets found under {root}.")
    return cases


def list_image_cases(image_root: str | Path) -> List[str]:
    """List all image stems under image_root."""
    root = Path(image_root)
    stems: List[str] = []
    for path in sorted(root.iterdir()):
        if path.is_file() and path.suffix.lower() in SUPPORTED_EXTS:
            stems.append(path.stem)
    if not stems:
        raise ValueError(f"No images found under {root}.")
    return stems


class PseudoColorFeatureDataset(Dataset):
    """Dataset that pairs raw images with pseudo color supervision."""

    def __init__(
        self,
        data_root: str | Path,
        image_root: str | Path,
        file_list: Optional[Sequence[str]] = None,
        transform: Optional[Any] = None,
    ) -> None:
        self.data_root = Path(data_root)
        self.image_root = Path(image_root)
        self.transform = transform
        all_cases = list_cases_with_targets(self.data_root, self.image_root)
        if file_list is not None:
            missing = [stem for stem in file_list if stem not in all_cases]
            if missing:
                raise ValueError(f"Missing cases in targets or images: {missing}")
            self.cases = list(file_list)
        else:
            self.cases = all_cases

    def __len__(self) -> int:
        return len(self.cases)

    def _load_case(self, case_name: str) -> Dict[str, np.ndarray]:
        data_dir = self.data_root / case_name / "data"
        pseudo_color = np.load(data_dir / REQUIRED_TARGET).astype(np.float32)
        img_path = find_image(case_name, self.image_root)
        if img_path is None:
            raise FileNotFoundError(f"Image not found for case {case_name} in {self.image_root}.")
        image = load_image(img_path)

        inputs = image[None, ...]
        target = pseudo_color.transpose(2, 0, 1)
        return {"inputs": inputs, "target": target}

    def __getitem__(self, index: int) -> Dict[str, Any]:
        case_name = self.cases[index]
        loaded = self._load_case(case_name)

        sample: Dict[str, Any] = {
            "inputs": torch.from_numpy(loaded["inputs"]),
            "target": torch.from_numpy(loaded["target"]),
            "meta": {
                "case": case_name,
                "data_dir": str(self.data_root / case_name / "data"),
            },
        }

        if self.transform is not None:
            sample = self.transform(sample)

        return sample


def collate_samples(batch: Sequence[Dict[str, Any]]) -> Dict[str, Any]:
    """Custom collate_fn that keeps metadata as a list of dictionaries."""
    inputs = torch.stack([item["inputs"] for item in batch])
    target = torch.stack([item["target"] for item in batch])
    meta = [item.get("meta", {}) for item in batch]
    return {"inputs": inputs, "target": target, "meta": meta}
