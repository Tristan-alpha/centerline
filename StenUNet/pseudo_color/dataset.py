"""Dataset utilities for pseudo color regression."""

from __future__ import annotations

from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence

import numpy as np
import torch
from torch.utils.data import Dataset

REQUIRED_TARGET = "feature_pseudo_color.npy"


def list_cases(data_root: str | Path) -> List[Path]:
    """List cases that contain the required target file."""
    root = Path(data_root)
    cases: List[Path] = []
    for case_dir in sorted(p for p in root.iterdir() if p.is_dir()):
        data_dir = case_dir / "data"
        if (data_dir / REQUIRED_TARGET).exists():
            cases.append(case_dir)
    if not cases:
        raise ValueError(f"No valid cases found under {root}. Expected {REQUIRED_TARGET}.")
    return cases


class PseudoColorFeatureDataset(Dataset):
    """Dataset that pairs masks with pseudo color supervision."""

    def __init__(
        self,
        data_root: str | Path,
        mask_root: str | Path,
        file_list: Optional[Sequence[str]] = None,
        transform: Optional[Any] = None,
    ) -> None:
        self.data_root = Path(data_root)
        self.mask_root = Path(mask_root)
        self.transform = transform
        all_cases = list_cases(self.data_root)
        if file_list is not None:
            case_lookup = {c.name: c for c in all_cases}
            missing = [stem for stem in file_list if stem not in case_lookup]
            if missing:
                raise ValueError(f"Missing cases in {self.data_root}: {missing}")
            self.cases = [case_lookup[stem] for stem in file_list]
        else:
            self.cases = all_cases

    def __len__(self) -> int:
        return len(self.cases)

    def _load_case(self, case_dir: Path) -> Dict[str, np.ndarray]:
        data_dir = case_dir / "data"
        pseudo_color = np.load(data_dir / REQUIRED_TARGET).astype(np.float32)
        mask_path = self.mask_root / f"{case_dir.name}.png"
        if not mask_path.exists():
            mask_path = self.mask_root / f"{case_dir.name}.npy"
        if not mask_path.exists():
            raise FileNotFoundError(f"Mask not found for case {case_dir.name} in {self.mask_root}.")

        if mask_path.suffix.lower() == ".npy":
            mask_arr = np.load(mask_path)
        else:
            import imageio.v2 as imageio  # lazy import to avoid dependency at module load
            mask_arr = imageio.imread(mask_path)
        mask = (mask_arr > 0).astype(np.float32)

        inputs = mask[None, ...]
        target = pseudo_color.transpose(2, 0, 1)
        return {"inputs": inputs, "mask": mask, "target": target}

    def __getitem__(self, index: int) -> Dict[str, Any]:
        case_dir = self.cases[index]
        loaded = self._load_case(case_dir)

        sample: Dict[str, Any] = {
            "inputs": torch.from_numpy(loaded["inputs"]),
            "mask": torch.from_numpy(loaded["mask"])[None, ...],
            "target": torch.from_numpy(loaded["target"]),
            "meta": {
                "case": case_dir.name,
                "data_dir": str(case_dir / "data"),
            },
        }

        if self.transform is not None:
            sample = self.transform(sample)

        return sample


def collate_samples(batch: Sequence[Dict[str, Any]]) -> Dict[str, Any]:
    """Custom collate_fn that keeps metadata as a list of dictionaries."""
    inputs = torch.stack([item["inputs"] for item in batch])
    mask = torch.stack([item["mask"] for item in batch])
    target = torch.stack([item["target"] for item in batch])
    meta = [item.get("meta", {}) for item in batch]
    return {"inputs": inputs, "mask": mask, "target": target, "meta": meta}
