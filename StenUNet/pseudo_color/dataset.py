"""Dataset utilities for pseudo color regression."""

from __future__ import annotations

from pathlib import Path
from typing import Any, Callable, Dict, List, Optional, Sequence, Tuple

import cv2
import numpy as np
import torch
from torch.utils.data import Dataset

SUPPORTED_EXTS: Tuple[str, ...] = (".npy", ".npz", ".png", ".jpg", ".jpeg", ".tif", ".tiff")


def load_image(path: Path) -> np.ndarray:
    """Load an image or numpy array and return an array of shape (H, W[, C])."""
    if path.suffix.lower() == ".npy":
        return np.load(path)
    if path.suffix.lower() == ".npz":
        with np.load(path) as data:
            keys = list(data.keys())
            if not keys:
                raise ValueError(f"No arrays were found in {path}.")
            return data[keys[0]]

    flag = cv2.IMREAD_UNCHANGED
    image = cv2.imread(str(path), flag)
    if image is None:
        raise FileNotFoundError(f"Failed to read {path}.")
    if image.ndim == 3:
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    return image


def ensure_float(array: np.ndarray) -> np.ndarray:
    if array.dtype == np.float32 or array.dtype == np.float64:
        return array.astype(np.float32, copy=False)
    return array.astype(np.float32) / (255.0 if array.dtype == np.uint8 else np.max(array) or 1.0)


def binarize_mask(mask: np.ndarray) -> np.ndarray:
    if mask.ndim == 3:
        mask = mask[..., 0]
    mask = ensure_float(mask)
    return (mask > 0.5).astype(np.float32)


def _signed_distance(mask: np.ndarray) -> np.ndarray:
    if mask.ndim != 2:
        raise ValueError("Signed distance calculation currently supports 2D masks only.")
    mask_uint8 = (mask > 0.5).astype(np.uint8)
    # Only compute distance inside the vessel; background stays at zero.
    distance = cv2.distanceTransform(mask_uint8, cv2.DIST_L2, 5)
    max_val = float(distance.max())
    if max_val > 0.0:
        distance /= max_val
    return distance.astype(np.float32)


def _coordinate_channels(shape: Tuple[int, ...]) -> np.ndarray:
    if len(shape) not in (2, 3):
        raise ValueError(f"Unsupported mask shape {shape}.")
    if len(shape) == 2:
        height, width = shape
        yy, xx = np.meshgrid(
            np.linspace(-1.0, 1.0, height, dtype=np.float32),
            np.linspace(-1.0, 1.0, width, dtype=np.float32),
            indexing="ij",
        )
        return np.stack([yy, xx], axis=0)
    depth, height, width = shape
    zz, yy, xx = np.meshgrid(
        np.linspace(-1.0, 1.0, depth, dtype=np.float32),
        np.linspace(-1.0, 1.0, height, dtype=np.float32),
        np.linspace(-1.0, 1.0, width, dtype=np.float32),
        indexing="ij",
    )
    return np.stack([zz, yy, xx], axis=0)


def _find_matching_file(stem: str, directory: Path) -> Path:
    for suffix in SUPPORTED_EXTS:
        path = directory / f"{stem}{suffix}"
        if path.exists():
            return path
    raise FileNotFoundError(f"Could not find a file for {stem} in {directory}.")


def prepare_input_channels(
    mask: np.ndarray, include_distance: bool, include_coords: bool
) -> np.ndarray:
    """Create model-ready input channels from a binary mask."""
    mask = mask.astype(np.float32)
    inputs: List[np.ndarray] = [mask[None, ...]]
    if include_distance:
        inputs.append(_signed_distance(mask)[None, ...])
    if include_coords:
        inputs.append(_coordinate_channels(mask.shape))
    return np.concatenate(inputs, axis=0).astype(np.float32)


class PseudoColorDataset(Dataset):
    """Dataset that pairs binary masks with pseudo color supervision."""

    def __init__(
        self,
        mask_dir: str | Path,
        target_dir: str | Path,
        target_mode: str = "scalar",
        include_distance: bool = True,
        include_coords: bool = True,
        transform: Optional[Callable[[Dict[str, torch.Tensor]], Dict[str, torch.Tensor]]] = None,
        file_list: Optional[Sequence[str]] = None,
    ) -> None:
        self.mask_dir = Path(mask_dir)
        self.target_dir = Path(target_dir)
        self.target_mode = target_mode
        if target_mode not in {"scalar", "rgb"}:
            raise ValueError(f"target_mode must be 'scalar' or 'rgb', got {target_mode}.")
        self.include_distance = include_distance
        self.include_coords = include_coords
        self.transform = transform

        if file_list is not None:
            self.sample_stems = list(file_list)
        else:
            self.sample_stems = sorted(
                file.stem
                for file in self.mask_dir.iterdir()
                if file.suffix.lower() in SUPPORTED_EXTS
            )
        if not self.sample_stems:
            raise ValueError(f"No mask files were found in {self.mask_dir}.")

    def __len__(self) -> int:
        return len(self.sample_stems)

    def _load_target(self, stem: str) -> np.ndarray:
        target_path = _find_matching_file(stem, self.target_dir)
        target = load_image(target_path)
        if self.target_mode == "scalar":
            if target.ndim == 3:
                target = target[..., 0]
            target = ensure_float(target)
        else:
            target = ensure_float(target)
            if target.ndim == 2:
                target = np.repeat(target[..., None], 3, axis=-1)
        return target

    def __getitem__(self, index: int) -> Dict[str, Any]:
        stem = self.sample_stems[index]
        mask_path = _find_matching_file(stem, self.mask_dir)
        mask = binarize_mask(load_image(mask_path))

        inputs_array = prepare_input_channels(mask, self.include_distance, self.include_coords)
        target_array = self._load_target(stem)

        if self.target_mode == "scalar":
            target_tensor = torch.from_numpy(target_array.astype(np.float32))[None, ...]
        else:
            target_tensor = torch.from_numpy(target_array.astype(np.float32)).permute(2, 0, 1)

        sample: Dict[str, Any] = {
            "inputs": torch.from_numpy(inputs_array),
            "mask": torch.from_numpy(mask.astype(np.float32))[None, ...],
            "target": target_tensor,
        }

        if self.transform is not None:
            sample = self.transform(sample)

        sample["meta"] = {"stem": stem, "mask_path": str(mask_path)}
        return sample


def collate_samples(batch: Sequence[Dict[str, Any]]) -> Dict[str, Any]:
    """Custom collate_fn that keeps metadata as a list of dictionaries."""
    inputs = torch.stack([item["inputs"] for item in batch])
    mask = torch.stack([item["mask"] for item in batch])
    target = torch.stack([item["target"] for item in batch])
    meta = [item.get("meta", {}) for item in batch]
    return {"inputs": inputs, "mask": mask, "target": target, "meta": meta}
