"""Inference script for the pseudo-color regression baseline."""

from __future__ import annotations

import argparse
from pathlib import Path
from typing import List

import imageio.v2 as imageio
import numpy as np
import torch
from tqdm import tqdm

from .dataset import list_cases
from .model import ResUNet


def _load_mask(mask_root: Path, case_name: str) -> np.ndarray:
    mask_path = mask_root / f"{case_name}.png"
    if not mask_path.exists():
        mask_path = mask_root / f"{case_name}.npy"
    if not mask_path.exists():
        raise FileNotFoundError(f"Mask not found for case {case_name} in {mask_root}.")
    if mask_path.suffix.lower() == ".npy":
        mask_arr = np.load(mask_path)
    else:
        mask_arr = imageio.imread(mask_path)
    return (mask_arr > 0).astype(np.float32)


def _apply_vessel_mask(tensor: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
    if mask.shape[1] == 1 and tensor.shape[1] > 1:
        mask = mask.expand(-1, tensor.shape[1], *mask.shape[2:])
    return tensor * mask


def run_inference(args: argparse.Namespace) -> None:
    device = torch.device(args.device if args.device else ("cuda" if torch.cuda.is_available() else "cpu"))
    checkpoint = torch.load(args.checkpoint, map_location=device)
    state_dict = checkpoint.get("model_state_dict", checkpoint)

    stem_weight = state_dict.get("stem.0.weight")
    head_weight = state_dict.get("head.2.weight")
    if stem_weight is None or head_weight is None:
        raise RuntimeError("Checkpoint is missing stem/head weights needed to build the model.")
    trained_in_channels = stem_weight.shape[1]
    out_channels = head_weight.shape[0]

    ckpt_args = checkpoint.get("args", {})
    base_channels = ckpt_args.get("base_channels", args.base_channels)
    num_stages = ckpt_args.get("num_stages", args.num_stages)

    model = ResUNet(
        in_channels=trained_in_channels,
        out_channels=out_channels,
        base_channels=base_channels,
        num_stages=num_stages,
        dimension=2,
    ).to(device)

    model.load_state_dict(state_dict)
    model.eval()

    data_root = Path(args.data_root)
    mask_root = Path(args.mask_dir)
    output_dir = Path(args.output_dir)
    npy_root = output_dir / "inferences"
    png_root = output_dir / "inference_png"

    case_dirs = list_cases(data_root)
    for case_dir in tqdm(case_dirs, desc="Inferring", unit="case"):
        mask = _load_mask(mask_root, case_dir.name)
        inputs = mask[None, ...].astype(np.float32)
        inputs_tensor = torch.from_numpy(inputs)[None, ...].to(device)
        mask_tensor = torch.from_numpy(mask[None, None, ...]).to(device=device, dtype=inputs_tensor.dtype)

        with torch.no_grad():
            prediction = model(inputs_tensor)
            prediction = _apply_vessel_mask(prediction, mask_tensor)
        prediction_cpu = prediction.squeeze(0).detach().cpu()

        rgb_float = prediction_cpu.clamp(0.0, 1.0).permute(1, 2, 0).numpy().astype(np.float32)

        npy_path = npy_root / case_dir.name / "data" / "predicted_pseudo_color.npy"
        npy_path.parent.mkdir(parents=True, exist_ok=True)
        np.save(npy_path, rgb_float)

        if args.save_png:
            png_path = png_root / f"{case_dir.name}.png"
            png_path.parent.mkdir(parents=True, exist_ok=True)
            imageio.imwrite(png_path, (rgb_float * 255.0).astype(np.uint8))


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Infer pseudo color scalar fields from binary masks.")
    parser.add_argument("--data-root", required=True, type=str, help="Root directory containing case folders with feature_* npy files.")
    parser.add_argument("--mask-dir", type=str, default="annotation/labelsTr", help="Directory containing binary masks aligned to cases.")
    parser.add_argument("--checkpoint", required=True, type=str, help="Path to trained model checkpoint.")
    parser.add_argument("--output-dir", type=str, default="./StenUNet/pseudo_color/runs", help="Directory to save outputs (npy + png).")
    parser.add_argument("--save-png", action="store_true", help="Export pseudo color PNG previews to a separate directory.")
    parser.add_argument("--device", type=str, default="", help="PyTorch device string.")
    parser.add_argument("--base-channels", type=int, default=32, help="Base channel count (must match training).")
    parser.add_argument("--num-stages", type=int, default=4, help="Number of encoder/decoder stages (match training).")
    return parser


def main() -> None:
    parser = build_parser()
    args = parser.parse_args()
    run_inference(args)


if __name__ == "__main__":
    main()
