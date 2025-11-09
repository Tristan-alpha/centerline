"""Inference script for the pseudo-color regression baseline."""

from __future__ import annotations

import argparse
from pathlib import Path
from typing import List

import cv2
import numpy as np
import torch
from tqdm import tqdm

from .colormap import apply_colormap
from .dataset import SUPPORTED_EXTS, binarize_mask, load_image, prepare_input_channels
from .model import ResUNet


def list_mask_files(mask_dir: Path) -> List[Path]:
    files = [
        path
        for path in mask_dir.iterdir()
        if path.is_file() and path.suffix.lower() in SUPPORTED_EXTS
    ]
    if not files:
        raise ValueError(f"No supported mask files were found in {mask_dir}.")
    return sorted(files)


def save_scalar_field(array: np.ndarray, path: Path, fmt: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    if fmt == "npy":
        np.save(path, array.astype(np.float32))
    elif fmt == "pt":
        torch.save(torch.from_numpy(array.astype(np.float32)), path)
    elif fmt == "png":
        scaled = np.clip(array, 0.0, 1.0)
        scaled = (scaled * 255.0).astype(np.uint8)
        cv2.imwrite(str(path), scaled)
    else:
        raise ValueError(f"Unsupported format {fmt}.")


def save_color_map(tensor: torch.Tensor, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    array = tensor.permute(1, 2, 0).cpu().numpy()
    array = np.clip(array, 0.0, 1.0)
    bgr = cv2.cvtColor((array * 255.0).astype(np.uint8), cv2.COLOR_RGB2BGR)
    cv2.imwrite(str(path), bgr)


def run_inference(args: argparse.Namespace) -> None:
    device = torch.device(args.device if args.device else ("cuda" if torch.cuda.is_available() else "cpu"))
    checkpoint = torch.load(args.checkpoint, map_location=device)

    in_channels = 1 + int(args.include_distance) + (2 if args.include_coords else 0)
    model = ResUNet(
        in_channels=in_channels,
        out_channels=1,
        base_channels=args.base_channels,
        num_stages=args.num_stages,
        dimension=2,
    ).to(device)

    state_dict = checkpoint.get("model_state_dict", checkpoint)
    model.load_state_dict(state_dict)
    model.eval()

    mask_dir = Path(args.mask_dir)
    output_dir = Path(args.output_dir)
    scalar_dir = output_dir / "scalar"
    color_dir = output_dir / "pseudo_color"

    mask_files = list_mask_files(mask_dir)
    for path in tqdm(mask_files, desc="Inferring", unit="image"):
        array = load_image(path)
        mask = binarize_mask(array)
        inputs = prepare_input_channels(mask, args.include_distance, args.include_coords)
        inputs_tensor = torch.from_numpy(inputs)[None, ...].to(device)

        with torch.no_grad():
            prediction = model(inputs_tensor)
            if args.sigmoid:
                prediction = prediction.sigmoid()
        prediction_cpu = prediction.squeeze(0).detach().cpu()

        scalar = prediction_cpu.numpy()[0]
        stem = path.stem
        scalar_path = scalar_dir / f"{stem}.{args.output_format}"
        save_scalar_field(scalar, scalar_path, args.output_format)

        if args.save_color:
            color = apply_colormap(prediction_cpu.unsqueeze(0)).squeeze(0)
            save_color_map(color, color_dir / f"{stem}.png")


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Infer pseudo color scalar fields from binary masks.")
    parser.add_argument("--mask-dir", required=True, type=str, help="Directory with binary mask files.")
    parser.add_argument("--checkpoint", required=True, type=str, help="Path to trained model checkpoint.")
    parser.add_argument("--output-dir", type=str, default="./pseudo_color_outputs", help="Directory to save outputs.")
    parser.add_argument("--include-distance", action="store_true", help="Enable distance channel to mirror training.")
    parser.add_argument("--include-coords", action="store_true", help="Enable coordinate channels to mirror training.")
    parser.add_argument("--output-format", type=str, default="npy", choices=["npy", "pt", "png"], help="Scalar field format.")
    parser.add_argument("--save-color", action="store_true", help="Export pseudo color PNG overlays.")
    parser.add_argument("--sigmoid", action="store_true", help="Apply sigmoid to prediction before saving.")
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
