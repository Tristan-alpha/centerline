"""Inference script for the pseudo-color regression baseline."""

from __future__ import annotations

import argparse
from pathlib import Path
from typing import Dict, List, Tuple

import imageio.v2 as imageio
import numpy as np
import torch
from matplotlib.colors import hsv_to_rgb
from tqdm import tqdm

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


def save_field(array: np.ndarray, path: Path, fmt: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    array = array.astype(np.float32)
    if fmt == "npy":
        np.save(path, array)
        return
    if fmt == "pt":
        torch.save(torch.from_numpy(array), path)
        return
    if fmt == "png":
        clipped = np.clip(array, 0.0, 1.0)
        if clipped.ndim == 2:
            scaled = (clipped * 255.0).astype(np.uint8)
        elif clipped.ndim == 3 and clipped.shape[2] == 3:
            scaled = (clipped * 255.0).astype(np.uint8)
            # imageio expects RGB, no conversion needed
        else:
            raise ValueError("PNG export expects either HxW or HxWx3 arrays.")
        imageio.imwrite(str(path), scaled)
        return
    raise ValueError(f"Unsupported format {fmt}.")


def save_color_map(tensor: torch.Tensor, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    array = tensor.permute(1, 2, 0).cpu().numpy()
    array = np.clip(array, 0.0, 1.0)
    rgb = (array * 255.0).astype(np.uint8)
    imageio.imwrite(str(path), rgb)


def export_stenunet_modality(hsv_array: np.ndarray, stem: str, args: argparse.Namespace) -> None:
    images_dir = Path(args.stenunet_images_dir)
    images_dir.mkdir(parents=True, exist_ok=True)
    suffix = args.stenunet_file_ending
    if not suffix.startswith("."):
        suffix = "." + suffix
    output_path = images_dir / f"{stem}_{args.stenunet_channel_index:04d}{suffix}"
    hsv_uint8 = np.clip(hsv_array * 255.0, 0.0, 255.0).astype(np.uint8)
    imageio.imwrite(str(output_path), hsv_uint8)


def _infer_channel_flags(total_in_channels: int) -> Tuple[bool, bool]:
    combos: Dict[int, Tuple[bool, bool]] = {
        1: (False, False),
        2: (True, False),
        3: (False, True),
        4: (True, True),
    }
    if total_in_channels not in combos:
        raise ValueError(f"Unsupported input channel count {total_in_channels} in checkpoint.")
    return combos[total_in_channels]


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

    include_distance, include_coords = _infer_channel_flags(trained_in_channels)
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

    mask_dir = Path(args.mask_dir)
    output_dir = Path(args.output_dir)
    hsv_dir = output_dir / "hsv"
    color_dir = output_dir / "pseudo_color"

    mask_files = list_mask_files(mask_dir)
    for path in tqdm(mask_files, desc="Inferring", unit="image"):
        array = load_image(path)
        mask = binarize_mask(array)
        inputs = prepare_input_channels(mask, include_distance, include_coords)
        inputs_tensor = torch.from_numpy(inputs)[None, ...].to(device)
        mask_tensor = torch.from_numpy(mask[None, None, ...]).to(device=device, dtype=inputs_tensor.dtype)

        with torch.no_grad():
            prediction = model(inputs_tensor)
            if args.sigmoid:
                prediction = prediction.sigmoid()
            prediction = prediction * mask_tensor
        prediction_cpu = prediction.squeeze(0).detach().cpu()
        stem = path.stem
        if out_channels == 1:
            scalar = prediction_cpu.numpy()[0]
            scalar_path = hsv_dir / f"{stem}.{args.output_format}"
            save_field(scalar, scalar_path, args.output_format)
            if args.save_color:
                color = prediction_cpu.unsqueeze(0).repeat(1, 3, 1, 1)
                save_color_map(color.squeeze(0), color_dir / f"{stem}.png")
        elif out_channels == 3:
            hsv = prediction_cpu.clamp(0.0, 1.0).permute(1, 2, 0).numpy()
            hsv_path = hsv_dir / f"{stem}.{args.output_format}"
            save_field(hsv, hsv_path, args.output_format)
            rgb = hsv_to_rgb(hsv)
            if args.save_color or args.output_format != "png":
                save_field(rgb, color_dir / f"{stem}.png", "png")
            if args.stenunet_images_dir:
                export_stenunet_modality(hsv, stem, args)
        else:
            raise ValueError(f"Unsupported number of output channels: {out_channels}")


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Infer pseudo color scalar fields from binary masks.")
    parser.add_argument("--mask-dir", required=True, type=str, help="Directory with binary mask files.")
    parser.add_argument("--checkpoint", required=True, type=str, help="Path to trained model checkpoint.")
    parser.add_argument("--output-dir", type=str, default="./StenUNet/pseudo_color/runs", help="Directory to save outputs.")
    parser.add_argument(
        "--output-format",
        type=str,
        default="npy",
        choices=["npy", "pt", "png"],
        help="Export format for raw network predictions.",
    )
    parser.add_argument("--save-color", action="store_true", help="Export pseudo color PNG overlays (scalar mode) or RGB predictions (RGB mode).")
    parser.add_argument("--sigmoid", action="store_true", help="Apply sigmoid to prediction before saving.")
    parser.add_argument("--device", type=str, default="", help="PyTorch device string.")
    parser.add_argument("--base-channels", type=int, default=32, help="Base channel count (must match training).")
    parser.add_argument("--num-stages", type=int, default=4, help="Number of encoder/decoder stages (match training).")
    parser.add_argument(
        "--stenunet-images-dir",
        type=str,
        default="",
        help=(
            "Optional path to a StenUNet Raw_data images directory (e.g., nnNet_training/Raw_data/"
            "Dataset_Train_val/imagesTr) where HSV representations will be exported as an extra modality."
        ),
    )
    parser.add_argument(
        "--stenunet-channel-index",
        type=int,
        default=2,
        help="Channel index suffix used when exporting to StenUNet format (produces *_XXXX.png).",
    )
    parser.add_argument(
        "--stenunet-file-ending",
        type=str,
        default=".png",
        help="File extension when exporting HSV modality for StenUNet integration.",
    )
    return parser


def main() -> None:
    parser = build_parser()
    args = parser.parse_args()
    run_inference(args)


if __name__ == "__main__":
    main()
