"""Train a residual U-Net to regress pseudo color fields from masks."""

from __future__ import annotations

import argparse
import json
import random
from pathlib import Path
from typing import Dict, Optional

import numpy as np
import torch
from torch import Tensor
from torch.amp import autocast
from torch.cuda.amp import GradScaler
from torch.optim import AdamW
from torch.optim.lr_scheduler import OneCycleLR
from torch.utils.data import DataLoader
from tqdm import tqdm

from .colormap import apply_colormap
from .dataset import PseudoColorDataset, collate_samples
from .losses import ScalarFieldLoss
from .model import ResUNet


def set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def _match_mask_shape(mask: Tensor, reference: Tensor) -> Tensor:
    if mask.shape == reference.shape:
        return mask
    if mask.ndim != reference.ndim:
        raise ValueError("Mask must share the same dimensionality as the reference tensor.")
    if mask.shape[0] != reference.shape[0]:
        raise ValueError("Batch dimension mismatch between mask and tensor.")
    if mask.shape[1] == 1:
        expand_shape = (-1, reference.shape[1], *reference.shape[2:])
        return mask.expand(expand_shape)
    raise ValueError("Mask must either match reference shape or provide a singleton channel.")


def apply_vessel_mask(tensor: Tensor, mask: Tensor) -> Tensor:
    mask_aligned = _match_mask_shape(mask, tensor)
    return tensor * mask_aligned


def compute_mask_mae(prediction: Tensor, target: Tensor, mask: Tensor) -> float:
    eps = 1e-6
    mask_aligned = _match_mask_shape(mask, prediction)
    error = (prediction - target).abs() * mask_aligned
    denom = mask_aligned.sum() + eps
    return float(error.sum().item() / denom.item())


def build_dataloader(
    mask_dir: str | Path,
    target_dir: str | Path,
    target_mode: str,
    include_distance: bool,
    include_coords: bool,
    batch_size: int,
    shuffle: bool,
    num_workers: int,
) -> DataLoader:
    dataset = PseudoColorDataset(
        mask_dir=mask_dir,
        target_dir=target_dir,
        target_mode=target_mode,
        include_distance=include_distance,
        include_coords=include_coords,
    )
    return DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=shuffle,
        num_workers=num_workers,
        pin_memory=True,
        collate_fn=collate_samples,
    )


def train(args: argparse.Namespace) -> None:
    device = torch.device(args.device if args.device else ("cuda" if torch.cuda.is_available() else "cpu"))
    set_seed(args.seed)

    target_mode = args.target_mode.lower()
    if target_mode not in {"scalar", "rgb"}:
        raise ValueError("target_mode must be 'scalar' or 'rgb'.")

    train_loader = build_dataloader(
        args.train_mask_dir,
        args.train_target_dir,
        target_mode,
        args.include_distance,
        args.include_coords,
        args.batch_size,
        shuffle=True,
        num_workers=args.num_workers,
    )

    val_loader: Optional[DataLoader]
    if args.val_mask_dir and args.val_target_dir:
        val_loader = build_dataloader(
            args.val_mask_dir,
            args.val_target_dir,
            target_mode,
            args.include_distance,
            args.include_coords,
            args.batch_size,
            shuffle=False,
            num_workers=args.num_workers,
        )
    else:
        val_loader = None

    in_channels = 1 + int(args.include_distance) + (2 if args.include_coords else 0)
    out_channels = 3 if target_mode == "rgb" else 1
    model = ResUNet(
        in_channels=in_channels,
        out_channels=out_channels,
        base_channels=args.base_channels,
        num_stages=args.num_stages,
        dimension=2,
    ).to(device)

    optimizer = AdamW(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)

    scheduler = OneCycleLR(
        optimizer,
        max_lr=args.lr,
        steps_per_epoch=len(train_loader),
        epochs=args.epochs,
        pct_start=0.1,
    )

    loss_fn = ScalarFieldLoss(
        l1_weight=1.0,
        grad_weight=args.grad_weight,
        eikonal_weight=args.eikonal_weight,
        eikonal_target=args.eikonal_target,
        outside_weight=args.outside_weight,
    )

    scaler = GradScaler(enabled=args.amp)
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    best_val_loss = float("inf")

    for epoch in range(1, args.epochs + 1):
        model.train()
        running_loss = 0.0
        pbar = tqdm(train_loader, desc=f"Epoch {epoch}/{args.epochs}", unit="batch")
        for batch in pbar:
            inputs = batch["inputs"].to(device, non_blocking=True)
            target = batch["target"].to(device, non_blocking=True)
            mask = batch["mask"].to(device, non_blocking=True)

            optimizer.zero_grad(set_to_none=True)
            with autocast(device_type='cuda', enabled=args.amp):
                prediction = model(inputs)
                prediction = apply_vessel_mask(prediction, mask)
                loss = loss_fn(prediction, target, mask)

            scaler.scale(loss).backward()
            if args.grad_clip > 0.0:
                scaler.unscale_(optimizer)
                torch.nn.utils.clip_grad_norm_(model.parameters(), args.grad_clip)
            scaler.step(optimizer)
            scaler.update()
            scheduler.step()

            running_loss += loss.item()
            pbar.set_postfix(loss=f"{loss.item():.4f}")

        train_loss = running_loss / max(len(train_loader), 1)
        metrics: Dict[str, float] = {"train_loss": train_loss}

        if val_loader is not None and epoch % args.val_interval == 0:
            model.eval()
            val_loss = 0.0
            mae_total = 0.0
            with torch.no_grad():
                with autocast(device_type='cuda', enabled=args.amp):
                    for batch in val_loader:
                        inputs = batch["inputs"].to(device, non_blocking=True)
                        target = batch["target"].to(device, non_blocking=True)
                        mask = batch["mask"].to(device, non_blocking=True)
                        prediction = model(inputs)
                    prediction = apply_vessel_mask(prediction, mask)
                    loss = loss_fn(prediction, target, mask)
                    val_loss += loss.item()
                    mae_total += compute_mask_mae(prediction, target, mask)
            val_loss /= max(len(val_loader), 1)
            mae_total /= max(len(val_loader), 1)
            metrics["val_loss"] = val_loss
            metrics["val_mae"] = mae_total
            if val_loss < best_val_loss:
                best_val_loss = val_loss
                torch.save(
                    {
                        "model_state_dict": model.state_dict(),
                        "optimizer_state_dict": optimizer.state_dict(),
                        "epoch": epoch,
                        "metrics": metrics,
                        "args": vars(args),
                    },
                    output_dir / "best_model.pt",
                )

            if args.export_preview:
                preview_dir = output_dir / "val_preview"
                preview_dir.mkdir(exist_ok=True)
                preview_batch = next(iter(val_loader))
                preview_inputs = preview_batch["inputs"].to(device)
                with torch.no_grad():
                    preview_prediction = model(preview_inputs).cpu()
                preview_mask = preview_batch["mask"]
                masked_prediction = apply_vessel_mask(preview_prediction, preview_mask)
                if target_mode == "rgb":
                    pseudo_color = masked_prediction.clamp(0.0, 1.0)
                else:
                    pseudo_color = apply_colormap(masked_prediction)
                for idx, meta in enumerate(preview_batch["meta"]):
                    stem = meta.get("stem", f"sample_{idx}")
                    path = preview_dir / f"{stem}.pt"
                    torch.save(
                        {
                            "prediction": masked_prediction[idx],
                            "pseudo_color": pseudo_color[idx],
                            "target": preview_batch["target"][idx],
                        },
                        path,
                    )

        torch.save(
            {
                "model_state_dict": model.state_dict(),
                "optimizer_state_dict": optimizer.state_dict(),
                "epoch": epoch,
                "metrics": metrics,
                "args": vars(args),
                "scheduler_state_dict": scheduler.state_dict(),
            },
            output_dir / "last_model.pt",
        )

        if args.metrics_file:
            metrics_path = Path(args.metrics_file)
            metrics_path.parent.mkdir(parents=True, exist_ok=True)
            with metrics_path.open("a", encoding="utf-8") as handle:
                handle.write(json.dumps({"epoch": epoch, **metrics}) + "\n")


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Train pseudo-color regression baseline.")
    parser.add_argument("--train-mask-dir", type=str, default="StenUNet/pseudo_color/train_data/masks_train", help="Directory with binary mask PNG/NPY files.")
    parser.add_argument(
        "--train-target-dir",
        type=str,
        default="StenUNet/pseudo_color/train_data/targets_train",
        help="Directory with scalar or RGB pseudo color supervision.",
    )
    parser.add_argument("--val-mask-dir", type=str, default="StenUNet/pseudo_color/train_data/masks_val", help="Optional validation mask directory.")
    parser.add_argument("--val-target-dir", type=str, default="StenUNet/pseudo_color/train_data/targets_val", help="Optional validation target directory.")
    parser.add_argument(
        "--target-mode",
        type=str,
        default="rgb",
        choices=["scalar", "rgb"],
        help="Supervision mode (scalar or full RGB pseudo color).",
    )
    parser.add_argument("--include-distance", action="store_true", help="Append signed distance channel to the input.")
    parser.add_argument("--include-coords", action="store_true", help="Append coordinate channels to the input.")
    parser.add_argument("--batch-size", type=int, default=4, help="Training batch size.")
    parser.add_argument("--epochs", type=int, default=50, help="Number of training epochs.")
    parser.add_argument("--lr", type=float, default=3e-4, help="Peak learning rate for OneCycleLR.")
    parser.add_argument("--weight-decay", type=float, default=1e-5, help="Weight decay.")
    parser.add_argument("--grad-clip", type=float, default=1.0, help="Gradient clipping norm (0 to disable).")
    parser.add_argument("--grad-weight", type=float, default=0.1, help="Weight for gradient matching loss.")
    parser.add_argument("--eikonal-weight", type=float, default=0.0, help="Weight for the eikonal regularizer.")
    parser.add_argument("--eikonal-target", type=float, default=1.0, help="Target gradient magnitude for eikonal loss.")
    parser.add_argument("--outside-weight", type=float, default=0.1, help="Relative weight for outside-mask loss.")
    parser.add_argument("--seed", type=int, default=42, help="Random seed.")
    parser.add_argument("--num-workers", type=int, default=4, help="Number of dataloader workers.")
    parser.add_argument("--device", type=str, default="", help="Torch device string, e.g. cuda:0 or cpu.")
    parser.add_argument("--base-channels", type=int, default=32, help="Base channel count for the ResUNet.")
    parser.add_argument("--num-stages", type=int, default=4, help="Number of encoder/decoder stages in ResUNet.")
    parser.add_argument("--output-dir", type=str, default="./StenUNet/pseudo_color/runs", help="Directory to store checkpoints.")
    parser.add_argument("--metrics-file", type=str, default="", help="Optional JSONL file to append metrics.")
    parser.add_argument("--val-interval", type=int, default=1, help="Validate every N epochs.")
    parser.add_argument("--amp", action="store_true", help="Use automatic mixed precision.")
    parser.add_argument("--export-preview", action="store_true", help="Save pseudo color previews for first val batch.")
    return parser


def main() -> None:
    parser = build_parser()
    args = parser.parse_args()
    train(args)


if __name__ == "__main__":
    main()
