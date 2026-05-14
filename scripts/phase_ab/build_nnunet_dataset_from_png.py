#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import shutil
from pathlib import Path

import numpy as np
from PIL import Image


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Build an nnUNetv2 2D dataset from PNG image/label pairs with optional distance/radius channels."
    )
    parser.add_argument("--dataset-id", type=int, required=True, help="Dataset id, for example 901")
    parser.add_argument("--dataset-name", type=str, required=True, help="Dataset name suffix, for example VesselA")
    parser.add_argument("--images-dir", type=Path, required=True, help="Directory of base images (.png)")
    parser.add_argument("--labels-dir", type=Path, required=True, help="Directory of labels (.png)")
    parser.add_argument(
        "--case-list-json",
        type=Path,
        default=None,
        help="Optional annotations JSON. If set, only these label keys are used.",
    )
    parser.add_argument(
        "--raw-root",
        type=Path,
        default=Path("/export/home3/dazhou/centerline/nnUNet_raw"),
        help="nnUNet_raw root directory",
    )
    parser.add_argument(
        "--mode",
        type=str,
        default="image_only",
        choices=("image_only", "image_dist", "image_radius", "image_dist_radius"),
        help="Input channel mode",
    )
    parser.add_argument(
        "--features-root",
        type=Path,
        default=None,
        help="Root directory containing per-case feature files at <case>/data/*.npy",
    )
    parser.add_argument(
        "--label-name",
        type=str,
        default="sten",
        help="Foreground label name in dataset.json",
    )
    parser.add_argument(
        "--skip-missing-features",
        action="store_true",
        help="When mode uses features, skip cases that do not have complete feature files.",
    )
    parser.add_argument(
        "--fill-missing-features-zero",
        action="store_true",
        help="When mode uses features, fill missing distance/radius channels with zeros and keep the case.",
    )
    parser.add_argument("--overwrite", action="store_true", help="Overwrite existing DatasetXXX_* directory")
    return parser.parse_args()


def ensure_exists(path: Path, what: str) -> None:
    if not path.exists():
        raise FileNotFoundError(f"{what} not found: {path}")


def load_grayscale_image(path: Path) -> np.ndarray:
    arr = np.array(Image.open(path))
    if arr.ndim == 3:
        arr = arr[..., 0]
    return arr.astype(np.uint8)


def normalize_distance(dist_map: np.ndarray) -> np.ndarray:
    # distance map is expected in [-1, 1], where -1 is background.
    dist = np.clip((dist_map.astype(np.float32) + 1.0) / 2.0, 0.0, 1.0)
    return (dist * 255.0).astype(np.uint8)


def normalize_radius(radius_map: np.ndarray) -> np.ndarray:
    rad = np.clip(radius_map.astype(np.float32), 0.0, 1.0)
    return (rad * 255.0).astype(np.uint8)


def resolve_image_path(images_dir: Path, case_id: str) -> Path:
    candidates = [
        images_dir / f"{case_id}.png",
        images_dir / f"{case_id}_0000.png",
    ]
    for candidate in candidates:
        if candidate.exists():
            return candidate
    raise FileNotFoundError(
        f"Missing image for case {case_id}. Tried: {', '.join(str(c) for c in candidates)}"
    )


def collect_cases(
    images_dir: Path,
    labels_dir: Path,
    case_list_json: Path | None,
) -> list[tuple[str, Path, Path]]:
    selected_cases: set[str] | None = None
    if case_list_json is not None:
        with case_list_json.open("r", encoding="utf-8") as f:
            payload = json.load(f)
        if not isinstance(payload, dict):
            raise ValueError(f"Expected dict in case list json: {case_list_json}")
        selected_cases = {Path(k).stem for k in payload.keys()}

    labels = sorted(labels_dir.glob("*.png"))
    if not labels:
        raise RuntimeError(f"No .png labels found in {labels_dir}")

    cases: list[tuple[str, Path, Path]] = []
    for label_path in labels:
        case_id = label_path.stem
        if selected_cases is not None and case_id not in selected_cases:
            continue
        image_path = resolve_image_path(images_dir, case_id)
        cases.append((case_id, image_path, label_path))

    if selected_cases is not None and not cases:
        raise RuntimeError(
            "No cases were matched between labels and case-list-json. "
            "Please verify filenames in labels-dir and annotation keys."
        )

    return cases


def build_channel_map(mode: str) -> dict[str, str]:
    channel_names: dict[str, str] = {"0": "image"}
    if mode == "image_dist":
        channel_names["1"] = "distance"
    elif mode == "image_radius":
        channel_names["1"] = "radius"
    elif mode == "image_dist_radius":
        channel_names["1"] = "distance"
        channel_names["2"] = "radius"
    return channel_names


def main() -> None:
    args = parse_args()

    ensure_exists(args.images_dir, "images-dir")
    ensure_exists(args.labels_dir, "labels-dir")
    if args.case_list_json is not None:
        ensure_exists(args.case_list_json, "case-list-json")

    need_features = args.mode in {"image_dist", "image_radius", "image_dist_radius"}
    if need_features:
        if args.features_root is None:
            raise ValueError("--features-root is required when mode is not image_only")
        ensure_exists(args.features_root, "features-root")

    dataset_folder = args.raw_root / f"Dataset{args.dataset_id:03d}_{args.dataset_name}"
    images_tr = dataset_folder / "imagesTr"
    labels_tr = dataset_folder / "labelsTr"

    if dataset_folder.exists():
        if args.overwrite:
            shutil.rmtree(dataset_folder)
        else:
            raise FileExistsError(
                f"Dataset folder already exists: {dataset_folder}. Use --overwrite to replace it."
            )

    images_tr.mkdir(parents=True, exist_ok=True)
    labels_tr.mkdir(parents=True, exist_ok=True)

    cases = collect_cases(args.images_dir, args.labels_dir, args.case_list_json)

    written_cases = 0

    for case_id, image_path, label_path in cases:
        image = load_grayscale_image(image_path)
        # nnUNet expects class labels as integer ids (0/1), not 0/255 masks.
        label = (load_grayscale_image(label_path) > 0).astype(np.uint8)

        if image.shape != label.shape:
            raise ValueError(
                f"Shape mismatch for {case_id}: image {image.shape} vs label {label.shape}"
            )
        dist = None
        radius = None
        if need_features:
            feature_dir = args.features_root / case_id / "data"
            feature_dir_exists = feature_dir.exists()
            if not feature_dir_exists and not args.fill_missing_features_zero:
                if args.skip_missing_features:
                    print(f"[WARN] Missing feature dir for {case_id}, skipping case.")
                    continue
                ensure_exists(feature_dir, f"feature dir for case {case_id}")

            if args.mode in {"image_dist", "image_dist_radius"}:
                dist_path = feature_dir / "feature_normalized_distance.npy"
                if not dist_path.exists() and args.fill_missing_features_zero:
                    dist = np.full(image.shape, -1.0, dtype=np.float32)
                elif not dist_path.exists() and args.skip_missing_features:
                    print(f"[WARN] Missing distance feature for {case_id}, skipping case.")
                    continue
                else:
                    ensure_exists(dist_path, f"distance feature for case {case_id}")
                    dist = np.load(dist_path)
                if dist.shape != image.shape:
                    raise ValueError(
                        f"Distance shape mismatch for {case_id}: {dist.shape} vs {image.shape}"
                    )

            if args.mode in {"image_radius", "image_dist_radius"}:
                radius_path = feature_dir / "feature_radius.npy"
                if not radius_path.exists() and args.fill_missing_features_zero:
                    radius = np.zeros(image.shape, dtype=np.float32)
                elif not radius_path.exists() and args.skip_missing_features:
                    print(f"[WARN] Missing radius feature for {case_id}, skipping case.")
                    continue
                else:
                    ensure_exists(radius_path, f"radius feature for case {case_id}")
                    radius = np.load(radius_path)
                if radius.shape != image.shape:
                    raise ValueError(
                        f"Radius shape mismatch for {case_id}: {radius.shape} vs {image.shape}"
                    )

        Image.fromarray(image).save(images_tr / f"{case_id}_0000.png")
        Image.fromarray(label).save(labels_tr / f"{case_id}.png")

        if need_features:
            if args.mode in {"image_dist", "image_dist_radius"} and dist is not None:
                Image.fromarray(normalize_distance(dist)).save(images_tr / f"{case_id}_0001.png")

            if args.mode in {"image_radius", "image_dist_radius"} and radius is not None:
                channel_idx = 1 if args.mode == "image_radius" else 2
                Image.fromarray(normalize_radius(radius)).save(
                    images_tr / f"{case_id}_{channel_idx:04d}.png"
                )
        written_cases += 1

    dataset_json = {
        "channel_names": build_channel_map(args.mode),
        "labels": {"background": 0, args.label_name: 1},
        "numTraining": written_cases,
        "file_ending": ".png",
    }

    with (dataset_folder / "dataset.json").open("w", encoding="utf-8") as f:
        json.dump(dataset_json, f, indent=2)

    print(f"[INFO] Built dataset: {dataset_folder}")
    print(f"[INFO] Cases written: {written_cases} / {len(cases)}")
    print(f"[INFO] Mode: {args.mode}")


if __name__ == "__main__":
    main()
