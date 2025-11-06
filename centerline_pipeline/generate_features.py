from __future__ import annotations

import argparse
import json
from pathlib import Path

import imageio.v2 as imageio
import numpy as np
from scipy.ndimage import distance_transform_edt


def load_mask(mask_path: Path) -> np.ndarray:
    """Load a binary vessel mask as uint8 (1=vessel, 0=background)."""
    mask = imageio.imread(mask_path)
    if mask.ndim == 3:
        mask = mask[..., 0]
    mask = (mask > 0).astype(np.uint8)
    return mask


def normalize_for_visualization(
    data: np.ndarray,
    valid_mask: np.ndarray,
    *,
    symmetric: bool = False,
) -> np.ndarray:
    """Normalize data into [0, 255] for visualization; background -> 0."""
    vis = np.zeros_like(data, dtype=np.float32)
    if not np.any(valid_mask):
        return vis.astype(np.uint8)

    values = data[valid_mask]
    if symmetric:
        max_val = float(np.max(np.abs(values)))
        if max_val == 0.0:
            scaled = np.full_like(values, 0.5, dtype=np.float32)
        else:
            scaled = 0.5 + 0.5 * (values / max_val)
    else:
        min_val = float(values.min())
        max_val = float(values.max())
        if np.isclose(max_val, min_val):
            scaled = np.zeros_like(values, dtype=np.float32)
        else:
            scaled = (values - min_val) / (max_val - min_val)

    vis[valid_mask] = np.clip(scaled, 0.0, 1.0)
    return (vis * 255).astype(np.uint8)


def compute_radius_rate(
    radii_along_centerline: np.ndarray,
    centerline_coords: np.ndarray,
) -> np.ndarray:
    """Compute signed radius change rate along a discretised centerline."""
    if radii_along_centerline.size <= 1:
        return np.zeros_like(radii_along_centerline, dtype=np.float32)

    diffs = np.diff(centerline_coords.astype(np.float64), axis=0)
    step_lengths = np.linalg.norm(diffs, axis=1)
    step_lengths[step_lengths == 0.0] = 1.0  # avoid division by zero for duplicate points

    dr = np.diff(radii_along_centerline.astype(np.float64))

    forward = np.zeros_like(radii_along_centerline, dtype=np.float64)
    forward[:-1] = dr / step_lengths
    forward[-1] = forward[-2] if forward.size > 1 else 0.0

    backward = np.zeros_like(radii_along_centerline, dtype=np.float64)
    backward[1:] = dr / step_lengths
    backward[0] = backward[1] if backward.size > 1 else 0.0

    rate = 0.5 * (forward + backward)
    # return rate.astype(np.float32)
    return forward.astype(np.float32)


def create_feature_maps(mask_path: Path, centerline_path: Path, output_dir: Path) -> None:
    """Generate three feature maps for a single case and save them to disk."""
    mask = load_mask(mask_path)

    with open(centerline_path, "r", encoding="utf-8") as f:
        centerline_data = json.load(f)

    centerline_coords = np.asarray(centerline_data.get("path", []), dtype=int)
    if centerline_coords.size == 0:
        print(f"[WARN] Centerline is empty for {centerline_path}. Skipping.")
        return

    height, width = mask.shape
    in_bounds = (
        (centerline_coords[:, 0] >= 0)
        & (centerline_coords[:, 0] < height)
        & (centerline_coords[:, 1] >= 0)
        & (centerline_coords[:, 1] < width)
    )
    if not np.all(in_bounds):
        centerline_coords = centerline_coords[in_bounds]
        if centerline_coords.size == 0:
            print(f"[WARN] All centerline points fall outside mask for {centerline_path}. Skipping.")
            return

    vessel_mask = mask.astype(bool)

    centerline_map = np.zeros_like(mask, dtype=bool)
    centerline_map[centerline_coords[:, 0], centerline_coords[:, 1]] = True

    dist_to_centerline, nearest_indices = distance_transform_edt(
        ~centerline_map,
        return_indices=True,
    )

    normalized_distance = np.full(mask.shape, -1.0, dtype=np.float32)
    if np.any(vessel_mask):
        vessel_distances = dist_to_centerline[vessel_mask]
        max_dist = float(vessel_distances.max(initial=0.0))
        if max_dist > 0.0:
            normalized_distance[vessel_mask] = (vessel_distances / max_dist).astype(np.float32)
        else:
            normalized_distance[vessel_mask] = 0.0

    dist_to_wall = distance_transform_edt(vessel_mask)
    radii_on_centerline = dist_to_wall[centerline_coords[:, 0], centerline_coords[:, 1]].astype(np.float32)

    radius_seed = np.zeros_like(mask, dtype=np.float32)
    radius_seed[centerline_coords[:, 0], centerline_coords[:, 1]] = radii_on_centerline
    radius_map = np.zeros_like(mask, dtype=np.float32)
    if np.any(vessel_mask):
        row_idx, col_idx = nearest_indices
        radius_map[vessel_mask] = radius_seed[row_idx[vessel_mask], col_idx[vessel_mask]]

# ----------------------------------------------

    rate_on_centerline = compute_radius_rate(radii_on_centerline, centerline_coords)
    rate_seed = np.zeros_like(mask, dtype=np.float32)
    rate_seed[centerline_coords[:, 0], centerline_coords[:, 1]] = rate_on_centerline
    rate_map = np.zeros_like(mask, dtype=np.float32)
    if np.any(vessel_mask):
        row_idx, col_idx = nearest_indices
        rate_map[vessel_mask] = rate_seed[row_idx[vessel_mask], col_idx[vessel_mask]]


    data_dir = output_dir / "data"
    data_dir.mkdir(parents=True, exist_ok=True)
    np.save(data_dir / "feature_normalized_distance.npy", normalized_distance)
    np.save(data_dir / "feature_radius.npy", radius_map)
    np.save(data_dir / "feature_radius_rate.npy", rate_map)

    dist_vis = normalize_for_visualization(normalized_distance, vessel_mask)
    radius_vis = normalize_for_visualization(radius_map, vessel_mask)
    rate_vis = normalize_for_visualization(rate_map, vessel_mask, symmetric=True)

    figure_dir = output_dir / "figures"
    figure_dir.mkdir(parents=True, exist_ok=True)
    imageio.imwrite(figure_dir / "feature_dist_transform.png", dist_vis)
    imageio.imwrite(figure_dir / "feature_radius_map.png", radius_vis)
    imageio.imwrite(figure_dir / "feature_rate_map.png", rate_vis)

    pseudo_color = np.stack([dist_vis, radius_vis, rate_vis], axis=-1)
    pseudo_color[~vessel_mask] = 0
    imageio.imwrite(figure_dir / "feature_pseudo_color.png", pseudo_color)

    print(f"[INFO] Saved feature maps for {centerline_path.parent.name} -> {output_dir}")


def resolve_mask_path(mask_dir: Path, centerline_json: Path) -> Path | None:
    """Infer the matching mask path for a given centerline json."""
    with open(centerline_json, "r", encoding="utf-8") as f:
        metadata = json.load(f)

    case_name = metadata.get("case_id")
    candidates = []
    if case_name:
        candidates.append(mask_dir / Path(case_name).name)
    stem = centerline_json.parent.name
    candidates.append(mask_dir / f"{stem}.png")
    candidates.append(mask_dir / stem)

    for candidate in candidates:
        if candidate.exists():
            return candidate
    return None


def process_all_cases(mask_dir: Path, output_dir: Path) -> None:
    """Generate feature maps for every case under output_dir."""
    json_paths = sorted(output_dir.glob("*/centerline.json"))
    if not json_paths:
        print(f"[WARN] No centerline.json files found in {output_dir}")
        return

    for centerline_json in json_paths:
        mask_path = resolve_mask_path(mask_dir, centerline_json)
        if mask_path is None:
            print(f"[WARN] Mask not found for {centerline_json}. Skipping.")
            continue
        create_feature_maps(mask_path, centerline_json, centerline_json.parent)


def main() -> None:
    parser = argparse.ArgumentParser(description="Generate centerline-derived feature maps.")
    parser.add_argument(
        "--mask-dir",
        type=Path,
        default=Path("/root/vessel/centerline/centerline_original/labelsTr"),
        help="Directory containing vessel mask PNGs.",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path("/root/vessel/centerline/centerline_original/output"),
        help="Directory containing centerline outputs.",
    )
    parser.add_argument(
        "--case-id",
        type=str,
        default=None,
        help="Optional case identifier (directory name) to process a single case.",
    )
    args = parser.parse_args()

    mask_dir = args.mask_dir
    output_dir = args.output_dir

    if args.case_id:
        case_stem = Path(args.case_id).stem
        centerline_json = output_dir / case_stem / "centerline.json"
        if not centerline_json.exists():
            print(f"[ERROR] centerline.json not found for case {case_stem} in {output_dir}.")
            return
        mask_path = resolve_mask_path(mask_dir, centerline_json)
        if mask_path is None:
            print(f"[ERROR] Mask not found for case {case_stem}.")
            return
        create_feature_maps(mask_path, centerline_json, centerline_json.parent)
    else:
        process_all_cases(mask_dir, output_dir)


if __name__ == "__main__":
    main()
