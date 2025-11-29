from __future__ import annotations

import argparse
import json
from pathlib import Path

import imageio.v2 as imageio
import numpy as np
from matplotlib.colors import LinearSegmentedColormap
from scipy.ndimage import distance_transform_edt

DEFAULT_MASK_DIR = Path("annotation/labelsTr")
DEFAULT_OUTPUT_DIR = Path("centerline/output")

_STENOSIS_COLORMAP = LinearSegmentedColormap.from_list(
    "stenosis_rdylbu",
    [
        (0.0, "#b10026"),  # critical narrowing - saturated red
        (0.3, "#f46d43"),  # severe narrowing - orange
        (0.5, "#fee08b"),  # moderate - yellow
        (0.75, "#66c2a5"),  # mild - aqua
        (1.0, "#3288bd"),  # normal vessel - blue
    ],
)

try:
    # Register once so downstream scripts can reuse the same name.
    from matplotlib import cm

    register_cmap = getattr(cm, "register_cmap", None)
    if callable(register_cmap):
        register_cmap(name=_STENOSIS_COLORMAP.name, cmap=_STENOSIS_COLORMAP)
except (ValueError, ImportError):
    pass


def _coerce_centerline_array(raw) -> np.ndarray:
    arr = np.asarray(raw, dtype=int)
    if arr.size == 0:
        return np.empty((0, 2), dtype=int)
    if arr.ndim == 1:
        if arr.size % 2 != 0:
            return np.empty((0, 2), dtype=int)
        arr = arr.reshape(-1, 2)
    if arr.ndim != 2 or arr.shape[1] != 2:
        return np.empty((0, 2), dtype=int)
    return arr


def _collect_centerline_coords(centerline_data: dict) -> np.ndarray:
    paths = centerline_data.get("paths")
    coord_sets = []
    if isinstance(paths, list):
        for entry in paths:
            if isinstance(entry, dict):
                arr = _coerce_centerline_array(entry.get("path", []))
            else:
                arr = _coerce_centerline_array(entry)
            if arr.size:
                coord_sets.append(arr)
    if not coord_sets:
        arr = _coerce_centerline_array(centerline_data.get("path", []))
        if arr.size:
            coord_sets.append(arr)
    if not coord_sets:
        return np.empty((0, 2), dtype=int)
    return np.vstack(coord_sets)

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


def get_max_radius(mask_path: Path, centerline_path: Path) -> float:
    """Computes the maximum radius for a single case."""
    mask = load_mask(mask_path)
    with open(centerline_path, "r", encoding="utf-8") as f:
        centerline_data = json.load(f)

    centerline_coords = _collect_centerline_coords(centerline_data)
    if centerline_coords.size == 0:
        return 0.0

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
            return 0.0

    vessel_mask = mask.astype(bool)
    if not np.any(vessel_mask):
        return 0.0

    dist_to_wall = distance_transform_edt(vessel_mask)
    radii_on_centerline = dist_to_wall[centerline_coords[:, 0], centerline_coords[:, 1]]
    return float(radii_on_centerline.max(initial=0.0))


def create_feature_maps(
    mask_path: Path, centerline_path: Path, output_dir: Path, global_max_radius: float
) -> None:
    """Generate three feature maps for a single case and save them to disk."""
    mask = load_mask(mask_path)

    with open(centerline_path, "r", encoding="utf-8") as f:
        centerline_data = json.load(f)

    centerline_coords = _collect_centerline_coords(centerline_data)
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
            normalized_distance[vessel_mask] = 1.0 - (vessel_distances / max_dist).astype(np.float32)
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

    # Normalize radius map with global max radius
    normalized_radius_map = radius_map / global_max_radius if global_max_radius > 0 else radius_map

    # Create a "Glowing Tube" visualization:
    # 1. Hue/Color is determined by the radius (Red=Narrow, Blue=Wide)
    # 2. Brightness/Intensity is determined by the distance to centerline (Center=Bright, Wall=Dark)
    
    # Step 1: Generate base color from radius using the stenosis colormap
    radius_norm = np.clip(normalized_radius_map, 0.0, 1.0)
    # _STENOSIS_COLORMAP returns RGBA, we take RGB and scale to 0-255
    base_rgb = (_STENOSIS_COLORMAP(radius_norm)[..., :3] * 255).astype(np.float32)

    # Step 2: Generate intensity factor from distance
    # normalized_distance is -1.0 (bg), 0.0 (wall) to 1.0 (center)
    # We clip to [0, 1] so background and wall are dark, center is full brightness
    intensity = np.clip(normalized_distance, 0.0, 1.0)
    # Add a channel dimension to broadcast: (H, W) -> (H, W, 1)
    intensity = intensity[..., np.newaxis]

    # Step 3: Combine color and intensity
    # The result is a 3D tube effect where color indicates stenosis severity
    glowing_tube = base_rgb * intensity
    
    # Ensure background is strictly black (masking)
    glowing_tube[~vessel_mask] = 0
    
    # Normalize to 0-1 float for downstream training
    pseudo_color_float = glowing_tube / 255.0

    data_dir = output_dir / "data"
    data_dir.mkdir(parents=True, exist_ok=True)
    np.save(data_dir / "feature_normalized_distance.npy", normalized_distance)
    np.save(data_dir / "feature_radius.npy", normalized_radius_map)
    np.save(data_dir / "feature_pseudo_color.npy", pseudo_color_float.astype(np.float32))

    dist_vis = normalize_for_visualization(normalized_distance, vessel_mask)
    radius_vis = normalize_for_visualization(normalized_radius_map, vessel_mask)

    figure_dir = output_dir / "figures"
    figure_dir.mkdir(parents=True, exist_ok=True)
    imageio.imwrite(figure_dir / "feature_dist_transform.png", dist_vis)
    imageio.imwrite(figure_dir / "feature_radius_map.png", radius_vis)
    pseudo_color_png = (pseudo_color_float * 255.0).astype(np.uint8)
    imageio.imwrite(figure_dir / "feature_pseudo_color.png", pseudo_color_png)

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

    # First pass: compute global max radius
    max_radii = []
    for centerline_json in json_paths:
        mask_path = resolve_mask_path(mask_dir, centerline_json)
        if mask_path is None:
            print(f"[WARN] Mask not found for {centerline_json}. Skipping radius calculation.")
            continue
        max_radii.append(get_max_radius(mask_path, centerline_json))
    
    global_max_radius = max(max_radii) if max_radii else 0.0
    print(f"[INFO] Global max radius calculated: {global_max_radius}")

    # Second pass: generate feature maps
    for centerline_json in json_paths:
        mask_path = resolve_mask_path(mask_dir, centerline_json)
        if mask_path is None:
            print(f"[WARN] Mask not found for {centerline_json}. Skipping feature generation.")
            continue
        create_feature_maps(mask_path, centerline_json, centerline_json.parent, global_max_radius)


def main() -> None:
    parser = argparse.ArgumentParser(description="Generate centerline-derived feature maps.")
    parser.add_argument(
        "--mask-dir",
        type=Path,
        default=DEFAULT_MASK_DIR,
        help=f"Directory containing vessel mask PNGs (default: {DEFAULT_MASK_DIR}).",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=DEFAULT_OUTPUT_DIR,
        help=f"Directory containing centerline outputs (default: {DEFAULT_OUTPUT_DIR}).",
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
        # For a single case, the global max is just the case's max
        max_radius = get_max_radius(mask_path, centerline_json)
        create_feature_maps(mask_path, centerline_json, centerline_json.parent, max_radius)
    else:
        process_all_cases(mask_dir, output_dir)


if __name__ == "__main__":
    main()
