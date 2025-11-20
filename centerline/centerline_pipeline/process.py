from __future__ import annotations

import argparse
import logging
from pathlib import Path

# from centerline.centerline_pipeline import distance, io, metrics, path, preprocess, visualization
from . import distance, io, metrics, path, preprocess, visualization

# Get the root directory (two levels up from this file)
_ROOT_DIR = Path(__file__).resolve().parent.parent.parent
DEFAULT_MASK_DIR = _ROOT_DIR / "annotation" / "labelsTr"
DEFAULT_ANNOTATIONS = _ROOT_DIR / "annotation" / "annotations.json"
DEFAULT_OUTPUT_DIR = _ROOT_DIR / "centerline" / "output"


def process_case(
    case: io.CaseData,
    output_root: Path,
    *,
    k: float,
    connectivity: int,
    snap_points: bool,
) -> None:
    mask = io.load_mask(case.mask_path)
    case_dir = output_root / Path(case.case_id).stem
    io.ensure_dir(case_dir)

    payload_paths = []

    for pair in case.pairs:
        prepared = preprocess.prepare_case(mask, pair.start, pair.end, snap=snap_points)

        for key, dist in prepared.snaps.items():
            logging.info(
                "[%s][pair %d] %s point snapped %.2f px onto vessel",
                case.case_id,
                pair.index + 1,
                key,
                dist,
            )

        if prepared.dilations:
            logging.info(
                "[%s][pair %d] applied %d dilation steps to connect mask",
                case.case_id,
                pair.index + 1,
                prepared.dilations,
            )

        distance_map, cost_map = distance.compute_distance_and_cost(prepared.mask, k=k)
        centerline = path.shortest_path(cost_map, prepared.start, prepared.end, connectivity=connectivity)

        metric = metrics.compute_metrics(distance_map, centerline)

        overlay = visualization.render_overlay(mask, centerline, prepared.start, prepared.end)
        overlay_path = case_dir / f"overlay_pair{pair.index + 1}.png"
        overlay.save(overlay_path)

        rows = [
            ("index", "row", "col", "radius_px", "diameter_px", "stenosis"),
        ]
        for idx, ((row, col), radius, diameter, stenosis_val) in enumerate(
            zip(centerline, metric.radii, metric.diameters, metric.stenosis_curve)
        ):
            rows.append((idx, int(row), int(col), float(radius), float(diameter), float(stenosis_val)))
        metrics_csv = case_dir / f"metrics_pair{pair.index + 1}.csv"
        io.save_csv(metrics_csv, rows)
        if pair.index == 0:
            io.save_csv(case_dir / "metrics.csv", rows)

        payload_paths.append(
            {
                "pair_index": pair.index,
                "annotation": {
                    "start_xy": list(map(int, pair.start_xy)),
                    "end_xy": list(map(int, pair.end_xy)),
                },
                "start": list(map(int, prepared.start)),
                "end": list(map(int, prepared.end)),
                "path": centerline.astype(int).tolist(),
                "max_stenosis": metric.max_stenosis,
                "max_stenosis_index": metric.max_stenosis_index,
                "path_length_px": metric.path_length,
                "dilation_steps": prepared.dilations,
                "k": k,
                "snap_distances_px": {k: float(v) for k, v in prepared.snaps.items()},
                "overlay": overlay_path.name,
                "metrics": metrics_csv.name,
            }
        )

        logging.info(
            "[%s][pair %d] centerline points: %d, max stenosis: %.3f at index %d",
            case.case_id,
            pair.index + 1,
            len(centerline),
            metric.max_stenosis,
            metric.max_stenosis_index,
        )

    json_payload = {
        "case_id": case.case_id,
        "mask_path": str(case.mask_path),
        "pair_count": len(payload_paths),
        "paths": payload_paths,
    }
    if payload_paths:
        # Populate legacy keys with the first path for backward compatibility.
        legacy = payload_paths[0]
        json_payload.update(
            {
                "start": legacy["start"],
                "end": legacy["end"],
                "path": legacy["path"],
                "max_stenosis": legacy["max_stenosis"],
                "max_stenosis_index": legacy["max_stenosis_index"],
                "path_length_px": legacy["path_length_px"],
                "dilation_steps": legacy["dilation_steps"],
                "k": legacy["k"],
                "snap_distances_px": legacy["snap_distances_px"],
            }
        )
    io.save_json(case_dir / "centerline.json", json_payload)

    if payload_paths:
        combined_overlay = visualization.render_multi_overlay(mask, payload_paths)
        combined_overlay.save(case_dir / "overlay.png")


def build_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Process 2-D vessel masks to extract centerlines.")
    parser.add_argument(
        "--mask-dir",
        type=Path,
        default=DEFAULT_MASK_DIR,
        help=f"Directory containing PNG masks (default: {DEFAULT_MASK_DIR}).",
    )
    parser.add_argument(
        "--annotations",
        type=Path,
        default=DEFAULT_ANNOTATIONS,
        help=f"annotations.json file (default: {DEFAULT_ANNOTATIONS}).",
    )
    parser.add_argument(
        "--out",
        type=Path,
        default=DEFAULT_OUTPUT_DIR,
        help=f"Output directory for results (default: {DEFAULT_OUTPUT_DIR}).",
    )
    parser.add_argument("--k", type=float, default=4.0, help="Exponential scaling factor for cost volume.")
    parser.add_argument(
        "--connectivity",
        type=int,
        default=8,
        choices=(4, 8),
        help="Pixel connectivity used for the shortest-path graph.",
    )
    parser.add_argument(
        "--no-snap",
        action="store_true",
        help="Disable snapping annotated start/end points onto the vessel mask.",
    )
    parser.add_argument(
        "--limit",
        type=int,
        default=None,
        help="Only process the first N cases (useful for debugging).",
    )
    parser.add_argument(
        "--log-level",
        default="INFO",
        choices=("DEBUG", "INFO", "WARNING", "ERROR"),
        help="Logging verbosity.",
    )
    return parser


def main(argv: list[str] | None = None) -> None:
    parser = build_arg_parser()
    args = parser.parse_args(argv)

    logging.basicConfig(level=getattr(logging, args.log_level), format="%(message)s")

    annotations = io.load_annotations(args.annotations)
    output_root = args.out
    io.ensure_dir(output_root)

    snap_points = not args.no_snap
    processed = 0
    for case in io.iter_cases(args.mask_dir, annotations):
        process_case(
            case,
            output_root=output_root,
            k=args.k,
            connectivity=args.connectivity,
            snap_points=snap_points,
        )
        processed += 1
        if args.limit is not None and processed >= args.limit:
            break

    logging.info("Processed %d cases into %s", processed, output_root)


if __name__ == "__main__":
    main()
