#!/usr/bin/env python3
from __future__ import annotations

import argparse
import csv
import glob
import json
import re
import statistics
import subprocess
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Dict, Iterable, List, Optional, Tuple


EXPERIMENT_ORDER = ["A_baseline", "B1_dist", "B2_radius", "B3_dist_radius"]


@dataclass
class FoldRun:
    phase: str
    tag: str
    dataset_id: str
    config: str
    fold: int
    job_id: str
    state: str = "UNKNOWN"
    exit_code: str = "NA"
    dataset_name: str = "NA"
    log_path: str = "NA"
    last_epoch_seen: Optional[int] = None
    latest_dice_epoch: Optional[int] = None
    latest_dice: Optional[float] = None
    best_dice_epoch: Optional[int] = None
    best_dice: Optional[float] = None
    n_dice_points: int = 0


def run_cmd(args: List[str]) -> str:
    proc = subprocess.run(args, stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True)
    if proc.returncode != 0:
        return ""
    return proc.stdout


def parse_kv_tokens(line: str) -> Dict[str, str]:
    out: Dict[str, str] = {}
    for token in line.strip().split():
        if "=" not in token:
            continue
        k, v = token.split("=", 1)
        out[k.strip()] = v.strip()
    return out


def pick_latest_file(pattern: str) -> Optional[Path]:
    candidates = [Path(p) for p in glob.glob(pattern)]
    if not candidates:
        return None
    return max(candidates, key=lambda p: p.stat().st_mtime)


def normalize_record_path(path: str | Path, root_dir: Path) -> str:
    p = Path(path)
    if p.is_absolute():
        return str(p.resolve())
    return str((root_dir / p).resolve())


def load_phase_a_jobs(job_record: Path) -> List[FoldRun]:
    runs: List[FoldRun] = []
    dataset_id = ""
    config = "2d"
    with job_record.open("r", encoding="utf-8") as f:
        for raw_line in f:
            line = raw_line.strip()
            if not line:
                continue
            if line.startswith("phase=A"):
                kv = parse_kv_tokens(line)
                dataset_id = kv.get("dataset", dataset_id)
                config = kv.get("config", config)
                continue
            if line.startswith("fold="):
                kv = parse_kv_tokens(line)
                if "fold" in kv and "job_id" in kv:
                    runs.append(
                        FoldRun(
                            phase="A",
                            tag="A_baseline",
                            dataset_id=dataset_id,
                            config=config,
                            fold=int(kv["fold"]),
                            job_id=kv["job_id"],
                        )
                    )
    return runs


def load_phase_b_jobs(job_record: Path) -> Tuple[List[FoldRun], str]:
    runs: List[FoldRun] = []
    config = "2d"
    with job_record.open("r", encoding="utf-8") as f:
        for raw_line in f:
            line = raw_line.strip()
            if not line:
                continue
            if line.startswith("phase=B"):
                kv = parse_kv_tokens(line)
                config = kv.get("config", config)
                continue
            if line.startswith("tag="):
                kv = parse_kv_tokens(line)
                if {"tag", "dataset", "fold", "job_id"}.issubset(kv):
                    runs.append(
                        FoldRun(
                            phase="B",
                            tag=kv["tag"],
                            dataset_id=kv["dataset"],
                            config=config,
                            fold=int(kv["fold"]),
                            job_id=kv["job_id"],
                        )
                    )
    return runs, config


def apply_retry_records(
    runs: List[FoldRun],
    retry_records: Iterable[Path],
    root_dir: Path,
    source_job_record: Path,
) -> None:
    index: Dict[Tuple[str, str, int], FoldRun] = {(r.tag, r.dataset_id, r.fold): r for r in runs}
    source_job_record_abs = str(source_job_record.resolve())

    for record in sorted(retry_records, key=lambda p: p.stat().st_mtime):
        source_ok = False
        with record.open("r", encoding="utf-8") as f:
            for raw_line in f:
                line = raw_line.strip()
                if not line:
                    continue
                if line.startswith("source_job_record="):
                    source = line.split("=", 1)[1].strip()
                    source_abs = normalize_record_path(source, root_dir)
                    source_ok = source_abs == source_job_record_abs
                    continue
                if not source_ok or not line.startswith("retry "):
                    continue

                kv = parse_kv_tokens(line.replace("retry ", "", 1))
                if {"tag", "dataset", "fold", "new_job_id"}.issubset(kv):
                    key = (kv["tag"], kv["dataset"], int(kv["fold"]))
                    if key in index:
                        index[key].job_id = kv["new_job_id"]


def query_job_states(job_ids: List[str]) -> Dict[str, Tuple[str, str]]:
    state_map: Dict[str, Tuple[str, str]] = {}
    if not job_ids:
        return state_map

    sacct_out = run_cmd(
        [
            "sacct",
            "-j",
            ",".join(job_ids),
            "--format=JobIDRaw,State,ExitCode",
            "-n",
            "-P",
        ]
    )

    for line in sacct_out.splitlines():
        parts = line.strip().split("|")
        if len(parts) < 3:
            continue
        job_id_raw, state, exit_code = parts[0].strip(), parts[1].strip(), parts[2].strip()
        if not job_id_raw or "." in job_id_raw:
            continue
        state_clean = state.split("+")[0] if state else "UNKNOWN"
        state_map[job_id_raw] = (state_clean or "UNKNOWN", exit_code or "NA")

    unresolved = [j for j in job_ids if j not in state_map]
    if unresolved:
        squeue_out = run_cmd(["squeue", "-h", "-j", ",".join(unresolved), "-o", "%i|%T"])
        for line in squeue_out.splitlines():
            parts = line.strip().split("|")
            if len(parts) != 2:
                continue
            jid, st = parts[0].strip(), parts[1].strip()
            if jid:
                state_map[jid] = (st.upper(), "NA")

    return state_map


def parse_training_log(log_path: Path) -> Dict[str, Optional[float] | Optional[int] | int]:
    epoch_re = re.compile(r"Epoch\s+(\d+)")
    dice_re = re.compile(r"Pseudo dice\s+\[(.*)\]")

    last_epoch_seen: Optional[int] = None
    current_epoch: Optional[int] = None
    points: List[Tuple[int, float]] = []

    with log_path.open("r", encoding="utf-8", errors="ignore") as f:
        for raw_line in f:
            line = raw_line.strip()
            m_epoch = epoch_re.search(line)
            if m_epoch:
                current_epoch = int(m_epoch.group(1))
                last_epoch_seen = current_epoch
                continue

            m_dice = dice_re.search(line)
            if not m_dice:
                continue

            inside = m_dice.group(1)
            cleaned = inside.replace("np.float32(", "").replace(")", "")
            values: List[float] = []
            for token in cleaned.split(","):
                token = token.strip()
                if not token:
                    continue
                try:
                    values.append(float(token))
                except ValueError:
                    pass

            if values:
                epoch = current_epoch if current_epoch is not None else -1
                points.append((epoch, float(sum(values) / len(values))))

    if not points:
        return {
            "last_epoch_seen": last_epoch_seen,
            "latest_dice_epoch": None,
            "latest_dice": None,
            "best_dice_epoch": None,
            "best_dice": None,
            "n_dice_points": 0,
        }

    latest_epoch, latest_dice = points[-1]
    best_epoch, best_dice = max(points, key=lambda x: x[1])

    return {
        "last_epoch_seen": last_epoch_seen,
        "latest_dice_epoch": latest_epoch,
        "latest_dice": latest_dice,
        "best_dice_epoch": best_epoch,
        "best_dice": best_dice,
        "n_dice_points": len(points),
    }


def resolve_latest_training_log(root_dir: Path, dataset_id: str, fold: int) -> Optional[Path]:
    pattern = str(
        root_dir
        / "nnUNet_results"
        / f"Dataset{dataset_id}_*"
        / "*"
        / f"fold_{fold}"
        / "training_log_*.txt"
    )
    candidates = [Path(p) for p in glob.glob(pattern)]
    if not candidates:
        return None
    return max(candidates, key=lambda p: p.stat().st_mtime)


def with_metrics(root_dir: Path, runs: List[FoldRun]) -> None:
    states = query_job_states([r.job_id for r in runs])

    for r in runs:
        state, exit_code = states.get(r.job_id, ("UNKNOWN", "NA"))
        r.state = state
        r.exit_code = exit_code

        log_path = resolve_latest_training_log(root_dir, r.dataset_id, r.fold)
        if log_path is None:
            continue

        r.log_path = str(log_path.relative_to(root_dir))
        dataset_match = re.search(r"(Dataset\d+_[^/]+)", r.log_path)
        if dataset_match:
            r.dataset_name = dataset_match.group(1)

        metrics = parse_training_log(log_path)
        r.last_epoch_seen = metrics["last_epoch_seen"]  # type: ignore[assignment]
        r.latest_dice_epoch = metrics["latest_dice_epoch"]  # type: ignore[assignment]
        r.latest_dice = metrics["latest_dice"]  # type: ignore[assignment]
        r.best_dice_epoch = metrics["best_dice_epoch"]  # type: ignore[assignment]
        r.best_dice = metrics["best_dice"]  # type: ignore[assignment]
        r.n_dice_points = int(metrics["n_dice_points"])


def fmt_float(v: Optional[float]) -> str:
    return "NA" if v is None else f"{v:.4f}"


def fmt_int(v: Optional[int]) -> str:
    return "NA" if v is None else str(v)


def write_outputs(root_dir: Path, runs: List[FoldRun], output_dir: Path) -> None:
    output_dir.mkdir(parents=True, exist_ok=True)

    runs_sorted = sorted(
        runs,
        key=lambda r: (
            EXPERIMENT_ORDER.index(r.tag) if r.tag in EXPERIMENT_ORDER else 99,
            r.fold,
        ),
    )

    csv_path = output_dir / "fold_metrics.csv"
    with csv_path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)
        writer.writerow(
            [
                "phase",
                "tag",
                "dataset_id",
                "dataset_name",
                "config",
                "fold",
                "job_id",
                "state",
                "exit_code",
                "last_epoch_seen",
                "latest_dice_epoch",
                "latest_dice",
                "best_dice_epoch",
                "best_dice",
                "n_dice_points",
                "log_path",
            ]
        )
        for r in runs_sorted:
            writer.writerow(
                [
                    r.phase,
                    r.tag,
                    r.dataset_id,
                    r.dataset_name,
                    r.config,
                    r.fold,
                    r.job_id,
                    r.state,
                    r.exit_code,
                    fmt_int(r.last_epoch_seen),
                    fmt_int(r.latest_dice_epoch),
                    fmt_float(r.latest_dice),
                    fmt_int(r.best_dice_epoch),
                    fmt_float(r.best_dice),
                    r.n_dice_points,
                    r.log_path,
                ]
            )

    by_tag: Dict[str, List[FoldRun]] = {}
    for r in runs_sorted:
        by_tag.setdefault(r.tag, []).append(r)

    summary = []
    for tag in EXPERIMENT_ORDER:
        rows = by_tag.get(tag, [])
        if not rows:
            continue
        best_vals = [r.best_dice for r in rows if r.best_dice is not None]
        latest_vals = [r.latest_dice for r in rows if r.latest_dice is not None]
        summary.append(
            {
                "tag": tag,
                "folds_total": len(rows),
                "folds_with_metrics": len(best_vals),
                "mean_best_dice": float(statistics.mean(best_vals)) if best_vals else None,
                "mean_latest_dice": float(statistics.mean(latest_vals)) if latest_vals else None,
                "states": ",".join(sorted({r.state for r in rows})),
            }
        )

    json_path = output_dir / "latest_metrics.json"
    with json_path.open("w", encoding="utf-8") as f:
        json.dump(
            {
                "fold_metrics": [asdict(r) for r in runs_sorted],
                "summary": summary,
            },
            f,
            indent=2,
            ensure_ascii=True,
        )

    md_path = output_dir / "comparison_table.md"
    lines: List[str] = []
    lines.append("# Phase A/B Interim Metrics")
    lines.append("")
    lines.append("## Fold-level")
    lines.append("")
    lines.append(
        "| Tag | Dataset | Fold | Job | State | Last Epoch | Latest Dice | Best Dice | Log |"
    )
    lines.append("|---|---:|---:|---:|---|---:|---:|---:|---|")

    for r in runs_sorted:
        lines.append(
            "| {tag} | {dataset} | {fold} | {job} | {state} | {epoch} | {latest} | {best} | {log} |".format(
                tag=r.tag,
                dataset=r.dataset_id,
                fold=r.fold,
                job=r.job_id,
                state=r.state,
                epoch=fmt_int(r.last_epoch_seen),
                latest=fmt_float(r.latest_dice),
                best=fmt_float(r.best_dice),
                log=r.log_path,
            )
        )

    lines.append("")
    lines.append("## Experiment-level")
    lines.append("")
    lines.append("| Tag | Folds with Metrics | Mean Best Dice | Mean Latest Dice | States |")
    lines.append("|---|---:|---:|---:|---|")
    for row in summary:
        lines.append(
            "| {tag} | {fw}/{ft} | {mb} | {ml} | {states} |".format(
                tag=row["tag"],
                fw=row["folds_with_metrics"],
                ft=row["folds_total"],
                mb=fmt_float(row["mean_best_dice"]),
                ml=fmt_float(row["mean_latest_dice"]),
                states=row["states"],
            )
        )

    md_path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def main() -> None:
    parser = argparse.ArgumentParser(description="Collect Phase A/B nnUNet fold metrics.")
    parser.add_argument("--root-dir", default="/export/home3/dazhou/centerline")
    parser.add_argument("--phase-a-job-record", default="")
    parser.add_argument("--phase-b-job-record", default="")
    parser.add_argument("--output-dir", default="")
    args = parser.parse_args()

    root_dir = Path(args.root_dir).resolve()

    if args.phase_a_job_record:
        phase_a_record = Path(normalize_record_path(args.phase_a_job_record, root_dir))
    else:
        phase_a_record = pick_latest_file(str(root_dir / "logs/phase_a/*/*/jobs_*.txt"))

    if args.phase_b_job_record:
        phase_b_record = Path(normalize_record_path(args.phase_b_job_record, root_dir))
    else:
        phase_b_record = pick_latest_file(str(root_dir / "logs/phase_b/jobs_*.txt"))

    if phase_a_record is None or not phase_a_record.exists():
        raise SystemExit("[ERROR] Phase A job record not found")
    if phase_b_record is None or not phase_b_record.exists():
        raise SystemExit("[ERROR] Phase B job record not found")

    runs_a = load_phase_a_jobs(phase_a_record)
    runs_b, _ = load_phase_b_jobs(phase_b_record)

    retry_records = [Path(p) for p in glob.glob(str(root_dir / "logs/phase_b/retry_jobs_*.txt"))]
    apply_retry_records(runs_b, retry_records, root_dir, phase_b_record)

    all_runs = runs_a + runs_b
    with_metrics(root_dir, all_runs)

    output_dir = Path(args.output_dir) if args.output_dir else root_dir / "logs/phase_ab_metrics"
    if not output_dir.is_absolute():
        output_dir = root_dir / output_dir

    write_outputs(root_dir, all_runs, output_dir)

    print(f"[INFO] Phase A record: {phase_a_record}")
    print(f"[INFO] Phase B record: {phase_b_record}")
    print(f"[INFO] Output dir: {output_dir}")
    print(f"[INFO] Fold rows: {len(all_runs)}")


if __name__ == "__main__":
    main()
