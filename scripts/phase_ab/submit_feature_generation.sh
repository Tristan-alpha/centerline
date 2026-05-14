#!/usr/bin/env bash
set -euo pipefail

MASK_DIR="${1:-/export/home3/dazhou/centerline/annotation/labelsTr_OR}"
ANNOTATIONS="${2:-/export/home3/dazhou/centerline/annotation/annotations_OR.json}"
OUT_DIR="${3:-/export/home3/dazhou/centerline/centerline/output_OR}"
LIMIT="${4:-}"

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
CONDA_ENV="${CONDA_ENV:-sten-nnunet}"
PARTITION="${PARTITION:-RTXA6Kq}"
CPUS="${CPUS:-8}"
MEM="${MEM:-32G}"
TIME_LIMIT="${TIME_LIMIT:-08:00:00}"

SBATCH_ARGS=(
    --parsable
    -p "${PARTITION}"
    --cpus-per-task="${CPUS}"
    --mem="${MEM}"
    -t "${TIME_LIMIT}"
    --export=ALL,ROOT_DIR="${ROOT_DIR}",CONDA_ENV="${CONDA_ENV}"
)

if [[ -n "${LIMIT}" ]]; then
    JOB_ID=$(sbatch "${SBATCH_ARGS[@]}" \
        "${ROOT_DIR}/scripts/phase_ab/run_feature_generation.sbatch" \
        "${MASK_DIR}" "${ANNOTATIONS}" "${OUT_DIR}" "${LIMIT}")
else
    JOB_ID=$(sbatch "${SBATCH_ARGS[@]}" \
        "${ROOT_DIR}/scripts/phase_ab/run_feature_generation.sbatch" \
        "${MASK_DIR}" "${ANNOTATIONS}" "${OUT_DIR}")
fi

echo "[INFO] Submitted feature generation as job ${JOB_ID}"
