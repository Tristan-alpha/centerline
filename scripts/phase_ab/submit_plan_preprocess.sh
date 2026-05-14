#!/usr/bin/env bash
set -euo pipefail

if [[ $# -lt 1 ]]; then
    echo "Usage: bash submit_plan_preprocess.sh DATASET_ID [CONFIG] [NPFP] [NP]"
    echo "Example: bash submit_plan_preprocess.sh 901 2d"
    exit 1
fi

DATASET_ID="$1"
CONFIG="${2:-2d}"
NPFP="${3:-8}"
NP="${4:-8}"

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
CONDA_ENV="${CONDA_ENV:-sten-nnunet}"
PARTITION="${PARTITION:-RTXA6Kq}"
CPUS="${CPUS:-8}"
MEM="${MEM:-64G}"
TIME_LIMIT="${TIME_LIMIT:-08:00:00}"

JOB_ID=$(sbatch --parsable \
    -p "${PARTITION}" \
    --cpus-per-task="${CPUS}" \
    --mem="${MEM}" \
    -t "${TIME_LIMIT}" \
    --export=ALL,ROOT_DIR="${ROOT_DIR}",CONDA_ENV="${CONDA_ENV}" \
    "${ROOT_DIR}/scripts/phase_ab/run_plan_preprocess.sbatch" \
    "${DATASET_ID}" "${CONFIG}" "${NPFP}" "${NP}")

echo "[INFO] Submitted plan+preprocess as job ${JOB_ID}"
