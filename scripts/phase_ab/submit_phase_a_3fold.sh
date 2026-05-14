#!/usr/bin/env bash
set -euo pipefail

if [[ $# -lt 1 ]]; then
    echo "Usage: bash submit_phase_a_3fold.sh DATASET_ID [CONFIG] [TRAINER] [PLANS]"
    echo "Example: bash submit_phase_a_3fold.sh 901 2d"
    exit 1
fi

DATASET_ID="$1"
CONFIG="${2:-2d}"
TRAINER="${3:-nnUNetTrainer}"
PLANS="${4:-nnUNetPlans}"

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
CONDA_ENV="${CONDA_ENV:-sten-nnunet}"
PARTITION="${PARTITION:-RTXA6Kq}"
GPUS="${GPUS:-1}"
CPUS="${CPUS:-8}"
MEM="${MEM:-64G}"
TIME_LIMIT="${TIME_LIMIT:-24:00:00}"
PLAN_DEP_JOB_ID="${PLAN_DEP_JOB_ID:-}"

RUN_STAMP="$(date +%Y%m%d_%H%M%S)"
JOB_RECORD="${ROOT_DIR}/logs/phase_a/${DATASET_ID}/${CONFIG}/jobs_${RUN_STAMP}.txt"
mkdir -p "$(dirname "${JOB_RECORD}")"

echo "phase=A dataset=${DATASET_ID} config=${CONFIG}" | tee "${JOB_RECORD}"

for FOLD in 0 1 2; do
    SBATCH_ARGS=(
        --parsable
        -p "${PARTITION}"
        --gres="gpu:${GPUS}"
        --cpus-per-task="${CPUS}"
        --mem="${MEM}"
        -t "${TIME_LIMIT}"
        --export=ALL,ROOT_DIR="${ROOT_DIR}",CONDA_ENV="${CONDA_ENV}"
    )

    if [[ -n "${PLAN_DEP_JOB_ID}" ]]; then
        SBATCH_ARGS+=(--dependency="afterok:${PLAN_DEP_JOB_ID}")
    fi

    JOB_ID=$(sbatch "${SBATCH_ARGS[@]}" \
        "${ROOT_DIR}/scripts/phase_ab/run_phase_a_fold.sbatch" \
        "${DATASET_ID}" "${CONFIG}" "${FOLD}" "${TRAINER}" "${PLANS}")

    echo "fold=${FOLD} job_id=${JOB_ID}" | tee -a "${JOB_RECORD}"
    echo "[INFO] Submitted Phase A fold ${FOLD} as job ${JOB_ID}"
done

echo "[INFO] Job records saved to ${JOB_RECORD}"
