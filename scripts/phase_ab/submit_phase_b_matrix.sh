#!/usr/bin/env bash
set -euo pipefail

if [[ $# -lt 3 ]]; then
    echo "Usage: bash submit_phase_b_matrix.sh DATASET_DIST DATASET_RADIUS DATASET_DIST_RADIUS [CONFIG]"
    echo "Example: bash submit_phase_b_matrix.sh 911 912 913 2d"
    exit 1
fi

DATASET_DIST="$1"
DATASET_RADIUS="$2"
DATASET_DIST_RADIUS="$3"
CONFIG="${4:-2d}"
TRAINER="${TRAINER:-nnUNetTrainer}"
PLANS="${PLANS:-nnUNetPlans}"

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
CONDA_ENV="${CONDA_ENV:-sten-nnunet}"
PARTITION="${PARTITION:-RTXA6Kq}"
GPUS="${GPUS:-1}"
CPUS="${CPUS:-8}"
MEM="${MEM:-64G}"
TIME_LIMIT="${TIME_LIMIT:-24:00:00}"
FEATURE_DEP_JOB_ID="${FEATURE_DEP_JOB_ID:-}"

RUN_STAMP="$(date +%Y%m%d_%H%M%S)"
JOB_RECORD="${ROOT_DIR}/logs/phase_b/jobs_${RUN_STAMP}.txt"
mkdir -p "$(dirname "${JOB_RECORD}")"

declare -A EXPERIMENT_DATASET=(
    [B1_dist]="${DATASET_DIST}"
    [B2_radius]="${DATASET_RADIUS}"
    [B3_dist_radius]="${DATASET_DIST_RADIUS}"
)

echo "phase=B config=${CONFIG}" | tee "${JOB_RECORD}"

for TAG in B1_dist B2_radius B3_dist_radius; do
    DATASET_ID="${EXPERIMENT_DATASET[${TAG}]}"
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

        if [[ -n "${FEATURE_DEP_JOB_ID}" ]]; then
            SBATCH_ARGS+=(--dependency="afterok:${FEATURE_DEP_JOB_ID}")
        fi

        JOB_ID=$(sbatch "${SBATCH_ARGS[@]}" \
            "${ROOT_DIR}/scripts/phase_ab/run_phase_b_fold.sbatch" \
            "${DATASET_ID}" "${CONFIG}" "${FOLD}" "${TAG}" "${TRAINER}" "${PLANS}")

        echo "tag=${TAG} dataset=${DATASET_ID} fold=${FOLD} job_id=${JOB_ID}" | tee -a "${JOB_RECORD}"
        echo "[INFO] Submitted ${TAG} fold ${FOLD} as job ${JOB_ID}"
    done
done

echo "[INFO] Job records saved to ${JOB_RECORD}"
