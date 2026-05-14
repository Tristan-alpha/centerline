#!/usr/bin/env bash
#SBATCH -J phaseA_nnunet
#SBATCH -p RTXA6Kq
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=64G
#SBATCH -t 24:00:00
#SBATCH -o /export/home3/dazhou/centerline/logs/slurm/%x_%j.out
#SBATCH -e /export/home3/dazhou/centerline/logs/slurm/%x_%j.err

set -euo pipefail

if [[ $# -lt 3 ]]; then
    echo "Usage: sbatch run_phase_a_fold.sbatch DATASET_ID CONFIG FOLD [TRAINER] [PLANS]"
    exit 1
fi

DATASET_ID="$1"
CONFIG="$2"
FOLD="$3"
TRAINER="${4:-nnUNetTrainer}"
PLANS="${5:-nnUNetPlans}"

ROOT_DIR="${ROOT_DIR:-/export/home3/dazhou/centerline}"
LOG_DIR="${ROOT_DIR}/logs/phase_a/${DATASET_ID}/${CONFIG}/fold${FOLD}"
mkdir -p "${LOG_DIR}" "${ROOT_DIR}/logs/slurm"

# shellcheck disable=SC1091
source "${ROOT_DIR}/scripts/phase_ab/common_env.sh"

bash "${ROOT_DIR}/scripts/phase_ab/precheck_gpu.sh" "${LOG_DIR}"

echo "[INFO] Starting Phase A training"
echo "[INFO] dataset=${DATASET_ID} config=${CONFIG} fold=${FOLD} trainer=${TRAINER} plans=${PLANS}"

nnUNetv2_train "${DATASET_ID}" "${CONFIG}" "${FOLD}" \
    -tr "${TRAINER}" -p "${PLANS}" -device cuda 2>&1 | tee "${LOG_DIR}/train.log"

echo "[INFO] Phase A fold completed at $(date -Iseconds)"
