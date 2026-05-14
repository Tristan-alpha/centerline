#!/usr/bin/env bash
#SBATCH -J plan_pp
#SBATCH -p RTXA6Kq
#SBATCH --cpus-per-task=8
#SBATCH --mem=64G
#SBATCH -t 08:00:00
#SBATCH -o /export/home3/dazhou/centerline/logs/slurm/%x_%j.out
#SBATCH -e /export/home3/dazhou/centerline/logs/slurm/%x_%j.err

set -euo pipefail

if [[ $# -lt 1 ]]; then
    echo "Usage: sbatch run_plan_preprocess.sbatch DATASET_ID [CONFIG] [NPFP] [NP]"
    exit 1
fi

DATASET_ID="$1"
CONFIG="${2:-2d}"
NPFP="${3:-8}"
NP="${4:-8}"

ROOT_DIR="${ROOT_DIR:-/export/home3/dazhou/centerline}"
LOG_DIR="${ROOT_DIR}/logs/plan_preprocess/dataset_${DATASET_ID}/${CONFIG}"
mkdir -p "${LOG_DIR}" "${ROOT_DIR}/logs/slurm"

# shellcheck disable=SC1091
source "${ROOT_DIR}/scripts/phase_ab/common_env.sh"

echo "[INFO] Running plan+preprocess for dataset ${DATASET_ID}, config=${CONFIG}"

nnUNetv2_plan_and_preprocess \
    -d "${DATASET_ID}" \
    --verify_dataset_integrity \
    -c "${CONFIG}" \
    -npfp "${NPFP}" \
    -np "${NP}" \
    --verbose 2>&1 | tee "${LOG_DIR}/plan_preprocess.log"

echo "[INFO] plan+preprocess completed at $(date -Iseconds)"
