#!/usr/bin/env bash
#SBATCH -J featgen_centerline
#SBATCH -p RTXA6Kq
#SBATCH --cpus-per-task=8
#SBATCH --mem=32G
#SBATCH -t 08:00:00
#SBATCH -o /export/home3/dazhou/centerline/logs/slurm/%x_%j.out
#SBATCH -e /export/home3/dazhou/centerline/logs/slurm/%x_%j.err

set -euo pipefail

if [[ $# -lt 3 ]]; then
    echo "Usage: sbatch run_feature_generation.sbatch MASK_DIR ANNOTATIONS OUT_DIR [LIMIT]"
    exit 1
fi

MASK_DIR="$1"
ANNOTATIONS="$2"
OUT_DIR="$3"
LIMIT="${4:-}"

ROOT_DIR="${ROOT_DIR:-/export/home3/dazhou/centerline}"
CONDA_ENV="${CONDA_ENV:-sten-nnunet}"
LOG_DIR="${ROOT_DIR}/logs/feature_generation"
mkdir -p "${LOG_DIR}" "${ROOT_DIR}/logs/slurm"

# shellcheck disable=SC1091
source "${ROOT_DIR}/scripts/phase_ab/common_env.sh"

cd "${ROOT_DIR}/centerline"

CMD=(python -m centerline_pipeline --mask-dir "${MASK_DIR}" --annotations "${ANNOTATIONS}" --out "${OUT_DIR}")
if [[ -n "${LIMIT}" ]]; then
    CMD+=(--limit "${LIMIT}")
fi

echo "[INFO] Running centerline extraction: ${CMD[*]}"
"${CMD[@]}" 2>&1 | tee "${LOG_DIR}/centerline_pipeline_$(date +%Y%m%d_%H%M%S).log"

echo "[INFO] Running feature map generation"
python centerline_pipeline/generate_features.py --mask-dir "${MASK_DIR}" --output-dir "${OUT_DIR}" \
    2>&1 | tee "${LOG_DIR}/generate_features_$(date +%Y%m%d_%H%M%S).log"

echo "[INFO] Feature generation completed at $(date -Iseconds)"
