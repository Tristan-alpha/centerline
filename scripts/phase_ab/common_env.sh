#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="${ROOT_DIR:-/export/home3/dazhou/centerline}"
CONDA_ENV="${CONDA_ENV:-sten-nnunet}"

export nnUNet_raw="${nnUNet_raw:-${ROOT_DIR}/nnUNet_raw}"
export nnUNet_preprocessed="${nnUNet_preprocessed:-${ROOT_DIR}/nnUNet_preprocessed}"
export nnUNet_results="${nnUNet_results:-${ROOT_DIR}/nnUNet_results}"

# torch.compile/inductor can be unstable on shared clusters due to cache races.
# Keep it disabled by default unless explicitly overridden by caller.
export nnUNet_compile="${nnUNet_compile:-false}"

mkdir -p "${nnUNet_raw}" "${nnUNet_preprocessed}" "${nnUNet_results}"

if [[ -f "${HOME}/.bashrc" ]]; then
    # shellcheck disable=SC1090
    source "${HOME}/.bashrc"
fi

if command -v conda >/dev/null 2>&1; then
    eval "$(conda shell.bash hook)"
    conda activate "${CONDA_ENV}"
else
    echo "[ERROR] conda is not available in PATH."
    exit 1
fi

# Make centerline_pipeline importable when running from workspace root scripts.
export PYTHONPATH="${ROOT_DIR}/centerline:${PYTHONPATH:-}"

echo "[INFO] ROOT_DIR=${ROOT_DIR}"
echo "[INFO] CONDA_ENV=${CONDA_ENV}"
echo "[INFO] nnUNet_raw=${nnUNet_raw}"
echo "[INFO] nnUNet_preprocessed=${nnUNet_preprocessed}"
echo "[INFO] nnUNet_results=${nnUNet_results}"
echo "[INFO] nnUNet_compile=${nnUNet_compile}"
