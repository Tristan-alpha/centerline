#!/usr/bin/env bash
set -euo pipefail

ENV_NAME="${1:-sten-nnunet}"
PYTHON_VERSION="${PYTHON_VERSION:-3.10}"

if [[ -f "${HOME}/.bashrc" ]]; then
    # shellcheck disable=SC1090
    source "${HOME}/.bashrc"
fi

if ! command -v conda >/dev/null 2>&1; then
    echo "[ERROR] conda is not available in PATH."
    exit 1
fi

eval "$(conda shell.bash hook)"

if conda env list | awk '{print $1}' | grep -qx "${ENV_NAME}"; then
    echo "[INFO] Conda env ${ENV_NAME} already exists. Skipping creation."
else
    echo "[INFO] Creating conda env ${ENV_NAME} with Python ${PYTHON_VERSION}."
    conda create -y -n "${ENV_NAME}" "python=${PYTHON_VERSION}"
fi

conda activate "${ENV_NAME}"

python -m pip install --upgrade pip setuptools wheel
python -m pip install --index-url https://download.pytorch.org/whl/cu121 torch torchvision
python -m pip install nnunetv2 scipy scikit-image pandas imageio matplotlib nibabel \
    SimpleITK connected-components-3d opencv-python-headless scikit-learn tqdm \
    seaborn yacs shapely typer PyYAML batchgenerators

echo "[INFO] Environment ${ENV_NAME} is ready."
python - <<'PY'
import torch
import nnunetv2
print('torch', torch.__version__, 'cuda_available', torch.cuda.is_available())
print('nnunetv2 import ok')
PY
