#!/usr/bin/env bash
set -euo pipefail

OUT_DIR="${1:-/tmp}"
mkdir -p "${OUT_DIR}"
LOG_FILE="${OUT_DIR}/gpu_precheck_$(date +%Y%m%d_%H%M%S).log"

if [[ -f "${HOME}/.bashrc" ]]; then
    # shellcheck disable=SC1090
    source "${HOME}/.bashrc"
fi

shopt -s expand_aliases || true

echo "[INFO] Running GPU precheck at $(date -Iseconds)" | tee -a "${LOG_FILE}"

if alias gpu >/dev/null 2>&1; then
    gpu | tee -a "${LOG_FILE}"
elif [[ -f "/export/home3/dazhou/debate-or-vote/scripts/gpu_monitor.py" ]]; then
    python /export/home3/dazhou/debate-or-vote/scripts/gpu_monitor.py | tee -a "${LOG_FILE}"
else
    echo "[ERROR] gpu alias and gpu_monitor.py are both unavailable." | tee -a "${LOG_FILE}"
    exit 2
fi

echo "[INFO] GPU precheck log: ${LOG_FILE}"
