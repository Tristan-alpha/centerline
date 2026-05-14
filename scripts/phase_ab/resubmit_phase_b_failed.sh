#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"

if [[ $# -ge 1 ]]; then
    JOB_RECORD="$1"
else
    JOB_RECORD="$(ls -1t "${ROOT_DIR}"/logs/phase_b/jobs_*.txt | head -n 1)"
fi

if [[ -z "${JOB_RECORD}" || ! -f "${JOB_RECORD}" ]]; then
    echo "[ERROR] Phase B job record not found. Provide it explicitly as first argument."
    exit 1
fi

CONFIG_FROM_RECORD="$(awk -F'config=' '/^phase=B/{print $2; exit}' "${JOB_RECORD}" | awk '{print $1}')"
CONFIG="${CONFIG:-${CONFIG_FROM_RECORD:-2d}}"
TRAINER="${TRAINER:-nnUNetTrainer}"
PLANS="${PLANS:-nnUNetPlans}"
CONDA_ENV="${CONDA_ENV:-sten-nnunet}"
PARTITION="${PARTITION:-RTXA6Kq}"
GPUS="${GPUS:-1}"
CPUS="${CPUS:-8}"
MEM="${MEM:-64G}"
TIME_LIMIT="${TIME_LIMIT:-24:00:00}"
FEATURE_DEP_JOB_ID="${FEATURE_DEP_JOB_ID:-}"

RUN_STAMP="$(date +%Y%m%d_%H%M%S)"
RETRY_RECORD="${ROOT_DIR}/logs/phase_b/retry_jobs_${RUN_STAMP}.txt"
mkdir -p "$(dirname "${RETRY_RECORD}")"

echo "source_job_record=${JOB_RECORD}" | tee "${RETRY_RECORD}"
echo "phase=B config=${CONFIG}" | tee -a "${RETRY_RECORD}"

echo "[INFO] Inspecting jobs listed in ${JOB_RECORD}"

retry_state() {
    case "$1" in
        FAILED|CANCELLED|TIMEOUT|OUT_OF_MEMORY|NODE_FAIL|PREEMPTED)
            return 0
            ;;
        *)
            return 1
            ;;
    esac
}

submitted=0
skipped=0

while IFS= read -r line; do
    [[ "${line}" == tag=* ]] || continue

    TAG=""
    DATASET_ID=""
    FOLD=""
    JOB_ID=""

    for token in ${line}; do
        key="${token%%=*}"
        value="${token#*=}"
        case "${key}" in
            tag) TAG="${value}" ;;
            dataset) DATASET_ID="${value}" ;;
            fold) FOLD="${value}" ;;
            job_id) JOB_ID="${value}" ;;
        esac
    done

    if [[ -z "${TAG}" || -z "${DATASET_ID}" || -z "${FOLD}" || -z "${JOB_ID}" ]]; then
        echo "[WARN] Skip malformed record: ${line}" | tee -a "${RETRY_RECORD}"
        skipped=$((skipped + 1))
        continue
    fi

    STATE="$(sacct -j "${JOB_ID}" --format=State --noheader -P | awk 'NR==1{print $1}' | cut -d'+' -f1)"
    STATE="${STATE:-UNKNOWN}"

    if retry_state "${STATE}"; then
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

        NEW_JOB_ID="$(sbatch "${SBATCH_ARGS[@]}" \
            "${ROOT_DIR}/scripts/phase_ab/run_phase_b_fold.sbatch" \
            "${DATASET_ID}" "${CONFIG}" "${FOLD}" "${TAG}" "${TRAINER}" "${PLANS}")"

        echo "retry tag=${TAG} dataset=${DATASET_ID} fold=${FOLD} old_job_id=${JOB_ID} old_state=${STATE} new_job_id=${NEW_JOB_ID}" | tee -a "${RETRY_RECORD}"
        echo "[INFO] Resubmitted ${TAG} fold ${FOLD}: ${JOB_ID} -> ${NEW_JOB_ID}"
        submitted=$((submitted + 1))
    else
        echo "keep tag=${TAG} dataset=${DATASET_ID} fold=${FOLD} job_id=${JOB_ID} state=${STATE}" | tee -a "${RETRY_RECORD}"
        skipped=$((skipped + 1))
    fi
done < "${JOB_RECORD}"

echo "[INFO] Retry summary: submitted=${submitted}, skipped=${skipped}" | tee -a "${RETRY_RECORD}"
echo "[INFO] Retry record: ${RETRY_RECORD}"
