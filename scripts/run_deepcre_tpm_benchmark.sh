#!/usr/bin/env bash
# Train the DeepCRE-style sequence CNN, re-evaluate it, and rebuild the full
# publication report alongside all frozen-embedding classifier heads.

#SBATCH --job-name=deepcre-tpm
#SBATCH --account=eecs
#SBATCH --partition=gpu
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=64G
#SBATCH --time=3-00:00:00
#SBATCH --output=deepcre-tpm-%j.log

set -euo pipefail

if [[ -n "${PROJECT_ROOT:-}" ]]; then
  project_candidate="${PROJECT_ROOT}"
elif [[ -n "${SLURM_SUBMIT_DIR:-}" ]] && \
     [[ -f "${SLURM_SUBMIT_DIR}/scripts/train_deepcre_tpm_classifier.py" ]]; then
  project_candidate="${SLURM_SUBMIT_DIR}"
else
  script_dir="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
  project_candidate="${script_dir}/.."
fi
PROJECT_ROOT="$(cd -- "${project_candidate}" && pwd)"
PYTHON_BIN="${PYTHON_BIN:-/nfs/stak/users/limjar/hpc-share/conda-envs/evo2/bin/python}"
SWEEP_DIR="${SWEEP_DIR:-/nfs/hpc/share/evo2_shared/tpm-classification-sweeps/tpm_balanced_v1}"
DATASET_ROOT="${DATASET_ROOT:-/nfs/hpc/share/evo2_shared/datasets/pgb_parquet_tpm_3class}"
RUN_DIR="${SWEEP_DIR}/runs/deepcre_cnn"
REPORT_DIR="${SWEEP_DIR}/report"
DEVICE="${DEVICE:-cuda}"

for required in train_deepcre_tpm_classifier.py report_tpm_classifiers.py; do
  if [[ ! -f "${PROJECT_ROOT}/scripts/${required}" ]]; then
    echo "ERROR: missing ${PROJECT_ROOT}/scripts/${required}" >&2
    exit 2
  fi
done

export CUBLAS_WORKSPACE_CONFIG="${CUBLAS_WORKSPACE_CONFIG:-:4096:8}"
export PYTHONUNBUFFERED="${PYTHONUNBUFFERED:-1}"
export OPENBLAS_NUM_THREADS="${OPENBLAS_NUM_THREADS:-1}"
export OMP_NUM_THREADS="${OMP_NUM_THREADS:-1}"
export MKL_NUM_THREADS="${MKL_NUM_THREADS:-1}"
export NUMEXPR_NUM_THREADS="${NUMEXPR_NUM_THREADS:-1}"
export MPLCONFIGDIR="${MPLCONFIGDIR:-${SWEEP_DIR}/.matplotlib}"
export XDG_CACHE_HOME="${XDG_CACHE_HOME:-${SWEEP_DIR}/.cache}"

echo "Project root: ${PROJECT_ROOT}"
echo "Sweep: ${SWEEP_DIR}"
echo "DeepCRE run: ${RUN_DIR}"

complete=1
if [[ ! -f "${RUN_DIR}/run_config.json" || ! -f "${RUN_DIR}/summary.json" ]]; then
  complete=0
else
  while IFS= read -r species; do
    [[ -f "${RUN_DIR}/${species}/best.pt" ]] || complete=0
  done < <("${PYTHON_BIN}" -c \
    'import json,sys; print(*json.load(open(sys.argv[1]))["species"], sep="\n")' \
    "${RUN_DIR}/run_config.json")
fi

if [[ "${complete}" == "1" ]]; then
  echo "DeepCRE training is already complete; preserving checkpoints."
else
  echo "Starting or resuming incomplete DeepCRE training."
  "${PYTHON_BIN}" "${PROJECT_ROOT}/scripts/train_deepcre_tpm_classifier.py" train \
    --dataset-root "${DATASET_ROOT}" --run-dir "${RUN_DIR}" \
    --class-weighting balanced --resume --device "${DEVICE}" "$@"
fi

"${PYTHON_BIN}" "${PROJECT_ROOT}/scripts/train_deepcre_tpm_classifier.py" evaluate \
  --run-dir "${RUN_DIR}" --output-dir "${RUN_DIR}/evaluation" --device "${DEVICE}"

run_dirs=()
for candidate in "${SWEEP_DIR}"/runs/*; do
  if [[ -f "${candidate}/run_config.json" ]] && [[ -f "${candidate}/summary.json" ]]; then
    run_dirs+=("${candidate}")
  fi
done
if [[ "${#run_dirs[@]}" -eq 0 ]]; then
  echo "ERROR: no completed classifier runs found under ${SWEEP_DIR}/runs" >&2
  exit 2
fi

"${PYTHON_BIN}" "${PROJECT_ROOT}/scripts/report_tpm_classifiers.py" \
  --run-dir "${run_dirs[@]}" --output-dir "${REPORT_DIR}"

echo "DeepCRE benchmark and publication report complete: ${REPORT_DIR}"
