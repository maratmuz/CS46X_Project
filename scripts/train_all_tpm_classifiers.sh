#!/usr/bin/env bash
# Train and evaluate one 3-class head for every cached genomic backbone.
#
# Submit from the project root, for example:
#   sbatch scripts/train_all_tpm_classifiers.sh
# or run inside an existing GPU allocation:
#   bash scripts/train_all_tpm_classifiers.sh
#
# Environment overrides include PYTHON_BIN, DATASET_ROOT, EMBED_ROOT,
# OUTPUT_ROOT, SWEEP_NAME, CLASS_WEIGHTING, DEVICE, SEED, and INCLUDE_NTV3.
# Extra command-line arguments are forwarded to every `train` invocation.

#SBATCH --job-name=tpm-classifier-heads
#SBATCH --account=eecs
#SBATCH --partition=dgxh
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=64G
#SBATCH --time=2-00:00:00
#SBATCH --output=tpm-classifier-heads-%j.log

set -euo pipefail

# sbatch executes a private copy from /var/spool/slurmd, so BASH_SOURCE no
# longer identifies the submitted file. Prefer an explicit override and then
# Slurm's original submission directory. The BASH_SOURCE route remains useful
# for direct `bash scripts/...` execution.
if [[ -n "${PROJECT_ROOT:-}" ]]; then
  project_candidate="${PROJECT_ROOT}"
elif [[ -n "${SLURM_SUBMIT_DIR:-}" ]] && \
     [[ -f "${SLURM_SUBMIT_DIR}/scripts/train_tpm_classifiers.py" ]]; then
  project_candidate="${SLURM_SUBMIT_DIR}"
else
  script_dir="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
  project_candidate="${script_dir}/.."
fi
PROJECT_ROOT="$(cd -- "${project_candidate}" && pwd)"
if [[ ! -f "${PROJECT_ROOT}/scripts/train_tpm_classifiers.py" ]] || \
   [[ ! -f "${PROJECT_ROOT}/scripts/report_tpm_classifiers.py" ]]; then
  echo "ERROR: Could not locate the project scripts under: ${PROJECT_ROOT}" >&2
  echo "Submit from the project root or set PROJECT_ROOT explicitly." >&2
  exit 2
fi

PYTHON_BIN="${PYTHON_BIN:-/nfs/stak/users/limjar/hpc-share/conda-envs/evo2/bin/python}"
DATASET_ROOT="${DATASET_ROOT:-/nfs/hpc/share/evo2_shared/datasets/pgb_parquet_tpm_3class}"
EMBED_ROOT="${EMBED_ROOT:-/nfs/hpc/share/evo2_shared/frozen-embeddings}"
OUTPUT_ROOT="${OUTPUT_ROOT:-/nfs/hpc/share/evo2_shared/tpm-classification-sweeps}"
SWEEP_NAME="${SWEEP_NAME:-$(date +%Y-%m-%d_%H-%M-%S)}"
SWEEP_DIR="${OUTPUT_ROOT}/${SWEEP_NAME}"
CLASS_WEIGHTING="${CLASS_WEIGHTING:-balanced}"
DEVICE="${DEVICE:-cuda}"
SEED="${SEED:-42}"
INCLUDE_NTV3="${INCLUDE_NTV3:-1}"

# Required by deterministic CUDA matrix multiplication in modern PyTorch.
export CUBLAS_WORKSPACE_CONFIG="${CUBLAS_WORKSPACE_CONFIG:-:4096:8}"
export PYTHONUNBUFFERED="${PYTHONUNBUFFERED:-1}"
export OPENBLAS_NUM_THREADS="${OPENBLAS_NUM_THREADS:-1}"
export OMP_NUM_THREADS="${OMP_NUM_THREADS:-1}"
export MKL_NUM_THREADS="${MKL_NUM_THREADS:-1}"
export MPLCONFIGDIR="${MPLCONFIGDIR:-${SWEEP_DIR}/.matplotlib}"
export XDG_CACHE_HOME="${XDG_CACHE_HOME:-${SWEEP_DIR}/.cache}"

case "${CLASS_WEIGHTING}" in
  none|balanced) ;;
  *) echo "CLASS_WEIGHTING must be 'none' or 'balanced'" >&2; exit 2 ;;
esac

models=(
  "agront 1b"
  "evo2 7b"
  "evo2 20b"
  "plantcad2 small"
  "plantcad2 medium"
  "plantcad2 large"
)
if [[ "${INCLUDE_NTV3}" == "1" ]]; then
  models+=("ntv3 100m" "ntv3 650m")
fi

mkdir -p "${SWEEP_DIR}/runs"
run_dirs=()

echo "Sweep directory: ${SWEEP_DIR}"
echo "Project root: ${PROJECT_ROOT}"
echo "Class weighting: ${CLASS_WEIGHTING}"
echo "Models: ${models[*]}"

for specification in "${models[@]}"; do
  read -r model size <<< "${specification}"
  run_dir="${SWEEP_DIR}/runs/${model}_${size}"
  run_dirs+=("${run_dir}")
  complete=1
  if [[ ! -f "${run_dir}/run_config.json" || ! -f "${run_dir}/summary.json" ]]; then
    complete=0
  else
    while IFS= read -r species; do
      [[ -f "${run_dir}/${species}/best.pt" ]] || complete=0
    done < <("${PYTHON_BIN}" -c \
      'import json,sys; print(*json.load(open(sys.argv[1]))["species"], sep="\n")' \
      "${run_dir}/run_config.json")
  fi

  if [[ "${complete}" == "1" ]]; then
    echo "Skipping completed training run: ${model}/${size}"
  else
    if [[ -d "${run_dir}" ]] && [[ -n "$(find "${run_dir}" -mindepth 1 -maxdepth 1 -print -quit)" ]]; then
      echo "Incomplete nonempty run directory; refusing to overwrite: ${run_dir}" >&2
      echo "Choose a new SWEEP_NAME or inspect/move that directory." >&2
      exit 2
    fi
    echo "Training ${model}/${size}"
    "${PYTHON_BIN}" "${PROJECT_ROOT}/scripts/train_tpm_classifiers.py" train \
      --model "${model}" --size "${size}" \
      --dataset-root "${DATASET_ROOT}" --embed-root "${EMBED_ROOT}" \
      --run-dir "${run_dir}" --class-weighting "${CLASS_WEIGHTING}" \
      --seed "${SEED}" --device "${DEVICE}" "$@"
  fi

  echo "Evaluating held-out test sets for ${model}/${size}"
  "${PYTHON_BIN}" "${PROJECT_ROOT}/scripts/train_tpm_classifiers.py" evaluate \
    --run-dir "${run_dir}" --output-dir "${run_dir}/evaluation" --device "${DEVICE}"
done

echo "Building comparison tables and confusion matrices"
"${PYTHON_BIN}" "${PROJECT_ROOT}/scripts/report_tpm_classifiers.py" \
  --run-dir "${run_dirs[@]}" --output-dir "${SWEEP_DIR}/report"

echo "Completed classifier sweep: ${SWEEP_DIR}"
echo "Metrics: ${SWEEP_DIR}/report/metrics_by_model_species.csv"
echo "Confusion matrices: ${SWEEP_DIR}/report/confusion_matrices_all_species.png"
