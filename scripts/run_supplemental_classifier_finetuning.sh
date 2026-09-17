#!/usr/bin/env bash
# Continue completed PGB heads on supplemental train loci with measured-PGB
# replay, then evaluate PGB test and supplemental test in separate reports.
#
# Submit from the project root:
#   sbatch scripts/run_supplemental_classifier_finetuning.sh
# To resume an interrupted job, submit again with the same OUTPUT_SWEEP_DIR.

#SBATCH --job-name=tpm-supp-finetune
#SBATCH --account=eecs
#SBATCH --partition=gpu
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=96G
#SBATCH --time=3-00:00:00
#SBATCH --output=tpm-supp-finetune-%j.log

set -euo pipefail

if [[ -n "${PROJECT_ROOT:-}" ]]; then
  project_candidate="${PROJECT_ROOT}"
elif [[ -n "${SLURM_SUBMIT_DIR:-}" ]] && \
     [[ -f "${SLURM_SUBMIT_DIR}/scripts/finetune_tpm_classifiers_supplemental.py" ]]; then
  project_candidate="${SLURM_SUBMIT_DIR}"
else
  script_dir="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
  project_candidate="${script_dir}/.."
fi
PROJECT_ROOT="$(cd -- "${project_candidate}" && pwd)"

SHARED_DIR="${SHARED_DIR:-/nfs/hpc/share/evo2_shared}"
PYTHON_BIN="${PYTHON_BIN:-/nfs/stak/users/limjar/hpc-share/conda-envs/evo2/bin/python}"
SOURCE_SWEEP_DIR="${SOURCE_SWEEP_DIR:-${SHARED_DIR}/tpm-classification-sweeps/tpm_balanced_v1}"
OUTPUT_SWEEP_DIR="${OUTPUT_SWEEP_DIR:-${SHARED_DIR}/tpm-classification-sweeps/tpm_balanced_v1_supplemental_finetune}"
SUPPLEMENTAL_DATASET_ROOT="${SUPPLEMENTAL_DATASET_ROOT:-${SHARED_DIR}/datasets/supplemental_loci_parquet}"
SUPPLEMENTAL_EMBED_ROOT="${SUPPLEMENTAL_EMBED_ROOT:-${SHARED_DIR}/frozen-embeddings/supplemental}"
DEVICE="${DEVICE:-cuda}"
REPLAY_RATIO="${REPLAY_RATIO:-1.0}"
SUPPLEMENTAL_LOSS_WEIGHT="${SUPPLEMENTAL_LOSS_WEIGHT:-0.2}"
LEARNING_RATE="${LEARNING_RATE:-1e-5}"
WEIGHT_DECAY="${WEIGHT_DECAY:-1e-2}"
EPOCHS="${EPOCHS:-20}"
PATIENCE="${PATIENCE:-5}"
SEED="${SEED:-42}"
SKIP_BASELINE_SUPPLEMENTAL="${SKIP_BASELINE_SUPPLEMENTAL:-0}"

if [[ "$#" -ne 0 ]]; then
  echo "ERROR: configure this job with environment variables, not positional arguments." >&2
  exit 2
fi
for required in finetune_tpm_classifiers_supplemental.py \
                evaluate_tpm_classifiers_supplemental.py \
                report_tpm_classifiers.py \
                compare_supplemental_finetuning.py \
                train_tpm_classifiers.py \
                train_deepcre_tpm_classifier.py; do
  if [[ ! -f "${PROJECT_ROOT}/scripts/${required}" ]]; then
    echo "ERROR: missing ${PROJECT_ROOT}/scripts/${required}" >&2
    exit 2
  fi
done
if [[ ! -x "${PYTHON_BIN}" ]]; then
  echo "ERROR: Python interpreter is not executable: ${PYTHON_BIN}" >&2
  exit 2
fi
if [[ ! -d "${SOURCE_SWEEP_DIR}/runs" ]]; then
  echo "ERROR: completed source sweep was not found: ${SOURCE_SWEEP_DIR}/runs" >&2
  exit 2
fi
if [[ ! -d "${SUPPLEMENTAL_DATASET_ROOT}/pseudogenes" ]] || \
   [[ ! -d "${SUPPLEMENTAL_DATASET_ROOT}/intergenic" ]]; then
  echo "ERROR: supplemental pseudogenes/intergenic Parquet splits are missing." >&2
  echo "Expected them under ${SUPPLEMENTAL_DATASET_ROOT}" >&2
  exit 2
fi
if [[ ! -d "${SUPPLEMENTAL_EMBED_ROOT}/pseudogenes" ]] || \
   [[ ! -d "${SUPPLEMENTAL_EMBED_ROOT}/intergenic" ]]; then
  echo "ERROR: supplemental pseudogene/intergenic embedding caches are missing." >&2
  echo "Expected them under ${SUPPLEMENTAL_EMBED_ROOT}" >&2
  exit 2
fi
if [[ "${OUTPUT_SWEEP_DIR}" == "${SOURCE_SWEEP_DIR}" ]]; then
  echo "ERROR: output sweep must be separate from the source sweep." >&2
  exit 2
fi

export CUBLAS_WORKSPACE_CONFIG="${CUBLAS_WORKSPACE_CONFIG:-:4096:8}"
export PYTHONUNBUFFERED="${PYTHONUNBUFFERED:-1}"
export OPENBLAS_NUM_THREADS="${OPENBLAS_NUM_THREADS:-1}"
export OMP_NUM_THREADS="${OMP_NUM_THREADS:-1}"
export MKL_NUM_THREADS="${MKL_NUM_THREADS:-1}"
export NUMEXPR_NUM_THREADS="${NUMEXPR_NUM_THREADS:-1}"
export MPLCONFIGDIR="${MPLCONFIGDIR:-${OUTPUT_SWEEP_DIR}/.matplotlib}"
export XDG_CACHE_HOME="${XDG_CACHE_HOME:-${OUTPUT_SWEEP_DIR}/.cache}"

mkdir -p "${OUTPUT_SWEEP_DIR}/runs"
source_runs=()
for candidate in "${SOURCE_SWEEP_DIR}"/runs/*; do
  if [[ -f "${candidate}/run_config.json" ]] && [[ -f "${candidate}/summary.json" ]]; then
    source_runs+=("${candidate}")
  fi
done
if [[ "${#source_runs[@]}" -eq 0 ]]; then
  echo "ERROR: no completed source classifier runs found." >&2
  exit 2
fi

echo "=== Supplemental classifier fine-tuning ==="
echo "Start time: $(date)"
echo "Job ID: ${SLURM_JOB_ID:-N/A}"
echo "Source sweep: ${SOURCE_SWEEP_DIR}"
echo "Output sweep: ${OUTPUT_SWEEP_DIR}"
echo "Supplemental dataset: ${SUPPLEMENTAL_DATASET_ROOT}"
echo "Supplemental caches: ${SUPPLEMENTAL_EMBED_ROOT}"
echo "Heads: ${#source_runs[@]}"
echo "Replay ratio: ${REPLAY_RATIO}; weak-loss weight: ${SUPPLEMENTAL_LOSS_WEIGHT}"
echo "Only supplemental train and PGB train update weights."

fine_runs=()
for source_run in "${source_runs[@]}"; do
  run_name="$(basename -- "${source_run}")"
  fine_run="${OUTPUT_SWEEP_DIR}/runs/${run_name}"
  fine_runs+=("${fine_run}")
  read -r model size < <("${PYTHON_BIN}" -c \
    'import json,sys; d=json.load(open(sys.argv[1])); print(d["model"],d["size"])' \
    "${source_run}/run_config.json")
  echo "=== ${model}/${size}: continuing ${source_run} -> ${fine_run} ==="
  batch_size=256
  eval_batch_size=512
  epochs="${EPOCHS}"
  patience="${PATIENCE}"
  weight_decay="${WEIGHT_DECAY}"
  if [[ "${model}" == "deepcre" ]]; then
    batch_size=32
    eval_batch_size=64
    epochs="${DEEPCRE_EPOCHS:-8}"
    patience="${DEEPCRE_PATIENCE:-3}"
    weight_decay="${DEEPCRE_WEIGHT_DECAY:-0}"
  fi

  "${PYTHON_BIN}" "${PROJECT_ROOT}/scripts/finetune_tpm_classifiers_supplemental.py" \
    --source-run-dir "${source_run}" \
    --run-dir "${fine_run}" \
    --supplemental-dataset-root "${SUPPLEMENTAL_DATASET_ROOT}" \
    --supplemental-embed-root "${SUPPLEMENTAL_EMBED_ROOT}" \
    --replay-ratio "${REPLAY_RATIO}" \
    --supplemental-loss-weight "${SUPPLEMENTAL_LOSS_WEIGHT}" \
    --learning-rate "${LEARNING_RATE}" \
    --weight-decay "${weight_decay}" \
    --epochs "${epochs}" --patience "${patience}" --seed "${SEED}" \
    --batch-size "${batch_size}" --eval-batch-size "${eval_batch_size}" \
    --device "${DEVICE}"

  if [[ "${model}" == "deepcre" ]]; then
    "${PYTHON_BIN}" "${PROJECT_ROOT}/scripts/train_deepcre_tpm_classifier.py" evaluate \
      --run-dir "${fine_run}" --output-dir "${fine_run}/evaluation" \
      --device "${DEVICE}" --eval-batch-size "${eval_batch_size}"
  else
    "${PYTHON_BIN}" "${PROJECT_ROOT}/scripts/train_tpm_classifiers.py" evaluate \
      --run-dir "${fine_run}" --output-dir "${fine_run}/evaluation" \
      --device "${DEVICE}" --eval-batch-size "${eval_batch_size}"
  fi
done

echo "=== Reporting measured held-out PGB test metrics ==="
"${PYTHON_BIN}" "${PROJECT_ROOT}/scripts/report_tpm_classifiers.py" \
  --run-dir "${fine_runs[@]}" \
  --output-dir "${OUTPUT_SWEEP_DIR}/report" \
  --fine-tuned-supplemental

if [[ "${SKIP_BASELINE_SUPPLEMENTAL}" != "1" ]]; then
  echo "=== Evaluating untouched source heads on supplemental test for comparison ==="
  "${PYTHON_BIN}" "${PROJECT_ROOT}/scripts/evaluate_tpm_classifiers_supplemental.py" \
    --sweep-dir "${SOURCE_SWEEP_DIR}" \
    --dataset-root "${SUPPLEMENTAL_DATASET_ROOT}" \
    --embed-root "${SUPPLEMENTAL_EMBED_ROOT}" \
    --output-dir "${OUTPUT_SWEEP_DIR}/baseline_supplemental_test" \
    --splits test --device "${DEVICE}" --resume
fi

echo "=== Evaluating fine-tuned heads on held-out supplemental test ==="
"${PYTHON_BIN}" "${PROJECT_ROOT}/scripts/evaluate_tpm_classifiers_supplemental.py" \
  --sweep-dir "${OUTPUT_SWEEP_DIR}" \
  --dataset-root "${SUPPLEMENTAL_DATASET_ROOT}" \
  --embed-root "${SUPPLEMENTAL_EMBED_ROOT}" \
  --output-dir "${OUTPUT_SWEEP_DIR}/supplemental_test" \
  --splits test --fine-tuned --device "${DEVICE}" --resume

"${PYTHON_BIN}" "${PROJECT_ROOT}/scripts/compare_supplemental_finetuning.py" \
  --source-sweep-dir "${SOURCE_SWEEP_DIR}" \
  --fine-sweep-dir "${OUTPUT_SWEEP_DIR}"

echo "Fine-tuned PGB metrics: ${OUTPUT_SWEEP_DIR}/report/metrics_by_model_species.csv"
echo "Fine-tuned supplemental low-call rates: ${OUTPUT_SWEEP_DIR}/supplemental_test/low_call_rate_table.csv"
if [[ "${SKIP_BASELINE_SUPPLEMENTAL}" != "1" ]]; then
  echo "Untouched-baseline supplemental low-call rates: ${OUTPUT_SWEEP_DIR}/baseline_supplemental_test/low_call_rate_table.csv"
fi
echo "Finish time: $(date)"
