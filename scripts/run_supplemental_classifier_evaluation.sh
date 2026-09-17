#!/usr/bin/env bash
# Extract only missing supplemental embeddings, then evaluate every trained TPM
# classifier (including NTv3 and the DeepCRE-style sequence CNN).

#SBATCH --job-name=tpm-supp-eval
#SBATCH --account=eecs
#SBATCH --partition=gpu
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=64G
#SBATCH --time=3-00:00:00
#SBATCH --output=tpm-supp-eval-%j.log

set -euo pipefail

if [[ -n "${PROJECT_ROOT:-}" ]]; then
  project_candidate="${PROJECT_ROOT}"
elif [[ -n "${SLURM_SUBMIT_DIR:-}" ]] && \
     [[ -f "${SLURM_SUBMIT_DIR}/scripts/evaluate_tpm_classifiers_supplemental.py" ]]; then
  project_candidate="${SLURM_SUBMIT_DIR}"
else
  script_dir="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
  project_candidate="${script_dir}/.."
fi
PROJECT_ROOT="$(cd -- "${project_candidate}" && pwd)"
SHARED_DIR="${SHARED_DIR:-/nfs/hpc/share/evo2_shared}"
PLANT_SEQ_REPO="${PLANT_SEQ_REPO:-${SHARED_DIR}/plant-seq-2-expr}"
PYTHON_BIN="${PYTHON_BIN:-/nfs/stak/users/limjar/hpc-share/conda-envs/evo2/bin/python}"
NTV3_SIF="${NTV3_SIF:-${SHARED_DIR}/containers/plantcad2.sif}"
NTV3_CODE="${NTV3_CODE:-${SHARED_DIR}/py-overlays/ntv3-code}"
SWEEP_DIR="${SWEEP_DIR:-${SHARED_DIR}/tpm-classification-sweeps/tpm_balanced_v1}"
DATASET_ROOT="${DATASET_ROOT:-${SHARED_DIR}/datasets/supplemental_loci_parquet}"
EMBED_ROOT="${EMBED_ROOT:-${SHARED_DIR}/frozen-embeddings/supplemental}"
OUTPUT_DIR="${OUTPUT_DIR:-${SWEEP_DIR}/supplemental_evaluation}"
SKIP_EXTRACTION="${SKIP_EXTRACTION:-0}"

if [[ ! -f "${PROJECT_ROOT}/scripts/evaluate_tpm_classifiers_supplemental.py" ]]; then
  echo "ERROR: evaluation script is missing under ${PROJECT_ROOT}/scripts" >&2
  exit 2
fi
if [[ ! -x "${PLANT_SEQ_REPO}/scripts/run_supplemental_extraction.sh" ]] || \
   [[ ! -f "${PLANT_SEQ_REPO}/train_ntv3.py" ]]; then
  echo "ERROR: plant-seq-2-expr supplemental extraction scripts are unavailable" >&2
  exit 2
fi
if [[ ! -f "${NTV3_SIF}" ]]; then
  echo "ERROR: NTv3-compatible container is missing: ${NTV3_SIF}" >&2
  exit 2
fi
for ntv3_module in configuration_ntv3_pretrained.py modeling_ntv3_pretrained.py tokenization_ntv3.py; do
  if [[ ! -f "${NTV3_CODE}/${ntv3_module}" ]]; then
    echo "ERROR: NTv3 compatibility module is missing: ${NTV3_CODE}/${ntv3_module}" >&2
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

models=(
  "agront 1b"
  "evo2 7b"
  "evo2 20b"
  "plantcad2 small"
  "plantcad2 medium"
  "plantcad2 large"
  "ntv3 100m"
  "ntv3 650m"
)
locus_sets=(pseudogenes intergenic)

echo "Sweep: ${SWEEP_DIR}"
echo "Supplemental data: ${DATASET_ROOT}"
echo "Supplemental embeddings: ${EMBED_ROOT}"
echo "Results: ${OUTPUT_DIR}"

if [[ "${SKIP_EXTRACTION}" != "1" ]]; then
  for specification in "${models[@]}"; do
    read -r model size <<< "${specification}"
    for locus_set in "${locus_sets[@]}"; do
      cache_dir="${EMBED_ROOT}/${locus_set}/${model}/${size}"
      if [[ -d "${cache_dir}" ]]; then
        cached_count="$(find "${cache_dir}" -type f -name '*.pt' | wc -l)"
      else
        cached_count=0
      fi
      if [[ "${cached_count}" -ge 15 ]]; then
        echo "Cached: ${model}/${size} ${locus_set} (${cached_count} files)"
        continue
      fi
      echo "Extracting missing cache: ${model}/${size} ${locus_set} (${cached_count}/15 present)"
      if [[ "${model}" == "ntv3" ]]; then
        # ntv3.sif currently combines transformers 5.8 with a Torch 2.6 alpha
        # lacking TransformGetItemToIndex. plantcad2.sif uses the same CUDA
        # base but pins the known-compatible transformers 4.49 and includes
        # NTv3's remote-code dependencies. Use train_ntv3.py unchanged while
        # routing its normal PGB loader/cache to this supplemental locus set.
        echo "NTv3 container: ${NTV3_SIF}"
        apptainer exec --cleanenv --nv \
          -B "${SHARED_DIR}:${SHARED_DIR}" \
          --env "HF_HOME=${SHARED_DIR}/hf-cache" \
          --env "HF_TOKEN=${HF_TOKEN:-${HUGGINGFACE_HUB_TOKEN:-}}" \
          --env "SHARED_DIR=${SHARED_DIR}" \
          --env "PGB_DIR=${DATASET_ROOT}/${locus_set}" \
          --env "EMBED_ROOT=${EMBED_ROOT}/${locus_set}" \
          --env "PYTHONPATH=${NTV3_CODE}" \
          --env CUDA_VISIBLE_DEVICES=0 \
          --env PYTHONNOUSERSITE=1 --env PYTHONUNBUFFERED=1 \
          --env OPENBLAS_NUM_THREADS=1 --env OMP_NUM_THREADS=1 \
          "${NTV3_SIF}" python -c \
          'import torch, transformers, configuration_ntv3_pretrained; from transformers import AutoModelForMaskedLM; assert torch.cuda.is_available(); print(f">>> NTv3 environment OK: torch={torch.__version__}, transformers={transformers.__version__}, gpu={torch.cuda.get_device_name(0)}", flush=True)'
        apptainer exec --cleanenv --nv \
          -B "${SHARED_DIR}:${SHARED_DIR}" \
          --env "HF_HOME=${SHARED_DIR}/hf-cache" \
          --env "HF_TOKEN=${HF_TOKEN:-${HUGGINGFACE_HUB_TOKEN:-}}" \
          --env "SHARED_DIR=${SHARED_DIR}" \
          --env "PGB_DIR=${DATASET_ROOT}/${locus_set}" \
          --env "EMBED_ROOT=${EMBED_ROOT}/${locus_set}" \
          --env "PYTHONPATH=${NTV3_CODE}" \
          --env CUDA_VISIBLE_DEVICES=0 \
          --env PYTHONNOUSERSITE=1 --env PYTHONUNBUFFERED=1 \
          --env OPENBLAS_NUM_THREADS=1 --env OMP_NUM_THREADS=1 \
          "${NTV3_SIF}" python "${PLANT_SEQ_REPO}/train_ntv3.py" \
          --size "${size}" --extract-only
      else
        "${PLANT_SEQ_REPO}/scripts/run_supplemental_extraction.sh" \
          "${model}" "${size}" "${locus_set}"
      fi
    done
  done
fi

"${PYTHON_BIN}" "${PROJECT_ROOT}/scripts/evaluate_tpm_classifiers_supplemental.py" \
  --sweep-dir "${SWEEP_DIR}" --dataset-root "${DATASET_ROOT}" \
  --embed-root "${EMBED_ROOT}" --output-dir "${OUTPUT_DIR}" \
  --device cuda --resume "$@"

echo "Supplemental classifier evaluation complete: ${OUTPUT_DIR}"
