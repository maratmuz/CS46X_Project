#!/usr/bin/env bash
set -euo pipefail

LOCUS_SET="${SUPPLEMENTAL_LOCUS_SET:-combined}"
if [[ "${1:-}" == "--locus-set" ]]; then
  LOCUS_SET="${2:?--locus-set requires combined, pseudogenes, or intergenic}"
  shift 2
fi
case "${LOCUS_SET}" in
  combined|pseudogenes|intergenic) ;;
  *) echo "ERROR: --locus-set must be combined, pseudogenes, or intergenic" >&2; exit 2 ;;
esac

MODEL="${1:?usage: extract_supplemental_embeddings.sh [--locus-set SET] <evo2|plantcad2|agront|ntv3> <size> [extra args...]}"
SIZE="${2:?usage: extract_supplemental_embeddings.sh [--locus-set SET] <evo2|plantcad2|agront|ntv3> <size> [extra args...]}"
shift 2

SHARED_DIR="${SHARED_DIR:-/nfs/hpc/share/evo2_shared}"
BASE_DATASET_DIR="${SUPPLEMENTAL_DATASET_DIR:-${SHARED_DIR}/datasets/supplemental_loci_parquet}"
BASE_EMBED_ROOT="${SUPPLEMENTAL_EMBED_ROOT:-${SHARED_DIR}/supplemental-loci-embeddings}"
if [[ "${LOCUS_SET}" == "combined" ]]; then
  DATASET_DIR="${BASE_DATASET_DIR}"
  EMBED_ROOT="${BASE_EMBED_ROOT}/combined"
else
  DATASET_DIR="${BASE_DATASET_DIR}/${LOCUS_SET}"
  EMBED_ROOT="${BASE_EMBED_ROOT}/${LOCUS_SET}"
fi
EXTRACT_SCRIPT="${PLANT_SEQ_EXTRACT_SCRIPT:-${SHARED_DIR}/plant-seq-2-expr/scripts/extract.sh}"

if [[ ! -d "${DATASET_DIR}" || ! -f "${DATASET_DIR}/manifest.json" ]]; then
  echo "ERROR: converted supplemental dataset is missing: ${DATASET_DIR}" >&2
  echo "Run scripts/convert_supplemental_loci.sh first." >&2
  exit 2
fi
if [[ ! -x "${EXTRACT_SCRIPT}" ]]; then
  echo "ERROR: extraction script is missing or not executable: ${EXTRACT_SCRIPT}" >&2
  exit 2
fi
if [[ "${BASE_EMBED_ROOT}" == "${SHARED_DIR}/frozen-embeddings" ]]; then
  echo "ERROR: refusing to use the normal PGB cache root for supplemental loci." >&2
  exit 2
fi

# APPTAINERENV_* reliably passes these overrides through the existing
# extract.sh wrapper and into each model-specific container.
export APPTAINERENV_PGB_DIR="${DATASET_DIR}"
export APPTAINERENV_EMBED_ROOT="${EMBED_ROOT}"
# Shared login/compute nodes can have tight process limits. Prevent NumPy's
# BLAS backend from trying to create dozens of unrelated CPU threads merely to
# read parquet metadata before GPU extraction begins.
CPU_THREADS="${SUPPLEMENTAL_CPU_THREADS:-1}"
export APPTAINERENV_OPENBLAS_NUM_THREADS="${CPU_THREADS}"
export APPTAINERENV_OMP_NUM_THREADS="${CPU_THREADS}"
export APPTAINERENV_MKL_NUM_THREADS="${CPU_THREADS}"
export APPTAINERENV_NUMEXPR_NUM_THREADS="${CPU_THREADS}"

EXTRA_ARGS=()
# train_agront.py dispatches to frozen extraction only when --frozen is
# present; the generic extract.sh currently does not add it itself.
if [[ "${MODEL}" == "agront" ]]; then
  EXTRA_ARGS+=(--frozen)
fi

echo ">>> Dataset: ${DATASET_DIR}"
echo ">>> Loci:    ${LOCUS_SET}"
echo ">>> Cache:   ${EMBED_ROOT}"
exec "${EXTRACT_SCRIPT}" "${MODEL}" "${SIZE}" "${EXTRA_ARGS[@]}" "$@"
