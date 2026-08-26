#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(cd -- "${SCRIPT_DIR}/.." && pwd)"
SHARED_DIR="${SHARED_DIR:-/nfs/hpc/share/evo2_shared}"
PYTHON_BIN="${PYTHON_BIN:-/nfs/stak/users/limjar/hpc-share/myVenv/bin/python}"
INPUT_DIR="${INPUT_DIR:-${SHARED_DIR}/datasets/supplemental_loci}"
OUTPUT_DIR="${OUTPUT_DIR:-${SHARED_DIR}/datasets/supplemental_loci_parquet}"

exec "${PYTHON_BIN}" "${PROJECT_ROOT}/scripts/convert_supplemental_loci_to_pgb.py" \
  --input "${INPUT_DIR}" \
  --output "${OUTPUT_DIR}" \
  "$@"
