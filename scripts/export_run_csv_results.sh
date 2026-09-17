#!/usr/bin/env bash
# Copy only readable CSV artifacts from runs/ into a parallel runs_results/
# tree while preserving every relative directory and filename.

set -euo pipefail

SWEEP_DIR="${SWEEP_DIR:-/nfs/hpc/share/evo2_shared/tpm-classification-sweeps/tpm_balanced_v1}"
SOURCE_ROOT="${SOURCE_ROOT:-${SWEEP_DIR}/runs}"
DESTINATION_ROOT="${DESTINATION_ROOT:-${SWEEP_DIR}/runs_results}"

SOURCE_ROOT="$(readlink -f -- "${SOURCE_ROOT}")"
if [[ ! -d "${SOURCE_ROOT}" ]]; then
  echo "ERROR: source runs directory is missing: ${SOURCE_ROOT}" >&2
  exit 2
fi
if [[ -e "${DESTINATION_ROOT}" ]]; then
  DESTINATION_ROOT="$(readlink -f -- "${DESTINATION_ROOT}")"
else
  destination_parent="$(readlink -f -- "$(dirname -- "${DESTINATION_ROOT}")")"
  DESTINATION_ROOT="${destination_parent}/$(basename -- "${DESTINATION_ROOT}")"
fi
if [[ "${DESTINATION_ROOT}" == "${SOURCE_ROOT}" || "${DESTINATION_ROOT}" == "${SOURCE_ROOT}/"* ]]; then
  echo "ERROR: destination must not be the source or a child of it" >&2
  exit 2
fi
if [[ -d "${DESTINATION_ROOT}" ]] && \
   [[ -n "$(find "${DESTINATION_ROOT}" -mindepth 1 -maxdepth 1 -print -quit)" ]]; then
  echo "ERROR: destination is nonempty; refusing to mix or overwrite results: ${DESTINATION_ROOT}" >&2
  exit 2
fi

mapfile -d '' csv_files < <(find "${SOURCE_ROOT}" -type f -name '*.csv' -print0 | sort -z)
if [[ "${#csv_files[@]}" -eq 0 ]]; then
  echo "ERROR: no CSV files found under ${SOURCE_ROOT}" >&2
  exit 2
fi

echo "Copying ${#csv_files[@]} CSV files"
echo "  from: ${SOURCE_ROOT}"
echo "  to:   ${DESTINATION_ROOT}"
mkdir -p -- "${DESTINATION_ROOT}"
for source in "${csv_files[@]}"; do
  relative="${source#"${SOURCE_ROOT}/"}"
  if [[ "${relative}" == "${source}" || -z "${relative}" || "${relative}" == /* ]]; then
    echo "ERROR: could not derive a safe relative path for ${source}" >&2
    exit 2
  fi
  target="${DESTINATION_ROOT}/${relative}"
  mkdir -p -- "$(dirname -- "${target}")"
  cp --preserve=timestamps -- "${source}" "${target}"
done

copied_count="$(find "${DESTINATION_ROOT}" -type f -name '*.csv' | wc -l)"
other_count="$(find "${DESTINATION_ROOT}" -type f ! -name '*.csv' | wc -l)"
if [[ "${copied_count}" -ne "${#csv_files[@]}" || "${other_count}" -ne 0 ]]; then
  echo "ERROR: verification failed: expected ${#csv_files[@]} CSVs, found ${copied_count}; other files=${other_count}" >&2
  exit 2
fi
echo "Verified ${copied_count} CSV files and no non-CSV files in ${DESTINATION_ROOT}"
