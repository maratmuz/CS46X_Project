#!/bin/bash
#SBATCH -t 1-00:00:00
#SBATCH -J build_supp_loci
#SBATCH -A eecs
#SBATCH -p gpu
#SBATCH --cpus-per-task=2
#SBATCH --mem=32G
#SBATCH -o build_supplemental_loci_%j.log
set -euo pipefail

SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(cd -- "${SCRIPT_DIR}/.." && pwd)"
cd "${PROJECT_ROOT}"

usage() {
  cat <<USAGE
Usage:
  sbatch scripts/build_supplemental_loci.sh PATH_TO_PGB [additional CLI options]

Example:
  sbatch scripts/build_supplemental_loci.sh shared/datasets/PGB

Environment overrides:
  CONDA_ENV          Conda environment name/path (default: nt)
  GENOMES_ROOT       NCBI genomes directory
  OUTPUT_DIR         Supplemental-loci output directory
  MAX_PSEUDOGENES   Per-compartment pseudogene cap (default: 5000)
  MAX_INTERGENIC     Per-compartment intergenic cap (default: 5000)
  GENE_BUFFER        Intergenic gene buffer in bp (default: 5000)
  SEED               Deterministic sampling seed (default: 42)

Additional options such as --pad-boundaries, --include-organelles,
--include-unknown, --exact-sequence-mapping, or --force are passed directly
to build_supplemental_loci.py.
USAGE
}

if [[ $# -lt 1 || "$1" == "-h" || "$1" == "--help" ]]; then
  usage
  if [[ $# -lt 1 ]]; then
    exit 2
  fi
  exit 0
fi

PGB_DIR="$1"
shift

CONDA_ENV="${CONDA_ENV:-/nfs/stak/users/limjar/hpc-share/conda-envs/evo2}"
GENOMES_ROOT="${GENOMES_ROOT:-/nfs/hpc/share/evo2_shared/datasets/NCBI_genomes}"
OUTPUT_DIR="${OUTPUT_DIR:-/nfs/hpc/share/evo2_shared/datasets/supplemental_loci}"
MAX_PSEUDOGENES="${MAX_PSEUDOGENES:-5000}"
MAX_INTERGENIC="${MAX_INTERGENIC:-5000}"
GENE_BUFFER="${GENE_BUFFER:-5000}"
SEED="${SEED:-42}"

source "$(conda info --base)/etc/profile.d/conda.sh"
conda activate "${CONDA_ENV}"

echo "=== Job started ==="
echo "Job ID: ${SLURM_JOB_ID:-N/A}"
echo "Start time: $(date)"
echo "Host: $(hostname)"
echo "Project root: ${PROJECT_ROOT}"
echo "Conda environment: ${CONDA_ENV}"
echo "Genomes root: ${GENOMES_ROOT}"
echo "PGB directory: ${PGB_DIR}"
echo "Output directory: ${OUTPUT_DIR}"
echo "CPUs allocated: ${SLURM_CPUS_PER_TASK:-2}"

if [[ ! -d "${GENOMES_ROOT}" ]]; then
  echo "ERROR: Genomes root is not accessible: ${GENOMES_ROOT}" >&2
  echo "Run this job on a node where the shared NFS path is mounted." >&2
  exit 2
fi

if [[ ! -d "${PGB_DIR}" ]]; then
  echo "ERROR: PGB directory is not accessible: ${PGB_DIR}" >&2
  exit 2
fi

if ! python -c 'import pyfaidx' >/dev/null 2>&1; then
  echo "ERROR: pyfaidx is not installed in conda environment: ${CONDA_ENV}" >&2
  echo "Install it with: conda install -c conda-forge pyfaidx" >&2
  exit 2
fi

python /nfs/stak/users/limjar/hpc-share/CS46X_Project/scripts/build_supplemental_loci.py \
  --genomes-root "${GENOMES_ROOT}" \
  --output "${OUTPUT_DIR}" \
  --pgb-dir "${PGB_DIR}" \
  --max-pseudogenes "${MAX_PSEUDOGENES}" \
  --max-intergenic "${MAX_INTERGENIC}" \
  --gene-buffer "${GENE_BUFFER}" \
  --seed "${SEED}" \
  "$@"

echo "Finish time: $(date)"
echo "=== Job finished ==="
