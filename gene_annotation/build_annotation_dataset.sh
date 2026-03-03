#!/bin/bash
#SBATCH -t 4-00:00:00
#SBATCH -J build_annotation_dataset
#SBATCH -A eecs
#SBATCH -p gpu
#SBATCH --mem=32G
#SBATCH -o build_annotation_dataset_%j.log
set -euo pipefail

source "$(conda info --base)/etc/profile.d/conda.sh"
conda activate /nfs/stak/users/limjar/hpc-share/conda-envs/evo2

echo "=== Job started ==="
echo "Job ID: ${SLURM_JOB_ID:-N/A}"
echo "Start time: $(date)"

python build_annotation_dataset.py \
  --species_root /nfs/hpc/share/evo2_shared/datasets/three_algae \
  --out_tsv      /nfs/hpc/share/evo2_shared/datasets/splits_v2/three_algae_w8192.tsv \
  --window 8192 \
  --n_intergenic 500 \
  --n_utr 500 \
  --n_cds 500 \
  --n_intron 500 \
  --index_dir /tmp/__fasta_index__ \
  --seed 0

echo "Finish time: $(date)"
echo "=== Job finished ==="
