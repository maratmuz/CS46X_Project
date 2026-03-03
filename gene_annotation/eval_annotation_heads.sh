#!/bin/bash
#SBATCH -t 1-00:00:00
#SBATCH -J eval_ann_heads
#SBATCH -A eecs
#SBATCH -p gpu
#SBATCH --gres=gpu:1
#SBATCH --mem=32G
#SBATCH -o eval_annotation_heads_%j.log
set -euo pipefail

source "$(conda info --base)/etc/profile.d/conda.sh"
conda activate /nfs/stak/users/limjar/hpc-share/conda-envs/evo2

nvidia-smi

echo "=== Job started ==="
echo "Job ID: ${SLURM_JOB_ID:-N/A}"
echo "Start time: $(date)"

#export WANDB_API_KEY=""
export CUDA_VISIBLE_DEVICES=0
export PYTORCH_NVML_BASED_CUDA_CHECK=0
echo "CUDA_VISIBLE_DEVICES=${CUDA_VISIBLE_DEVICES:-<unset>}"

# Update these paths as needed. OUT_DIR refers to the directory where the checkpoints are stored.
DATASET_DIR=/nfs/hpc/share/evo2_shared/datasets/algae_splits
OUT_DIR=/nfs/hpc/share/evo2_shared/gene_annotator_v1

python eval_annotation_heads.py \
  --dataset_dir "${DATASET_DIR}" \
  --out_dir "${OUT_DIR}" \
  --batch_size 16 \
  --seed 0 \
  --use_cuda true

echo "Finish time: $(date)"
echo "=== Job finished ==="
