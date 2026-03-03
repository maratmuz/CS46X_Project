#!/bin/bash
#SBATCH -t 2-00:00:00
#SBATCH -J train_ann_heads
#SBATCH -A eecs
#SBATCH -p gpu
#SBATCH --gres=gpu:1
#SBATCH --mem=64G
#SBATCH -o train_annotation_heads_%j.log
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

# Update these paths as needed.
DATASET_DIR=/nfs/hpc/share/evo2_shared/datasets/algae_splits
OUT_DIR=/nfs/hpc/share/evo2_shared/gene_annotator_v1

python train_annotation_heads.py \
  --dataset_dir "${DATASET_DIR}" \
  --out_dir "${OUT_DIR}" \
  --epochs 5 \
  --batch_size 16 \
  --lr 5e-5 \
  --weight_decay 2e-4 \
  --dropout 0.1 \
  --class_weights_feature 0.7,1.6,1.2,1.2 \
  --seed 0 \
  --use_cuda true \
#  --wandb_project genome_annotation \
#  --wandb_run_name train_annotation_heads

echo "Finish time: $(date)"
echo "=== Job finished ==="
