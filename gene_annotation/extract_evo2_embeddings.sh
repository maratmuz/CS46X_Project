#!/bin/bash
#SBATCH -t 2-00:00:00
#SBATCH -J extract_evo2_emb
#SBATCH -A eecs
#SBATCH -p gpu
#SBATCH --gres=gpu:1
#SBATCH --mem=64G
#SBATCH -o extract_evo2_embeddings_%j.log
set -euo pipefail

source "$(conda info --base)/etc/profile.d/conda.sh"
conda activate /nfs/stak/users/limjar/hpc-share/conda-envs/evo2

nvidia-smi

export CUDA_VISIBLE_DEVICES=0

echo "=== Job started ==="
echo "Job ID: ${SLURM_JOB_ID:-N/A}"
echo "Start time: $(date)"

# If run directly (not via sbatch), avoid Evo2 auto-sharding across all visible GPUs.
# Pick the GPU with the most free memory unless CUDA_VISIBLE_DEVICES is already set.
if [[ -z "${SLURM_JOB_ID:-}" && -z "${CUDA_VISIBLE_DEVICES:-}" ]]; then
  if command -v nvidia-smi >/dev/null 2>&1; then
    BEST_GPU=$(
      nvidia-smi --query-gpu=index,memory.free --format=csv,noheader,nounits \
      | sort -t, -k2 -nr \
      | head -n1 \
      | cut -d, -f1 \
      | tr -d ' '
    )
    export CUDA_VISIBLE_DEVICES="${BEST_GPU}"
    echo "[WARN] Running outside SLURM. Set CUDA_VISIBLE_DEVICES=${CUDA_VISIBLE_DEVICES}"
  else
    export CUDA_VISIBLE_DEVICES=0
    echo "[WARN] Running outside SLURM and nvidia-smi unavailable. Set CUDA_VISIBLE_DEVICES=0"
  fi
fi

echo "CUDA_VISIBLE_DEVICES=${CUDA_VISIBLE_DEVICES:-<unset>}"

python extract_evo2_embeddings.py \
  --in_tsv /nfs/hpc/share/evo2_shared/datasets/splits_v2/three_algae_w8192.tsv \
  --out_dir /nfs/hpc/share/evo2_shared/datasets/splits_v2/three_algae_w8192_evo2_blocks26 \
  --model_name evo2_7b_base \
  --layer_name blocks.26 \
  --batch_size 2 \
  --device cuda:0 \
  --overwrite

echo "Finish time: $(date)"
echo "=== Job finished ==="
