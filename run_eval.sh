#!/bin/bash
#SBATCH -t 0-24:00:00
#SBATCH -J evo2_gene_eval
#SBATCH -A eecs
#SBATCH -p dgxh
#SBATCH --gres=gpu:1
#SBATCH --mem=32G
#SBATCH -o evo2_gene_eval.log

ml cuda/12.8
ml cudnn/8.9_cuda12

export CUDA_VISIBLE_DEVICES=0

# Set HuggingFace cache directories manually (if not provided in conda env initialization)
# export HF_HOME=/nfs/stak/users/limjar/hpc-share/evo2/hf-cache
# export HF_HUB_CACHE=$HF_HOME
# export TRANSFORMERS_CACHE=$HF_HOME/transformers
# export TMPDIR=/nfs/stak/users/limjar/hpc-share/evo2/tmp

# Activate conda env
source /nfs/stak/users/limjar/hpc-share/conda-envs/evo2/bin/activate

# Set env variables
export HOME="/nfs/stak/users/limjar"
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True

# Change to project directory
cd /nfs/stak/users/limjar/hpc-share/CS46X_Project

date

# Run evaluation
python run_eval.py

