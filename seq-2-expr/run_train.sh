#!/bin/bash
#SBATCH --job-name=agront_all
#SBATCH --output=logs/agront_all_%j.out
#SBATCH --error=logs/agront_all_%j.err
#SBATCH --time=24:00:00
#SBATCH -A eecs
#SBATCH -p dgxh
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=16
#SBATCH --mem=256G

module load cuda/12.8

source ~/miniconda3/etc/profile.d/conda.sh
conda activate nt

model=agro_nt 
species=('glycine_max' 'oryza_sativa' 'solanum_lycopersicum' 'zea_mays' 'arabidopsis_thaliana')

for name in "${species[@]}"; do
    python train.py \
        --model_name $model \
        --task_name $name \
        --fine_tune_method "lora" \
        --report_to "wandb" \
        --wandb_project "agront_seq2expr" \
        --max_steps 2000 \
        --batch_size 32 
done
