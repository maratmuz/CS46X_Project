#!/bin/bash
# Rice Model Training Script (Rigorous Split)
# Run this on a GPU node (dgxh-1 or dgxh-2)

set -e  # Exit on error

echo "================================================"
echo "Helixer Rice Model Training"
echo "================================================"

# Activate environment
source /nfs/stak/users/minchle/miniconda3/etc/profile.d/conda.sh
conda activate helixer_env

# Set library paths for GPU and HDF5
export LD_LIBRARY_PATH=/lib64:$LD_LIBRARY_PATH
export CUDALIBS=$(find /nfs/stak/users/minchle/miniconda3/envs/helixer_env/lib/python3.10/site-packages/nvidia -name lib -type d 2>/dev/null | paste -sd ":" -)
export LD_LIBRARY_PATH=$LD_LIBRARY_PATH:$CUDALIBS

# Verify GPU
echo "Checking GPU..."
python -c "import tensorflow as tf; gpus = tf.config.list_physical_devices('GPU'); print(f'GPUs found: {len(gpus)}'); assert len(gpus) > 0, 'No GPU found!'"

# Paths
DATA_DIR="/nfs/stak/users/minchle/hpc-share/projects/CS46X_Project/training_data/split_v2/"
MODEL_PATH="/nfs/stak/users/minchle/hpc-share/projects/CS46X_Project/training_data/rice/rice_model_v2_rigorous.h5"
LOG_FILE="/nfs/stak/users/minchle/hpc-share/projects/CS46X_Project/training_log_$(date +%Y%m%d_%H%M%S).txt"

echo "Data directory: $DATA_DIR"
echo "Model output: $MODEL_PATH"
echo "Log file: $LOG_FILE"
echo "================================================"

# Run training with logging
/nfs/stak/users/minchle/miniconda3/envs/helixer_env/bin/HybridModel.py \
  --data-dir "$DATA_DIR" \
  --save-model-path "$MODEL_PATH" \
  --epochs 50 \
  --predict-phase \
  --gpu-id 0 \
  --batch-size 32 \
  2>&1 | tee "$LOG_FILE"

# Check if model was saved
if [ -f "$MODEL_PATH" ]; then
    echo "================================================"
    echo "SUCCESS: Model saved to $MODEL_PATH"
    ls -lh "$MODEL_PATH"
    echo "================================================"
else
    echo "================================================"
    echo "ERROR: Model file not found after training!"
    echo "Check the log file: $LOG_FILE"
    echo "================================================"
    exit 1
fi
