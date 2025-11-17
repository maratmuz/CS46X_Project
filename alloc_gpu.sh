#!/bin/bash
# Make sure to `chmod +x alloc_gpu.sh` this script if it does not work!
# request allocation
salloc --partition=dgxh --gres=gpu:1 --mem=16g --time=12:00:00 <<'EOF'
hostname > /tmp/alloc_node_$USER.txt
EOF

# read the assigned node
NODE=$(cat /tmp/alloc_node_$USER.txt)
echo "Allocated node: $NODE"

# export CUDA device
export CUDA_VISIBLE_DEVICES=0

# SSH into the allocated node
echo "Connecting..."
ssh $NODE
