#!/bin/bash

# run this script from project root, so you get root/models/ -- otherwise models/ will be created in in scripts/setup/

set -e
read -p "Conda env name [Press 'enter' for default: evo2]: " ENV_NAME
ENV_NAME=${ENV_NAME:-evo2}
  
conda create -y -n "$ENV_NAME" python=3.12
source "$(conda info --base)/etc/profile.d/conda.sh"
conda activate "$ENV_NAME"


python -m pip install --no-cache-dir --index-url https://download.pytorch.org/whl/cu128 torch==2.7.1 torchvision==0.22.1 torchaudio==2.7.1
  
if [ ! -d "models/Evo2/.git" ]; then
    git clone https://github.com/ArcInstitute/evo2 models/Evo2
else
    echo "Directory models/Evo2 already exists, skipping git clone."
fi

pip install -e models/Evo2

conda install -y -c conda-forge nccl

pip3 install --no-build-isolation transformer_engine[pytorch]

pip install ninja psutil
pip install --no-cache-dir --no-build-isolation flash-attn

pip install biopython bcbio-gff omegaconf pandas
