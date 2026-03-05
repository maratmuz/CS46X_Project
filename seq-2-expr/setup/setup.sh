#!/bin/bash
# setup_env.sh
conda env create -f environment.yaml
eval "$(conda shell.bash hook)"
conda activate nt
mkdir -p $CONDA_PREFIX/etc/conda/activate.d
echo 'export LD_LIBRARY_PATH=$CONDA_PREFIX/lib:$LD_LIBRARY_PATH' > $CONDA_PREFIX/etc/conda/activate.d/env_vars.sh
conda deactivate