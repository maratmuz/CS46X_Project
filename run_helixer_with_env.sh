#!/bin/bash
# Wrapper script to run Helixer.py with proper environment
export PATH="/nfs/stak/users/minchle/miniconda3/envs/helixer_env/bin:$PATH"
export LD_LIBRARY_PATH="/lib64:/nfs/stak/users/minchle/miniconda3/envs/helixer_env/lib:$LD_LIBRARY_PATH"

/nfs/stak/users/minchle/miniconda3/envs/helixer_env/bin/Helixer.py "$@"
