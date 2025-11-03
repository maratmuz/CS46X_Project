conda create -n evo2 python=3.12

cd evo2/
pip install -e .

module load cuda/12.8
module load cudnn/8.9_cuda12

conda install -c conda-forge nccl

pip3 install --no-build-isolation transformer_engine[pytorch]

pip install flash-attn==2.7.3

pip install -r requirements.txt