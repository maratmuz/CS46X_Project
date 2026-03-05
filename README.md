# CS46X_Project

EVO2 Model and any scripts for pulling data, assisting in training, and assisting in testing. 

Student Team: 
- Aiden Gabriel (gabrieai@oregonstate.edu)
- Jared Lim (limjar@oregonstate.edu)
- Caleb Lowe (lowecal@oregonstate.edu)
- Levi Minch (minchle@oregonstate.edu)
- Marat Muzaffarov (muzaffam@oregonstate.edu)

Sponsor: Ken Janik (ken.janik@gmail.com)

Mentors:
- Professor Pankaj Jaiswal (pankaj.jaiswal@oregonstate.edu)
- Professor Molly Megraw (molly.megraw@oregonstate.edu)


# Setup Evo2

Run the following command from the project root to setup the conda environment for Evo2.
```
./scripts/setup/setup_evo2_conda.sh
```

You will likely also need to set CUDA and cuDNN versions before running the setup script, we use `CUDA 12.8` and `cuDNN 8.9`.
On the Oregon State HPC cluster, or any cluster with Lmod, these can be set with the following commands.
```
module load cuda/12.8
module load cudnn/8.9_cuda12
```
