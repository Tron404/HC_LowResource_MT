#!/bin/bash
#SBATCH --nodes=1
#SBATCH --tasks-per-node=6
#SBATCH --time=01:00:00
#SBATCH --job-name=jupyter
#SBATCH --mem=16G
#SBATCH --partition=regular

# Clear the module environment
module purge
# Load the Python version that has been used to construct the virtual environme$
# we are using below
module load Python/3.10.8-GCCcore-12.2.0

# Activate the virtual environment
source ~/virtual_env/HC/bin/activate

jupyter notebook --no-browser --ip=$( hostname )
