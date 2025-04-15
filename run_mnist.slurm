#!/bin/bash
#SBATCH --job-name=mnist_experiment
#SBATCH --output=mnist_%j.out
#SBATCH --error=mnist_%j.err
#SBATCH --time=10:00:00
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --mem=16G
#SBATCH --partition=gpu
#SBATCH --gpus-per-node=rtx3090:1

# Load shell environment
source ~/.bashrc

# Activate the Conda environment
conda activate ddls

# Navigate to your project directory
cd /storage/homefs/ej24n024/ddls/IPBML

# Run the MNIST experiment script
python3 run_mnist.py
