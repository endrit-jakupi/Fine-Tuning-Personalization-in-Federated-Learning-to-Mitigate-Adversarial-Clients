# Fine-Tuning Personalization in Federated Learning to Mitigate Adversarial Clients

## Project Overview

This repository contains a reproduction of the experimental results from the paper titled [Fine-Tuning Personalization in Federated Learning to Mitigate Adversarial Clients](https://nips.cc/virtual/2024/poster/94850).

The project is done as part of the Distributed Deep Learning Systems course at the University of Bern. The goal of the project is to first reproduce the paper and secondly to come up with solutions to further improve performance metrics of it. 

## Reproduction Setup

The experiments were executed on the **UBELIX** high-performance computing cluster provided by the University of Bern. The following SLURM job file was used to reproduce the MNIST experiments:

**run_mnist.slurm**

```
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

# Navigate to your project directory
cd /storage/homefs/ej24n024/ddls/IPBML

# Run the MNIST experiment script
python3 run_mnist.py
```

Modify the following line to match the path to your own project directory:
```
cd /storage/homefs/ej24n024/ddls/IPBML
```

If you are not using UBELIX, you can run the experiments locally by simply executing:
```
python3 run_mnist.py
```
Note that with the specifications defined in the SLURM job file (RTX 3090 GPU, 16GB RAM, 4 CPUs), reproducing the results on UBELIX took approximately 4 hours. Running the experiments locally without similar resources may take significantly longer.

## Experiment Parameters

The script `run_mnist.py` automatically runs a series of federated learning experiments by looping over a grid of parameters:

- Number of Byzantine clients: `f ∈ {0, 3, 6, 9}`
- Number of clients per round: `m ∈ {8, 16}`
- Dirichlet distribution parameter: `α ∈ {1.0, 3.0, 100}`

Each unique configuration is repeated for **5 runs** (`nb_run = 5`) and trained for **100 communication rounds** (`T = 100`).

This results in a total of **24 unique experiment setups** (2 values of `m` × 3 values of `alpha` × 4 values of `f`), with potential extra result folders due to multiple runs of the same configuration.

## Reproduced Results

All experiment outputs are saved in the directory:

```
experiments/mnist/dirichlet_mnist
```
