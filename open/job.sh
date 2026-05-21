#!/bin/bash
#SBATCH -A c_gnn_001               # Account name to be debited
#SBATCH --job-name=gnn         # Job name
#SBATCH --time=0-01:00:00        # Maximum walltime (30 minutes)
#SBATCH --partition=cpu     # Select the ai partition
## --gres=gpu:1          # Request 1 to 4 GPUs per node
#SBATCH --mem-per-cpu=80000       # Memory per CPU core (16 GB)
#SBATCH --nodes=1               # Request 1 node

python3 test3.py
