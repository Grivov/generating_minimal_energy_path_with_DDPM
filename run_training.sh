#!/bin/bash
#SBATCH --array=0
#SBATCH --job-name=learn
#SBATCH --time=48:00:00
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=6
#SBATCH --mem-per-cpu=4GB
#SBATCH --partition=a100

module load anaconda
conda activate jupyter

python train.py
