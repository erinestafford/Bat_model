#!/bin/bash

#SBATCH -A hpc2n2024-020
#SBATCH -n 1
#SBATCH -c 1
#SBATCH -t 24:00:00
#SBATCH -a 0-100
#SBATCH --mem-per-cpu=40G
#SBATCH -o results/res_slurm/%A_%a.txt


ml GCC/12.3.0 Python/3.11.3
ml SciPy-bundle/2023.07

python ./main.py

