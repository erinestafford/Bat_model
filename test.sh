#!/bin/bash

#SBATCH -A hpc2n2024-020
#SBATCH -n 1
#SBATCH -c 1
#SBATCH -t 0:01:00
#SBATCH -a 0-10
#SBATCH -o results/res_slurm/%A_%a.txt


ml GCC/12.3.0 Python/3.11.3
ml SciPy-bundle/2023.07
ml matplotlib/3.7.2

python ./hello_world.py