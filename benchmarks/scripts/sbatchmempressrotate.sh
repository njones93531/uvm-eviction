#!/bin/bash
#SBATCH -N 1
#SBATCH -w voltron
#SBATCH -J mempress
#SBATCH --exclusive
#SBATCH -t 48:00:00
#SBATCH -o %x.%j.out

export IGNORE_CC_MISMATCH=1
module load cuda/12.2
module load gcc/12.2.0
#module load rocm
#export HIP_PLATFORM=nvidia
export TIMEOUT=7200
export ITERS=2
python3 -u mem_pressure_rotate_all.py
