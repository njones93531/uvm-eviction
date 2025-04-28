#!/bin/bash
#SBATCH -N 1
#SBATCH -w voltron
#SBATCH -J cublas-perf-async
#SBATCH --exclusive
#SBATCH -t 6:00:00
#SBATCH -o %x.%j.out

export IGNORE_CC_MISMATCH=1
module load cuda
python3 run.py
