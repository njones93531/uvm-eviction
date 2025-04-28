#!/bin/bash
#SBATCH -N 1
#SBATCH -w voltron
#SBATCH -J touch_all
#SBATCH --exclusive
#SBATCH -t 24:00:00
#SBATCH -o %x.%j.out

export IGNORE_CC_MISMATCH=1
module load cuda

make all
