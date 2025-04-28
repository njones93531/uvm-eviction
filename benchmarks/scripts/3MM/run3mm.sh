#!/bin/bash
#SBATCH -N 1
#SBATCH -w voltron
#SBATCH -J 3mm_exprmts
#SBATCH --exclusive
#SBATCH -t 36:00:00
#SBATCH -o %x.%j.out

export IGNORE_CC_MISMATCH=1
module load cuda gcc

python3 -u run3mm.py > 3mmData.txt
