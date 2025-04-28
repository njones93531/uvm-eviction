#!/bin/bash
#SBATCH -N 1
#SBATCH -w voltron
#SBATCH -J bfs_exprmts
#SBATCH --exclusive
#SBATCH -t 1:00:00
#SBATCH -o %x.%j.out

export IGNORE_CC_MISMATCH=1
module load cuda gcc

python3 -u runBFS.py >> BFSData.txt
