#!/bin/bash
#SBATCH -N 1
#SBATCH -w voltron
#SBATCH -J eviction-ac-tracking
#SBATCH --exclusive
#SBATCH -t 24:00:00
#SBATCH -o %x.%j.out

export IGNORE_CC_MISMATCH=1
module load cuda
#FIXME this isn't a great way to do this because diff apps may need different mpi modules. how fix?
module load mpi/openmpi-x86_64
python3 -u run.py 
