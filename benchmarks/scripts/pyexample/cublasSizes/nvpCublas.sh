#!/bin/bash
#SBATCH -N 1
#SBATCH -w voltron
#SBATCH -J nvp_cublas
#SBATCH --exclusive
#SBATCH -t 24:00:00
#SBATCH -o %x.%j.out

export IGNORE_CC_MISMATCH=1
export TMPDIR="../../default/scripts/pyexample/cublasSizes/tmp/"

module load cuda gcc


python3 -u runCublas.py > nvp_cublas_data_1.out
sbatch nvpCublasCleanup.sh
