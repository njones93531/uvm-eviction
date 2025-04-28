#!/bin/bash
#SBATCH -N 1
#SBATCH -w voltron
#SBATCH -J cublas_exprmts
#SBATCH --exclusive
#SBATCH -t 36:00:00
#SBATCH -o %x.%j.out

export IGNORE_CC_MISMATCH=1
module load cuda gcc


for i in 6 7 8
do
	python3 -u runCublas.py > cublasData_$i.out
done
