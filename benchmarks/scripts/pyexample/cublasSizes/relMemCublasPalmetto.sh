#!/bin/bash
#PBS -N cuda
#PBS -l select=1:ncpus=64:ngpus=2:gpu_model=a100:mem=250gb 
#PBS -l walltime=02:30:00

export IGNORE_CC_MISMATCH=1
module load cuda/11.6.2-gcc/9.5.0
#pip3 install --user sh

cd $PBS_O_WORKDIR

nvcc mem_user.cu
./a.out 76 9000 &

for run in 0 1 2 3 4
do
	for a in 13000 15500 18000 20500 23000 26000 
	do 
		./matrixMul2 -wA=$a -hA=$a -wB=$a -hB=$a >> relMemData$run.txt
	done
done
