#!/bin/bash
#SBATCH -N 1
#SBATCH -w voltron
#SBATCH -J cublas_exprmts
#SBATCH --exclusive
#SBATCH -t 2:30:00
#SBATCH -o %x.%j.out


export IGNORE_CC_MISMATCH=1
module load cuda gcc


nvcc mem_user.cu
./a.out 8 9000 &

for run in 0 1 2 3 4 
do	
	rm cublas_x86_64-520.61.05_vanilla_4gb_relmem_$run.txt
	for a in 13000 15500 18000 20500 23000 26000 29000 32000 35000 38000 41000 44000
	do 
		~/uvm-eviction/benchmarks/matmul/matmulDef/matrixMul2 -wA=$a -hA=$a -wB=$a -hB=$a >> cublas_x86_64-520.61.05_vanilla_4gb_relmem_$run.txt
	done
	python3 parser.py cublas_x86_64-520.61.05_vanilla_4gb_relmem_$run.txt > relMemParsed$run.tmp   
    	nvidia-smi	
done

python3 mean_finder.py relMemParsed0.tmp relMemParsed1.tmp relMemParsed2.tmp relMemParsed3.tmp relMemParsed4.tmp >  cublas_x86_64-520.61.05_vanilla_4gb_relmem_avg.data

for run in 0 1 2 3 4 
do 
	rm relMemParsed$run.tmp
done
