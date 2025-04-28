#!/bin/bash
#SBATCH -N 1
#SBATCH -w voltron
#SBATCH -J eval
#SBATCH --exclusive
#SBATCH -t 72:00:00
#SBATCH -o slurm_out/%x.%j.out

export IGNORE_CC_MISMATCH=1
module load cuda/12.5


cd ../strategied/spmv-coo-graph/
make
./tmpexp.sh
cd -

cd ../strategied/conjugateGradientUM/
make 
./tmpexp.sh
cd -

cd ../strategied/tealeaf/
./build.sh
./tmpexp.sh
cd - 
