#!/bin/bash
#SBATCH -N 1
#SBATCH -w voltron
#SBATCH -J faults
	###SBATCH --exclusive
#SBATCH -t 1:00:00
#SBATCH -o slurm_out/%x.%j.out

export IGNORE_CC_MISMATCH=1
module load cuda/12.5
module load mpi

#cudaMalloc -> cudaMallocManaged
input_file="../default/spmm-csr/spmm_csr_explicit.c"
output_file="../default/spmm-csr/spmm_csr_managed.c"

bash auto_conv.sh $input_file $output_file

export TIMEOUT=12000
python3 -u auto_logs.py

# at least set module to default config; should probably reinstall vanilla
#sudo rmmod nvidia-uvm
#sudo modprobe nvidia-uvm #uvm_perf_prefetch_enable=0 uvm_perf_fault_coalesce=0 #uvm_perf_fault_batch_count=1
