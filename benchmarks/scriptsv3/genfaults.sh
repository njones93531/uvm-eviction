#!/bin/bash
#SBATCH -N 1
#SBATCH -w voltron
#SBATCH -J faults
#SBATCH --exclusive
#SBATCH -t 72:00:00
#SBATCH -o slurm_out/%x.%j.out

export IGNORE_CC_MISMATCH=1
module load cuda/12.5
module load mpi

export TIMEOUT=12000
python3 -u faults.py

# at least set module to default config; should probably reinstall vanilla
sudo rmmod nvidia-uvm
sudo modprobe nvidia-uvm #uvm_perf_prefetch_enable=0 uvm_perf_fault_coalesce=0 #uvm_perf_fault_batch_count=1
