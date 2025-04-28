#!/bin/bash
#SBATCH -N 1
#SBATCH -w voltron
#SBATCH -J ac-tracking
#SBATCH --exclusive
#SBATCH -t 1:00:00
#SBATCH -o %x.%j.out

export IGNORE_CC_MISMATCH=1
module load cuda

python3 -u benchmark_exps.py

# at least set module to default config; should probably reinstall vanilla
sudo rmmod nvidia-uvm
sudo modprobe nvidia-uvm #uvm_perf_prefetch_enable=0 uvm_perf_fault_coalesce=0 #uvm_perf_fault_batch_count=1
