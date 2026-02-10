#!/bin/bash
#SBATCH -N 1
#SBATCH -w voltron
#SBATCH -J faults
#SBATCH --exclusive
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

cd ~/uvm-eviction/tools/fault_plotsv3

# TODO get rid of hardcoded nonsense
# 	may make more sense doing in auto_logs
python3 metric_plot.py ../../benchmarks/default/spmm-csr/x86_64-555.42.02_faults-new_1_spmm-csr/klog_spmm-csr.txt -o metrics_stats_relative

# at least set module to default config; should probably reinstall vanilla
# 	done in auto_logs.py
#sudo rmmod nvidia-uvm
#sudo modprobe nvidia-uvm #uvm_perf_prefetch_enable=0 uvm_perf_fault_coalesce=0 #uvm_perf_fault_batch_count=1
