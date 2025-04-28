#!/bin/bash
#SBATCH -N 1
#SBATCH -w voltron
#SBATCH -J plot_zero_copy
#SBATCH --exclusive
#SBATCH -t 10:00:00
#SBATCH -o %x.%j.out

export IGNORE_CC_MISMATCH=1
module load cuda
#pip3 install matplotlib --user
#pip3 install natsort --user
pip3 install numpy --user
#python3 zero_copy.py ~/uvm-eviction/tools/zero_copy_plots/data/x86_64-460.27.04_ac-tracking-full_mimc_momc_8192_klog_cublas.txt
python3 zero_copy_plots.py #-o figure.png ~/uvm-eviction/tools/zero_copy_plots/data/x86_64-460.27.04_ac-tracking-full_mimc_momc_8192_klog_cublas.txt
