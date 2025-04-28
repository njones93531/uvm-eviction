#!/bin/bash 
#SBATCH -N 1
#SBATCH -w ivy
#SBATCH -J fault_plot
#SBATCH -t 01:00:00

output=faults-evictions-pref.data
#base=/home/najones/uvm-eviction/benchmarks
driver=x86_64-535.104.05
#figdir=/home/najones/uvm-eval/figures/fault-eviction
echo Begin: >> $output

#polybench
base=../../benchmarks/default
for bmark in GEMM GRAMSCHM FDTD-2D MVT bfs-worst cublas stream
do
	for psize in 15
	do
		bench=(${base}/$bmark/log_${driver}_faults_${psize}_${bmark}/${bmark}_klog.txt)
                echo $bench
                time python3 faults_evictions_table.py -t  ${bench} >> $output 
                
	done
done

for bmark in spmv-coo-twitter7 
do
	bench=(${base}/$bmark/log_${driver}_faults_${bmark}/${bmark}_klog.txt)
	echo $bench
	time python3 faults_evictions_table.py -t  ${bench} >> $output 
done

