#!/bin/bash 
#SBATCH -N 1
#SBATCH -w ivy
#SBATCH -J fault_plot
#SBATCH -t 01:00:00


driver=x86_64-535.104.05
base=../../benchmarks/default
for bmark in stream cublas GRAMSCHM GEMM bfs-worst MVT FDTD-2D
do
	figdir=../fault_plots/figures
	for psize in 15
	do 
		prefbench=(${base}/$bmark/log_${driver}_faults_${psize}_${bmark}/${bmark}_klog.txt)
		nopfbench=(${base}/$bmark/log_${driver}_faults_nopf_${psize}_${bmark}/${bmark}_klog.txt)
		echo $bench
		time python3 polyfit.py $prefbench $nopfbench -o ${figdir}
	done
done

bmark=spmv-coo-twitter7
figdir=../fault_plots/figures
prefbench=(${base}/$bmark/log_${driver}_faults_${bmark}/${bmark}_klog.txt)
nopfbench=(${base}/$bmark/log_${driver}_faults_nopf_${bmark}/${bmark}_klog.txt)
echo $bench
time python3 polyfit.py $prefbench $nopfbench -o ${figdir}

#for bmark in stream GEMM
#do
#	figdir=../fault_plots/figures
#	for psize in 9
#	do 
#		prefbench=(${base}/$bmark/log_${driver}_faults_${psize}_${bmark}/${bmark}_klog.txt)
#		nopfbench=(${base}/$bmark/log_${driver}_faults_nopf_${psize}_${bmark}/${bmark}_klog.txt)
#		echo $bench
#		time python3 polyfit.py $prefbench $nopfbench -o ${figdir}
#	done
#done
#

