#!/bin/bash 
#SBATCH -N 1
#SBATCH -w ivy
#SBATCH -J fault_plot
#SBATCH -t 01:00:00


driver=x86_64-535.104.05
base=../../benchmarks/default
for bmark in bfs-worst MVT FDTD-2D stream cublas GRAMSCHM GEMM 
do
	for method in "_nopf"
	do
		for psize in 15
		do 
			bench=(${base}/$bmark/log_${driver}_faults${method}_${psize}_${bmark}/${bmark}_klog.txt)
			tmp=(${base}/$bmark/log_${driver}_faults${method}_${psize}_${bmark}/tmp_klog.txt)
			echo $bench
			head -60000000 $bench > $tmp
			time python3 Fault.py $tmp	
			time python3 faulttest.py $tmp
			rm $tmp
		done
	done
done

bmark=spmv-coo-twitter7
for method in "_nopf"
do
	bench=(${base}/$bmark/log_${driver}_faults${method}_${bmark}/${bmark}_klog.txt)
	tmp=(${base}/$bmark/log_${driver}_faults${method}_${bmark}/tmp_klog.txt)
	echo $bench
	head -60000000 $bench > $tmp
	time python3 Fault.py $tmp	
	time python3 faulttest.py $tmp
	rm $tmp
done
