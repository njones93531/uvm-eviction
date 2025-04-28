#!/bin/bash 
#SBATCH -N 1
#SBATCH -w ivy
#SBATCH -J fault_plot
#SBATCH -t 01:00:00


driver=x86_64-535.104.05
base=../../benchmarks/default

for method in "" "_nopf"
do
  figdir=../fault_plots/figures/metrics_pref
  if [ "$method" = "_nopf" ]; then
    figdir=../fault_plots/figures/metrics_nopf
  fi


  for bmark in stream cublas GRAMSCHM GEMM bfs-worst MVT FDTD-2D
  do
    for psize in 15
    do 
      bench=(${base}/$bmark/log_${driver}_faults${method}_${psize}_${bmark}/${bmark}_klog.txt)
      echo $bench
      time python3 density.py ${bench} -o ${figdir}
    done
  done

  bmark=spmv-coo-twitter7
  bench=(${base}/$bmark/log_${driver}_faults${method}_${bmark}/${bmark}_klog.txt)
  echo $bench
  time python3 density.py ${bench} -o ${figdir}
done
