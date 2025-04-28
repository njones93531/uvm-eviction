#!/bin/bash


driver=x86_64-535.104.05_faults
base=../../benchmarks/default
figdir=figures/metrics_nopf/faults-per-mig
mkdir $figdir



for bmark in FDTD-2D stream
do
   for size in 9 12 15 18 21
   do 
      bench=$base/$bmark/log_${driver}_nopf_${size}_${bmark}/${bmark}_klog.txt 
      tmp=$base/$bmark/log_${driver}_nopf_${size}_${bmark}/tmp_klog.txt 
      head -60000000 $bench > $tmp
      python3 faults_per_VABlock.py $tmp ${figdir}/${bmark}_${size}_faults_per_mig_per_VABlock.png
      rm $tmp
   done
done

