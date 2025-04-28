#!/bin/bash 

base=../..
figs=../../../../figs/voltron
figdir_base=$figs/migration
datadir=../figures/raw_data
for method in 'pref' #'nopf'
do
  for bmark in FDTD-2D GRAMSCHM bfs-worst cublas stream 
  do
    lc=$(echo $bmark | tr [:upper:] [:lower:])
    bench=(${base}/$bmark/${lc}_numa_${method}.data)
    echo $bench
    #echo "Bmark already done"
    python3 perf_data2csv.py $bench
    bench="${bench%.data}.csv"	
    cp $bench $datadir
    python3 migration_oversub_plot.py $bench ${figdir_base}/$method
  done
done

