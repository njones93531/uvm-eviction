#!/bin/bash 

base=../..
figdir_base=../figures/mempress
datadir=../figures/raw_data
method='pref' 'nopf'
do
  for bmark in FDTD-2D GRAMSCHM bfs-worst sgemm stream 
  do
    lc=$(echo $bmark | tr [:upper:] [:lower:])
    bench=(${base}/$bmark/${lc}_numa_${method}.data)
    echo $bench
    #echo "Bmark already done"
    python3 perf_data2csv.py $bench
    bench="${bench%.data}.csv"	
    cp $bench $datadir
    python3 mempress_plot.py $bench ${figdir_base}/$method
  done
done

