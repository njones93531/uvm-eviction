#!/bin/bash 

base=../..
method='pref'
for bmark in FDTD-2D GRAMSCHM bfs-worst sgemm stream tealeaf conjugateGradientUM MVT 
do
  lc=$(echo $bmark | tr [:upper:] [:lower:])
  bench=(${base}/$bmark/${lc}_numa_${method}.data)
  echo $bench
  #echo "Bmark already done"
  python3 perf_data2csv.py $bench
  bench="${bench%.data}.csv"	
done

