#!/bin/bash


for method in 'pref' 'nopf'
do
  for bmark in FDTD-2D GRAMSCHM bfs-worst sgemm stream MVT tealeaf conjugateGradientUM
  do
    lc=$(echo $bmark | tr [:upper:] [:lower:])
    datalog=(../$bmark/${lc}_numa_${method}.data)
    bkp=${datalog%data}bkp
    date >> $bkp
    cat $datalog >> $bkp
    echo "removing $datalog"
    rm $datalog
  done
done

