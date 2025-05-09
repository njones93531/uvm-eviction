#!/bin/bash 

base=../../common/figures
datadir=$base/raw_data
for dir in $base $datadir
do

  # Check if the directory exists
  if [ ! -d "$dir" ]; then
    # If not, create the directory
    mkdir -p "$dir"
    echo "Directory created: $dir"
  else
    echo "Directory already exists: $dir"
  fi
done

#polybench
base=../..
pref=_pref
for bmark in FDTD-2D GRAMSCHM stream sgemm bfs-worst spmv-coo-twitter7
do
	lc=$(echo $bmark | tr [:upper:] [:lower:])
	bench=(${base}/$bmark/${lc}_numa${pref}.data)
	echo $bench
	#echo "Bmark already done"
	python3 perf_data2csv.py $bench
	bench="${bench%.data}.csv"	
	cp $bench $datadir/
	python3 oversub_plot.py $bench 
done

