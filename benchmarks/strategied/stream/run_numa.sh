#!/bin/bash

logbase=stream
psize=$(python3 genSizes.py $1)
cmd="./cuda-stream --float --triad-only -n 10 -s $psize ${@:2}" 
out="./cuda-stream $1  --float --triad-only -n 10 -s $psize ${@:2}"

../common/numa_exp.sh $logbase "$cmd" "$out"
