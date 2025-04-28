#!/bin/bash
logbase=bfs-worst
sizes=$(python3 genSizes.py $1)
exe=./bfs-worst
cmd="$exe $sizes ${@:2}"
out="$exe $1 $sizes ${@:2}"

../common/numa_exp.sh $logbase "$cmd" "$out"
