#!/bin/bash -xe

logbase=congrad
exe=./conjugateGradientUM
cmd="$exe -s=$1 ${@:2}"
out=$cmd
../common/numa_exp.sh $logbase "$cmd" "$out"
