#!/bin/bash -xe

logbase=cg
exe=./conjugateGradientUM
cmd="$exe -s=$1 ${@:2}"
out="$exe $1 ${@:2}"
../common/numa_exp.sh $logbase "$cmd" "$out"
