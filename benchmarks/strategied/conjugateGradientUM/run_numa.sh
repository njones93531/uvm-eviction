#!/bin/bash -xe

logbase=conjugategradientum
exe=./conjugateGradientUM
cmd="$exe -s=$1 ${@:2}"
out="$exe $1 ${@:2}"
../common/numa_exp.sh $logbase "$cmd" "$out"
