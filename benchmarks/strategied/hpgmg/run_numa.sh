#!/bin/bash -xe

logbase=hpgmg
exe=./build/bin/hpgmg-fv
cmd="$exe $1 ${@:2}"
out=$cmd
../common/numa_exp.sh $logbase "$cmd" "$out"
