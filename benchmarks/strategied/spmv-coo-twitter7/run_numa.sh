#!/bin/bash
logbase=spmv-coo-twitter7
exe=./spmv_coo
graph=/home/share/cusparse_graphs/twitter7.mtx
cmd="$exe $graph ${@:2}"
out="$exe 16.72 $graph ${@:2}"

../common/numa_exp.sh $logbase "$cmd" "$out"


