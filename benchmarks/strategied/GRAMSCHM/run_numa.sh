#!/bin/bash
logbase=gramschm
exe=./gramschmidt.exe
cmd="$exe $1 0 $TIMEOUT ${@:2}"
out=$cmd

../common/numa_exp.sh $logbase "$cmd" "$out"
