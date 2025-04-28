#!/bin/bash
logbase=gramschm
exe=./gramschmidt.exe
cmd="$exe $1 0 ${@:2}"
out=$cmd

../common/numa_exp.sh $logbase "$cmd" "$out"
