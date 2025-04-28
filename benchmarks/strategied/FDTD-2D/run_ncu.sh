#!/bin/bash

ncu --log-file hhm_ncu_$1.log --replay-mode application --launch-count 1 ./fdtd2d.exe $1 0 -p hhm
ncu --log-file hhd_ncu_$1.log --replay-mode application --launch-count 1 ./fdtd2d.exe $1 0 -p hhd
