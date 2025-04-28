#!/bin/bash

for method in "" "_nopf"
do
  input=conclusion_15_pref.txt
  figdir=../fault_plots/figures/metrics_pref
  if [ "$method" = "_nopf" ]; then
    figdir=../fault_plots/figures/metrics_nopf
    input=conclusion_15_nopf.txt
  fi

  python3 big_violin_serial.py $input -o $figdir

done
