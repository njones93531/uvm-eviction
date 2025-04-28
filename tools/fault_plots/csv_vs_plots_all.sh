#!/bin/bash

x=Degree_of_Subscription
y=all
outdir=../fault_plots/figures/metrics_nopf/vs
python3 csv_vs_plots.py $1 $x $y -x -y -o $outdir/lin_scale 
python3 csv_vs_plots.py $1 $x $y -x -y -l -o $outdir/log_scale

for bench in FDTD-2D stream cublas GEMM GRAMSCHM bfs-worst
do
  python3 csv_vs_plots.py $1 $x $y -a -x -y -o $outdir/lin_scale -b $bench
  python3 csv_vs_plots.py $1 $x $y -a -x -y -l -o $outdir/log_scale -b $bench
done

