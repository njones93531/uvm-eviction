#!/bin/bash
#SBATCH -N 1
#SBATCH -w ivy
#SBATCH -J fault_plot
#SBATCH -t 01:00:00

module load julia



if ! [ -f sys_plots_`hostname`.so ]; then
    cd ..
    ./precompile.sh
    cd -
fi

echo $outname

size="1.5"
small="0.5"
format="png"
driver=x86_64-535.104.05
threshold=60000000

base=../../benchmarks/default
for method in "" "_nopf"
do
  figdir=../fault_plots/figures/fault_plots_pref
  if [ "$method" = "_nopf" ]; then
    figdir=../fault_plots/figures/fault_plots_nopf
  fi

  #ensure figdir exists
  if [ -d "$figdir" ]; then
    echo "figdir exists"
  else
    mkdir $figdir
  fi


  #Do all bmarks that have a psize
  for bmark in stream FDTD-2D bfs-worst cublas GEMM GRAMSCHM MVT
  do
    for psize in 15
    do
      
        bench=${base}/$bmark/log_${driver}_faults${method}_${psize}_${bmark}/${bmark}_klog.txt
        echo $bench	
        head -$threshold ${bench} > temp_klog.txt
        bench=temp_klog.txt
        time julia --compile=all -O3  --sysimage sys_plots_`hostname`.so  ./pattern_plot.jl -m ','  -n ${figdir}/${bmark}_${psize}.${format} ${bench} -s $size -i ${bmark}_${psize}
        time julia --compile=all -O3  --sysimage sys_plots_`hostname`.so  ./pattern_plot.jl -m ','  -n ${figdir}/${bmark}_${psize}_time.${format} ${bench} -s $size -i ${bmark}_${psize} -t
        rm temp_klog.txt
    done
	done

  #SPMV twitter has no psize 
  bmark=spmv-coo-twitter7
  bench=${base}/$bmark/log_${driver}_faults${method}_${bmark}/${bmark}_klog.txt


  head -$threshold ${bench} > temp_klog.txt
  bench=temp_klog.txt
  time julia --compile=all -O3  --sysimage sys_plots_`hostname`.so  ./pattern_plot.jl -m ','  -n ${figdir}/${bmark}_${psize}.${format} ${bench} -s $size -i ${bmark}_${psize}
  time julia --compile=all -O3  --sysimage sys_plots_`hostname`.so  ./pattern_plot.jl -m ','  -n ${figdir}/${bmark}_${psize}_time.${format} ${bench} -s $size -i ${bmark}_${psize} -t

done


