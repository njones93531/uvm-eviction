#!/bin/bash 
#SBATCH -N 1
#SBATCH -w ivy
#SBATCH -J fault_plot
#SBATCH -t 01:00:00


method="_nopf"
output=fault_data${method}.csv
output_bkp=${output%.csv}.bkp
driver=x86_64-535.104.05
base=../../benchmarks/default

##Move old output to the backup, then delete it
char="-"
line=$(printf "%0.s$char" {1..40})
echo $line >> $output_bkp
date >> $output_bkp
echo $line >> $output_bkp
cat $output >> $output_bkp
rm $output


#Append Data to csv
files=()
i=0
for bmark in stream FDTD-2D bfs-worst cublas GEMM GRAMSCHM
do
  for psize in 9 12 15 18 21 
  do 
    bench=(${base}/$bmark/log_${driver}_faults${method}_${psize}_${bmark}/${bmark}_klog.txt)
    files[$i]=$bench
    ((++i))
  done
done

python3 fault_csv.py --print_header ${files[@]} > ${output}

