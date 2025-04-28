#!/bin/bash -xe
#SBATCH -N 1
#SBATCH -w voltron
#SBATCH -J async-evict-perftest
#SBATCH --exclusive
#SBATCH -t 72:00:00
#SBATCH -o %x.%j.out

log_dir="logs"
mkdir -p $log_dir

run_experiment() {
    local size=$1
    local iteration=$2

    local vanilla_log_file="$log_dir/cublas-vanilla-$size-$iteration.log"
    local asyncevict_log_file="$log_dir/cublas-asyncevict-$size-$iteration.log"

    if [ ! -f "$vanilla_log_file" ]; then
        ./cublas-vanilla.sh $size |& tee "$vanilla_log_file"
    else
        echo "File $vanilla_log_file already exists, skipping..."
    fi

    if [ ! -f "$asyncevict_log_file" ]; then
        ./cublas-asyncevict.sh $size |& tee "$asyncevict_log_file"
    else
        echo "File $asyncevict_log_file already exists, skipping..."
    fi
}


extract_performance_metrics() {
    local output_file="cublas_perf_values.csv"
    echo "Size,Iteration,Performance" > $output_file

    for log_file in $log_dir/*.log
    do
        size=$(echo $log_file | grep -oP '(?<=cublas-vanilla-).*(?=-\d+.log)')
        iteration=$(echo $log_file | grep -oP '(?<=-)\d+(?=.log)')
        performance=$(grep "perf," $log_file | cut -d',' -f2)

        echo "$size,$iteration,$performance" >> $output_file
    done
}

NUM_ITERATIONS=5
for ((multiplier=4; multiplier<=36; multiplier++))
do
    size=`expr 1024 \* $multiplier`
    for ((i=1; i<=$NUM_ITERATIONS; i++))
    do
        run_experiment $size $i
    done
done

sync
sleep 5
extract_performance_metrics

# Invoke Python script to generate the plot
python3 generate_cublas_chart.py

