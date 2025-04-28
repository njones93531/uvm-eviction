#!/bin/bash -xe
#SBATCH -N 1
#SBATCH -w voltron
#SBATCH -J async-evict-perftest
#SBATCH --exclusive
#SBATCH -t 24:00:00
#SBATCH -o %x.%j.out

log_dir="logs"
mkdir -p $log_dir

run_experiment() {
    local size=$1
    local iteration=$2

    # Define all case types and iterate through them
    for case_type in "vanilla" "preempt" "preempt-conservative"; do
        local log_file="$log_dir/cublas-$case_type-$size-$iteration.log"
        
        if [ ! -f "$log_file" ]; then
            ./cublas.sh $case_type $size |& tee "$log_file"
        else
            echo "File $log_file already exists, skipping..."
        fi
    done
}

extract_performance_metrics() {
    local output_file="cublas_perf_values.csv"
    echo "Size,Iteration,Performance,Case" > $output_file

    for log_file in $log_dir/*.log
    do
        case_type=$(echo $log_file | grep -oP '(?<=cublas-).*(?=-\d+-\d+.log)')
        size=$(echo $log_file | grep -oP '(?<=cublas-'$case_type'-).*(?=-\d+.log)')
        iteration=$(echo $log_file | grep -oP '(?<=-)\d+(?=.log)')
        performance=$(grep "perf," $log_file | cut -d',' -f2)

        echo "$size,$iteration,$performance,$case_type" >> $output_file
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

