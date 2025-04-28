#!/bin/bash -xe
#SBATCH -N 1
#SBATCH -w voltron
#SBATCH -J async-evict-perftest
#SBATCH --exclusive
#SBATCH -t 24:00:00
#SBATCH -o %x.%j.out

log_dir="logs"
config_dir="config"
mkdir -p $log_dir

# Function to check if the module is still in use
module_is_in_use() {
    if lsmod | grep "^nvidia-uvm" > /dev/null; then
        local use_count=$(lsmod | grep "^nvidia-uvm" | awk '{print $2}')
        [ "$use_count" -gt "0" ]
        sudo rmmod -f nvidia-uvm
        if [ "$?" -eq 0 ]; then
            return 0
        fi
    fi
    return 1
}

update_kernel_module() {
    local case_type=$1
    DIR="../../drivers/x86_64-535.104.05/$case_type/kernel"

    if [ ! -d "$DIR" ]; then
        echo "Error: Directory $DIR does not exist."
        exit 1
    fi

    cd "$DIR"

    make -j
    sudo make modules_install
    cd -

    local end_time=$((SECONDS+300))

    while module_is_in_use; do
        if [ $SECONDS -gt $end_time ]; then
            echo "Timeout reached, exiting script."
            exit 1
        fi
        sleep 5
    done

    sudo rmmod -f nvidia-uvm
    sudo modprobe nvidia-uvm
}

run_experiment() {
    local benchmark=$1
    local executable=$2
    local size_label=$3
    local size_args=$4
    local iteration=$5
    local case_type=$6
    local last_case_type=$7

    if [ "$last_case_type" != "$case_type" ]; then
        update_kernel_module $case_type
    fi

    local log_file="$log_dir/$benchmark-$size_label-$case_type-$iteration.log"
    
    if [ ! -f "$log_file" ]; then
        ../$benchmark/$executable $size_args |& tee "$log_file"
    else
        echo "File $log_file already exists, skipping..."
    fi
}

extract_performance_metrics() {
    for benchmark in "${!benchmarks[@]}"; do
        local output_file="${log_dir}/${benchmark}_perf_values.csv"
        echo "SizeLabel,Iteration,Performance,Case" > $output_file

        for log_file in $log_dir/${benchmark}-*.log; do
            local filename=$(basename "$log_file")
            local size_label=$(echo "$filename" | awk -F '-' '{print $2}')
            local iteration=$(echo "$filename" | grep -oP '\d+(?=\.log)')

            # Extract case_type without size_label
            local case_type=$(echo "$filename" | sed -e "s/^${benchmark}-${size_label}-//" -e "s/-${iteration}\.log$//")

            local performance=$(grep "perf," "$log_file" | cut -d',' -f2)

            echo "$size_label,$iteration,$performance,$case_type" >> $output_file
        done
    done
}


declare -A benchmarks
benchmarks=(["cublas"]="matrixMul2" ["stream"]="cuda-stream" ["linear"]="page" ["random"]="page")

NUM_ITERATIONS=5
last_case_type=""

for benchmark in "${!benchmarks[@]}"; do
    executable=${benchmarks[$benchmark]}
    for case_type in "vanilla" "preempt" "preempt-conservative"; do
        while IFS=, read -r size_label size_args; do
            for ((i=1; i<=$NUM_ITERATIONS; i++)); do
                run_experiment $benchmark $executable "$size_label" "$size_args" $i $case_type $last_case_type
                last_case_type=$case_type
            done
        done < "$config_dir/$benchmark.conf"
    done
done

sync
sleep 5
extract_performance_metrics

# Loop to generate the chart for each CSV
for csv_file in ${log_dir}/*_perf_values.csv; do
    python3 generate_chart.py "$csv_file"
done

