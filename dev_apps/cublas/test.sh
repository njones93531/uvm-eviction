#!/bin/bash -xe
module load cuda

# Command to query the compute capability
output=$(nvidia-smi --query-gpu=compute_cap --format=csv)
# Extract the second line of output (first line of actual data)
second_line=$(echo "$output" | sed -n '2p')

# Compare the extracted number with 7.5
if (( $(echo "$second_line >= 7.5" | bc -l) )); then
    srcv="kernel-open"
else
    srcv="kernel"
fi

for i in "faults-new"; do
#for i in "vanilla" "faults" "faults-new"; do
#for i in  "faults-new" "faults" "vanilla"; do
#for i in  "faults-new"  "vanilla"; do
    make -j -C ~/dev/uvm-eviction/drivers/x86_64-555.42.02/${i}/${srcv} &> ${i}-kbuild.log
    sudo rmmod -f nvidia-uvm || true
    sudo insmod ~/dev/uvm-eviction/drivers/x86_64-555.42.02/${i}/${srcv}/nvidia-uvm.ko uvm_perf_prefetch_enable=0 hpcs_log_short=1
    if [ "$i" == "faults-new" ]; then
        echo "Waiting for logging interface to initialize"
        while [ ! -e /dev/hpcs_logger ]; do sleep 1; done
        ../../tools/sysloggerv2/log "junk.txt" &
        pid=$!
    fi

    time ./sgemm -n `expr 4096 \* 6` | tee ${i}.log


    if [ "$i" == "faults-new" ]; then
        kill $pid
    fi
done

sudo rmmod nvidia-uvm
