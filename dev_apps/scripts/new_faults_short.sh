#!/bin/bash -ex
#SBATCH -N 1
#SBATCH -w voltron
#SBATCH -J cublas-faults
#SBATCH --exclusive
#SBATCH -t 48:00:00

export IGNORE_CC_MISMATCH=1
ITERS=1

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

MODULE_DIR=../../drivers/x86_64-555.42.02/faults-new/${srcv}/
MODULE_PATH=${MODULE_DIR}/nvidia-uvm.ko

module load cuda
cd ${MODULE_DIR}
make -j
sudo rmmod -f nvidia-uvm || true
cd -

sudo insmod ${MODULE_PATH} uvm_perf_prefetch_enable=0 hpcs_log_short=1 #uvm_perf_fault_coalesce=0 #uvm_perf_fault_batch_count=1

psizes=()
if [ $# -gt 0 ]; then
    for ((i=1; i<$#+1;i++));do
        psizes+=(${!i})
    done
else
    psizes=(4096 $(expr 4096 \* 4) $(expr 4096 \* 6) )
    #psizes=( $(expr 4096 \* 16) )
    #psizes=( 256 )
fi

cd ../cublas
make -j
for ((i=0;i<${#psizes[@]}; i++)); do
    psize=${psizes[$i]}
    logdir="log_short_${psize}"
    mkdir -p $logdir

    for ((j=0;j<$ITERS;j++)); do
        sudo dmesg -C
        sudo rmmod -f nvidia-uvm
        sudo insmod ${MODULE_PATH} uvm_perf_prefetch_enable=0 hpcs_log_short=1 #uvm_perf_fault_coalesce=0 #uvm_perf_fault_batch_count=1
        file=cublas_$j
        logfile=$logdir/$file
        pwd

        while [ ! -e /dev/hpcs_logger ]; do sleep 1; done
        make -C ../../tools/sysloggerv2
        ../../tools/sysloggerv2/log "$logfile" &
        pid=$!

        echo "pid: $pid"
        #time ./matrixMul2 -wA=${psize} -hA=${psize} -wB=${psize} -hB=${psize}
        time ./sgemm -n ${psize}

        len=`cat "$logfile" | wc -l`
        sleep 5
        until [ $(expr $(cat "$logfile" | wc -l)  - ${len}) -eq 0 ]; do
            len=`cat "$logfile" | wc -l`
            sleep 3
        done
        sleep 1
        kill $pid
    done

done
cd -

sudo rmmod nvidia-uvm
sudo modprobe nvidia-uvm #uvm_perf_prefetch_enable=0 uvm_perf_fault_coalesce=0 #uvm_perf_fault_batch_count=1
