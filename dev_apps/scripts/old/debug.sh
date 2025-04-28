#!/bin/bash -ex
#SBATCH -N 1
#SBATCH -w voltron
#SBATCH -J matmul-trace
#SBATCH --exclusive
#SBATCH -t 12:00:00

ITERS=1

module load cuda

export IGNORE_CC_MISMATCH=1
# make sure driver builds, exists
cd ../../drivers/x86_64-535.104.05/preempt-conservative/kernel
make -j
sudo make modules_install
cd -

# ensure driver loads our newly built one
sudo modprobe nvidia-uvm 
sudo rmmod -f nvidia-uvm
sudo modprobe nvidia-uvm 

# benchmark-specific problem size
psizes=()
if [ $# -gt 0 ]; then
    for ((i=1; i<$#+1;i++));do
        psizes+=(${!i})
    done
else
    psizes=($(expr 1024 \* 28) $(expr 1024 \* 29) $(expr 1024 \* 30) $(expr 1024 \* 31) $(expr 1024 \* 32)  $(expr 1024 \* 33)  $(expr 1024 \* 34)  $(expr 1024 \* 35)  $(expr 1024 \* 36))
fi

# go to benchmark directory and build
cd ../cublas
module load gcc # was gcc/12.2.0 at time of writing
make

# iterate over problem sizes
for ((i=0;i<${#psizes[@]}; i++)); do
    psize=${psizes[$i]}
    logdir="log_${psize}"
    mkdir -p $logdir
    # multiple iterations?
    for ((j=0;j<$ITERS;j++)); do
        file=matmul_$j
        # scratch dir is faster write; maybe overkill - only works on voltron unless server configured with scratch
        logfile=/scratch1/$file
        pwd

        # engage syslogging tool; uses large raw buffer needs 8 seconds to malloc
        ../../tools/syslogger/log "$logfile" &
        pid=$!
        sleep 8

        # problem size
        echo "pid: $pid"
        time ./matrixMul2 -wA=${psize} -hA=${psize} -wB=${psize} -hB=${psize}

        # make sure logfile isn't truncated; overkill
        len=`cat "$logfile" | wc -l`
        sleep 5
        until [ $(expr $(cat "$logfile" | wc -l)  - ${len}) -eq 0 ]; do
            len=`cat "$logfile" | wc -l`
            sleep 3
        done
        sleep 1
        
        # kill log process and move file to correct directory
        kill $pid
        mv $logfile $logdir/
        ../../tools/sys2csv/log2csv.sh $logdir/$file
        dmesg > $logdir/dmesg_$i
    done

done
cd -


# make sure driver builds, exists
cd ../../drivers/x86_64-535.104.05/vanilla/kernel
make -j
sudo make modules_install
cd -

# at least set module to default config; should probably reinstall vanilla
sudo rmmod nvidia-uvm
sudo modprobe nvidia-uvm #uvm_perf_prefetch_enable=0 uvm_perf_fault_coalesce=0 #uvm_perf_fault_batch_count=1

