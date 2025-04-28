#!/bin/bash -ex
#SBATCH -N 1
#SBATCH -w voltron
#SBATCH -J oversub-ac-tracking-test
#SBATCH --exclusive
#SBATCH -t 48:00:00

ITERS=1

module load cuda

# enable using newer version of cuad compiler than supported
export IGNORE_CC_MISMATCH=1
# make sure driver builds, exists
cd ../../drivers/x86_64-460.27.04/ac-tracking/kernel
make
sudo make modules_install
sudo modprobe nvidia-uvm
cd -

# build our app
make

#################################### Run with block_streaming access pattern, zero-copy
# clear system log
sudo dmesg -C
# remove nvidia-uvm module
sudo rmmod -f nvidia-uvm
# reload nvidia-uvm module with our own settings
sudo modprobe nvidia-uvm #uvm_perf_access_counter_mimc_migration_enable=1 uvm_perf_access_counter_momc_migration_enable=1

# collect dmesg message log - TODO this is oversimplification; if ac-tracking is too large, we need to use our own tool in /tools/syslogger
sudo dmesg -w > zero_copy.txt &
pid=$! # collect PID from dmesg running in background
time ./uvm_oversubs -p 2.0 -a block_streaming -m zero_copy # - Test oversubscription with 2x GPU memory size working set, using zero-copy (data placed in CPU memory and directly accessed), and streaming access pattern (see corresponding developer blog for detail).
kill $pid # kill demsg logging now that we're done

#################################### Run with block_streaming access pattern, fault migration
sudo dmesg -C
sudo rmmod -f nvidia-uvm
sudo modprobe nvidia-uvm #uvm_perf_access_counter_mimc_migration_enable=1 uvm_perf_access_counter_momc_migration_enable=1

sudo dmesg -w > fault.txt &
pid=$!
time ./uvm_oversubs -p 2.0 -a block_streaming -m fault # - Test oversubscription with 2x GPU memory allocated using Unified Memory (`cudaMallocManaged`) and block strided kernel read data with page-fault induced migration.
kill $pid
################################### Run with block_streaming access pattern, zero-copy except with access-counter guided migration
sudo dmesg -C
sudo rmmod -f nvidia-uvm
sudo modprobe nvidia-uvm uvm_perf_access_counter_mimc_migration_enable=1 uvm_perf_access_counter_momc_migration_enable=1

sudo dmesg -w > zero_copy_mimo.txt &
pid=$!
time ./uvm_oversubs -p 2.0 -a block_streaming -m zero_copy # - Test oversubscription with 2x GPU memory size working set, using zero-copy (data placed in CPU memory and directly accessed), and streaming access pattern (see corresponding developer blog for detail).
kill $pid

#################################### Run with block_streaming access pattern, page faults - what does access counter guided migration do here?
sudo dmesg -C
sudo rmmod -f nvidia-uvm
sudo modprobe nvidia-uvm uvm_perf_access_counter_mimc_migration_enable=1 uvm_perf_access_counter_momc_migration_enable=1

sudo dmesg -w > fault_mimo.txt &
pid=$!
time ./uvm_oversubs -p 2.0 -a block_streaming -m fault # - Test oversubscription with half GPU memory allocated using Unified Memory (`cudaMallocManaged`) and block strided kernel read data with page-fault induced migration.
kill $pid
###################################

# at least set module to default config; should probably reinstall vanilla
sudo rmmod nvidia-uvm
sudo modprobe nvidia-uvm #uvm_perf_prefetch_enable=0 uvm_perf_fault_coalesce=0 #uvm_perf_fault_batch_count=1


#time ./uvm_oversubs -p 1.5 -a stripe_gpu_cpu -m random_warp # - Test oversubscription with 1.5x GPU memory working set, with memory pages striped between GPU and CPU. Random warp kernel accesses a different 128 byte region of allocation in each loop iteration.

#time ./uvm_oversubs -p 1.5 -a stripe_gpu_cpu -m random_warp

