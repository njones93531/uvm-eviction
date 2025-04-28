#!/bin/bash -e
#SBATCH -N 1
#SBATCH -w voltron
#SBATCH -J oversub-perf-test
#SBATCH --exclusive
#SBATCH -t 10:00:00 
# Ideally, if each case took 2 minutes, everything will be done in 6.6 hours

ITERS=1

module load cuda

# enable using newer version of cuad compiler than supported
export IGNORE_CC_MISMATCH=1
# make sure driver builds, exists
cd ../../drivers/x86_64-460.27.04/vanilla/kernel
make -j
sudo make modules_install
sudo modprobe nvidia-uvm
cd -

# build our app
make

#################################### Run with block_streaming access pattern, zero-copy

TIMEOUT="FALSE"
ACCESS_PATTERN="random_warp"

for UVM_MODE in page_fault zero_copy stripe_gpu_cpu
do  
  # for ACCESS_PATTERN in streaming block_streaming
    # do
      TIMEOUT="FALSE"
      for O_FACTOR in $(seq -f "%1.1f" 0.8 0.1 3.0)  
      do
	  if [[ "${TIMEOUT}" = "FALSE" ]]
	  then 
  	      output=$(./uvm_oversubs -p $O_FACTOR -a $ACCESS_PATTERN -m $UVM_MODE)
	      echo "$output"
	      bandwidth=$(echo "$output" | awk '{print $(NF-1)}')
	      if (( $(echo "$bandwidth < 1.0" |bc -l) ));
	      then
		  TIMEOUT="TRUE"
	      fi
	  fi 
      done
  # done
done
##./uvm_oversubs -p 2.0 -a block_streaming -m zero_copy # - Test oversubscription with 2x GPU memory size working set, using zero-copy (data placed in CPU memory and directly accessed), and streaming access pattern (see corresponding developer blog for detail).
#################################### Run with block_streaming access pattern, fault migration
##./uvm_oversubs -p 2.0 -a block_streaming -m fault # - Test oversubscription with 2x GPU memory allocated using Unified Memory (`cudaMallocManaged`) and block strided kernel read data with page-fault induced migration.
################################### Run with block_streaming access pattern, zero-copy except with access-counter guided migration
##./uvm_oversubs -p 2.0 -a block_streaming -m zero_copy # - Test oversubscription with 2x GPU memory size working set, using zero-copy (data placed in CPU memory and directly accessed), and streaming access pattern (see corresponding developer blog for detail).
#################################### Run with block_streaming access pattern, page faults - what does access counter guided migration do here?
#./uvm_oversubs -p 2.0 -a block_streaming -m fault # - Test oversubscription with half GPU memory allocated using Unified Memory (`cudaMallocManaged`) and block strided kernel read data with page-fault induced migration.
###################################

# at least set module to default config; should probably reinstall vanilla
sudo rmmod nvidia-uvm
sudo modprobe nvidia-uvm #uvm_perf_prefetch_enable=0 uvm_perf_fault_coalesce=0 #uvm_perf_fault_batch_count=1


#time ./uvm_oversubs -p 1.5 -a stripe_gpu_cpu -m random_warp # - Test oversubscription with 1.5x GPU memory working set, with memory pages striped between GPU and CPU. Random warp kernel accesses a different 128 byte region of allocation in each loop iteration.

#time ./uvm_oversubs -p 1.5 -a stripe_gpu_cpu -m random_warp

