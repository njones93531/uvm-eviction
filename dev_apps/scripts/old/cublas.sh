#!/bin/bash -ex
#SBATCH -N 1
#SBATCH -w voltron
#SBATCH -J matmul-trace
#SBATCH --exclusive
#SBATCH -t 12:00:00

# Function to check if the module is still in use
module_is_in_use() {
    if lsmod | grep "^nvidia-uvm" > /dev/null; then
        # Check the second field (number of instances using the module)
        local use_count=$(lsmod | grep "^nvidia-uvm" | awk '{print $2}')
        [ "$use_count" -gt "0" ]
        sudo rmmod -f nvidia-uvm
        if [ "$?" -eq 0 ]; then
            return 0
        fi
    fi
    return 1
}


# Check for command line argument
if [ -z "$1" ]; then
    echo "Error: Please provide a command line argument."
    exit 1
fi

ITERS=1

module load cuda

export IGNORE_CC_MISMATCH=1

# Use the command line argument in place of 'preempt'
DIR="../../drivers/x86_64-535.104.05/$1/kernel"

# Check if directory exists
if [ ! -d "$DIR" ]; then
    echo "Error: Directory $DIR does not exist."
    exit 1
fi

# Change to the directory
cd "$DIR"

make clean -j
make -j
sudo make modules_install
cd -

# Set a timeout of 5 minutes
end_time=$((SECONDS+300))

# Loop until the module is not in use or timeout is reached
while module_is_in_use; do
    if [ $SECONDS -gt $end_time ]; then
        echo "Timeout reached, exiting script."
        exit 1
    fi
    sleep 5
done

sudo rmmod -f nvidia-uvm

sudo modprobe nvidia-uvm


# benchmark-specific problem size
psizes=()
if [ $# -gt 1 ]; then
    for ((i=2; i<$#+1;i++));do
        psizes+=(${!i})
    done
else
    psizes=($(expr 1024 \* 32))
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

        time ./matrixMul2 -wA=${psize} -hA=${psize} -wB=${psize} -hB=${psize}
    done
done
cd -
