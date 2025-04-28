#!/bin/bash

KERNEL_VERSION=$(uname -r)
VMLINUZ_PATH="/boot/vmlinuz-$KERNEL_VERSION+debug"
NVIDIA_UVM_KO_PATH="/lib/modules/$KERNEL_VERSION/kernel/drivers/video/nvidia-uvm.ko"
BASE_ADDR="0x3502080"

# Check if the required files exist
if [[ ! -f $VMLINUZ_PATH ]]; then
    echo "Error: $VMLINUZ_PATH not found!"
    exit 1
fi

if [[ ! -f $NVIDIA_UVM_KO_PATH ]]; then
    echo "Error: $NVIDIA_UVM_KO_PATH not found!"
    exit 1
fi

# Create a temporary GDB command file
GDB_CMDS=$(mktemp)
echo "add-symbol-file $NVIDIA_UVM_KO_PATH $BASE_ADDR" > $GDB_CMDS
echo "list uvm_pmm_gpu_alloc" >> $GDB_CMDS
echo "list *(uvm_pmm_gpu_alloc+0x98)" >> $GDB_CMDS
#echo "disassemble uvm_pmm_gpu_alloc" >> $GDB_CMDS
echo "quit" >> $GDB_CMDS

# Run GDB with the commands
gdb -q $VMLINUZ_PATH -x $GDB_CMDS

# Cleanup
rm -f $GDB_CMDS

