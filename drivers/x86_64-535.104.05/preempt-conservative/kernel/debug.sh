#!/bin/bash

# Get the current kernel version using uname -r
KERNEL_VERSION=$(uname -r)

# Try to find the debug vmlinuz version first, and then the regular one if the debug isn't available
if [ -f "/boot/vmlinuz-${KERNEL_VERSION}+debug" ]; then
    VMLINUX_PATH="/boot/vmlinuz-${KERNEL_VERSION}+debug"
else
    VMLINUX_PATH="/boot/vmlinuz-${KERNEL_VERSION}"
fi

# Get the base address where the module is loaded
BASE_ADDR=$(grep -Pho 'nvidia_uvm[ \t]+[0-9a-f]+' /proc/modules | awk '{print "0x"$2}')

# The path to nvidia-uvm.ko from modinfo
NVIDIA_UVM_PATH=$(modinfo nvidia-uvm | grep filename | awk '{print $2}')

# Start GDB, load the vmlinux symbols and the nvidia_uvm symbols, and then list the code at the address
echo "Using vmlinux from $VMLINUX_PATH and nvidia-uvm.ko from $NVIDIA_UVM_PATH with base address $BASE_ADDR"

gdb -q $VMLINUX_PATH <<EOF
add-symbol-file $NVIDIA_UVM_PATH $BASE_ADDR
list *(uvm_pmm_gpu_alloc+0x98)
quit
EOF

