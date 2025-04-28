#!/bin/bash -x

module load cuda gcc/12.2.0
make -j
cd /home/tnallen/dev/uvm-eviction/drivers/x86_64-535.104.05/syscall/kernel
make -j
sudo make -j modules_install
cd -
sudo rmmod nvidia-uvm
sudo modprobe nvidia-uvm
sudo dmesg -C
sleep 5
dmesg
echo "running test"
./test
sleep 2
dmesg
