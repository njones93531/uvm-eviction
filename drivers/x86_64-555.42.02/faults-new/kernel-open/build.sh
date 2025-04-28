#!/bin/bash

make -j
sudo rmmod nvidia-uvm
sudo insmod ./nvidia-uvm.ko
