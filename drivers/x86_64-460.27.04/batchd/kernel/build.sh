#!/bin/bash -xe

make modules -j
sudo make modules_install
sudo rmmod nvidia_uvm
