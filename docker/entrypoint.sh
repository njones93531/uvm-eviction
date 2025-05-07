#!/bin/bash

echo "Setting up the environment"
root=/uvm-eviction

#Python dependencies
cd $root
pip install -r requirements.txt 

VRAM_MB=$(nvidia-smi -i 0 --query-gpu=memory.total --format=csv,noheader,nounits)
VRAM_GB=$(( ($VRAM_MB + 512) / 1024 ))

#Set config file
cd $root/benchmarks/scriptsv3/
python3 set_config.py --root $root --vram-size $VRAM_GB

cd $root

add-apt-repository -y ppa:ubuntu-toolchain-r/test
apt update 
apt install -y gcc-12 g++-12
apt install -y libstdc++6
export CC=gcc-12
export CXX=g++-12

echo "Environment complete" 



exec "$@"
