#!/bin/bash -xe

#Ensure kernel headers exist
./check-kernel-headers.sh
if [ $? -eq 1 ]; then
  exit 1
fi

#Login to NGC for container download
./ngc_login.sh
if [ $? -eq 1 ]; then
  exit 1
fi

docker  build . -t cuda-container

docker run --init --hostname=cuda-host --privileged=true -it --gpus all --ipc=host \
  --ulimit memlock=-1 --ulimit stack=67108864 --rm  \
  -v `pwd`/../:/uvm-eviction -w /uvm-eviction \
  -v /lib/modules:/lib/modules \
  -v /usr/src/:/usr/src:ro \
  cuda-container bash
