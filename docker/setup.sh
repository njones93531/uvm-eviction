#!/bin/bash -xe

./ngc_login.sh

if [ $? -eq 1 ]; then
  exit 1
fi

docker  build . -t cuda-test

docker run --privileged=true -it --gpus all --ipc=host --ulimit memlock=-1 --ulimit stack=67108864 --rm  -v `pwd`/../:uvm-eviction
cuda-test
