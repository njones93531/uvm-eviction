#!/bin/bash -xe


docker  build . -t cuda-test

docker run --privileged=true -it --gpus all --ipc=host --ulimit memlock=-1 --ulimit stack=67108864 --rm  -v `pwd`/NVIDIA-Linux-x86_64-570.124.06:/workspace/cuda-driver cuda-test
