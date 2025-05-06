FROM nvcr.io/nvidia/cuda-dl-base:25.03-cuda12.8-runtime-ubuntu24.04

RUN apt update && apt upgrade -y && apt install -y build-essential linux-headers-$(uname -r) dkms libncurses-dev gawk flex bison libssl-dev libelf-dev libudev-dev libpci-dev libiberty-dev autoconf llvm dwarves
