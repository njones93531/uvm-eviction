FROM nvcr.io/nvidia/cuda:12.5.0-devel-ubuntu22.04

ENV DEBIAN_FRONTEND=noninteractive

RUN apt update && apt upgrade -y && apt install -y build-essential dkms libncurses-dev gawk flex bison libssl-dev libelf-dev libudev-dev libpci-dev libiberty-dev autoconf llvm dwarves python3-pip locales software-properties-common numactl vim texlive texlive-latex-extra texlive-fonts-recommended texlive-latex-recommended texlive-fonts-extra texlive-luatex texlive-science dvipng cm-super && \
    locale-gen en_US.UTF-8 && \
    update-locale LANG=en_US.UTF-8 LC_ALL=en_US.UTF-8

ENV LANG=en_US.UTF-8 \
    LANGUAGE=en_US:en \
    LC_ALL=en_US.UTF-8

ENTRYPOINT ["/uvm-eviction/docker/entrypoint.sh"]
