#!/bin/bash -xe

# find MPI compiler
CC=`which mpicc`
#CC=`which mpiicc`

# find NVCC compiler
NVCC=`which nvcc`

# set gpu architectures to compile for
#CUDA_ARCH+="-gencode arch=compute_60,code=sm_60 "
CUDA_ARCH+="-gencode arch=compute_70,code=sm_70 "
#CUDA_ARCH+="-gencode arch=compute_80,code=sm_80 "

# main tile size
OPTS+="-DBLOCKCOPY_TILE_I=32 "
OPTS+="-DBLOCKCOPY_TILE_J=4 "
OPTS+="-DBLOCKCOPY_TILE_K=8 "

# special tile size for boundary conditions
OPTS+="-DBOUNDARY_TILE_I=64 "
OPTS+="-DBOUNDARY_TILE_J=16 "
OPTS+="-DBOUNDARY_TILE_K=16 "

# host level threshold: number of grid elements
OPTS+="-DHOST_LEVEL_SIZE_THRESHOLD=10000 "

# max number of solves after warmup
OPTS+="-DMAX_SOLVES=10 "

# unified memory allocation options
OPTS+="-DCUDA_UM_ALLOC "
#OPTS+="-DCUDA_UM_ZERO_COPY "

# MPI buffers allocation policy
OPTS+="-DMPI_ALLOC_ZERO_COPY "
#OPTS+="-DMPI_ALLOC_PINNED "

# stencil optimizations
OPTS+="-DUSE_REG "
OPTS+="-DUSE_TEX "
#OPTS+="-DUSE_SHM "

# GSRB smoother options
#OPTS+="-DGSRB_FP "
#OPTS+="-DGSRB_STRIDE2 "
#OPTS+="-DGSRB_BRANCH "
#OPTS+="-DGSRB_OOP "

# tools
#OPTS+="-DUSE_PROFILE "
#OPTS+="-DUSE_NVTX "
#OPTS+="-DUSE_ERROR "

# override MVAPICH flags for C++
OPTS+="-DMPICH_IGNORE_CXX_SEEK "
OPTS+="-DMPICH_SKIP_MPICXX "

rm -rf build

# GSRB smoother (default)
./configure --no-fv-mpi --CC=$CC --NVCC=$NVCC --CFLAGS="-O2 -g -fopenmp $OPTS" --NVCCFLAGS="-O2 -lineinfo -lnvToolsExt $OPTS" --CUDAARCH="$CUDA_ARCH" --no-fe

# Chebyshev smoother
#./configure --CC=$CC --NVCC=$NVCC --CFLAGS="-O2 -fopenmp $OPTS" --NVCCFLAGS="-O2 -lineinfo -lnvToolsExt $OPTS" --CUDAARCH="$CUDA_ARCH" --fv-smoother="cheby" --no-fe

#make clean -C build
make V=1 -j3 -C build
