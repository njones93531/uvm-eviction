CC = g++
NVCC = nvcc
CFLAGS = -Wall -Wextra -O3 -g
LDFLAGS = -lopenblas
NVFLAGS = -lcublas -Xcompiler "${CFLAGS} -fopenmp" -gencode=arch=compute_70,code=sm_70 -gencode=arch=compute_90,code=sm_90


all: sgemm

sgemm: sgemm.cu
	$(NVCC) -o sgemm sgemm.cu $(LDFLAGS) $(NVFLAGS)

clean:
	rm -f sgemm
