CXXFLAGS=-O3
CUDA_CXX=nvcc

cuda-stream: main.cpp CUDAStream.cu
	$(CUDA_CXX)  -std=c++11 $(CXXFLAGS) -DCUDA $^ $(EXTRA_FLAGS) -Xptxas "-v" -o $@   -gencode arch=compute_70,code=sm_70 -gencode arch=compute_80,code=sm_80 -gencode arch=compute_86,code=sm_86 -gencode arch=compute_90,code=sm_90 -Xcompiler "-Wall -Wextra -O3 -fopenmp"

.PHONY: clean
clean:
	rm -f cuda-stream

