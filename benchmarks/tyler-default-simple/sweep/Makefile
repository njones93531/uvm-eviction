CXXFLAGS=-O3 -g -std=c++11
CUDA_CXX=nvcc
EXTRA_FLAGS=-ccbin=/usr/share/gcc-7/bin/gcc -Xptxas "-v" -arch=sm_70
SOURCES="sweep.cu"

all: sweep

sweep: sweep.cu Makefile
	$(CUDA_CXX) $(DEFS) $(CXXFLAGS) $(SOURCES) $(EXTRA_FLAGS) -o $@
	$(CUDA_CXX) -ptx -src-in-ptx $(CXXFLAGS) $(SOURCES) $(EXTRA_FLAGS) -o $@.ptx

.PHONY: clean
clean:
	rm -f sweep
