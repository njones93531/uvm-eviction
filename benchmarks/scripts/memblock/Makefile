all: memblock

memblock: memblock.cu Makefile
	nvcc -arch=sm_70 -Xcompiler "-Wall -Wextra" memblock.cu -o memblock

clean:
	rm -rf memblock
	
