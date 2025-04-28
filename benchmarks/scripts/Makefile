all: warmup

warmup: warmup.cu Makefile
	nvcc -arch=sm_70 -Xcompiler "-Wall -Wextra" warmup.cu -o warmup
	
