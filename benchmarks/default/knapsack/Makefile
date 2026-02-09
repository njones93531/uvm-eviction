all: main

main: main.cu problem_handle.h 
	nvcc -arch=sm_70 -lineinfo -Xcompiler "-Wall -Wextra -g -O0" -o main main.cu -lm

clean:
	rm main
