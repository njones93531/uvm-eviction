all:
	nvcc -O3 ${CUFILES} ${DEF} -o ${EXECUTABLE} -gencode arch=compute_61,code=sm_61  -gencode arch=compute_70,code=sm_70 -gencode arch=compute_80,code=sm_80 -gencode arch=compute_86,code=sm_86 -gencode arch=compute_90,code=sm_90 -Xcompiler "-Wall -Wextra -O3 -fopenmp"

clean:
	rm -f *~ *.exe
