MODE=gpu
DEBUG=no
OPT=yes

ifeq ($(DEBUG),yes)
COMMON_FLAGS=-g 
endif

ifeq ($(OPT),yes)
COMMON_FLAGS+= -O3
endif

ifeq ($(OPT),no)
COMMON_FLAGS+= -O0
endif


ifeq ($(MODE),mic)
CC=icc
CXX=icpc
CPP=icpc
CCC=icpc
LD=icpc

CCFLAGS=$(COMMON_FLAGS) -openmp 
CXXFLAGS=$(COMMON_FLAGS) -openmp
LDFLAGS=$(COMMON_FLAGS) -openmp
endif

ifeq ($(MODE),gpu)
CC=gcc
CXX=g++
LD=nvcc
CCFLAGS=$(COMMON_FLAGS) -std=c++11 -fopenmp  -fomit-frame-pointer -malign-double -fstrict-aliasing -ffast-math # -Wall -Wextra
CXXFLAGS=$(COMMON_FLAGS) -std=c++11  -fopenmp -fomit-frame-pointer -malign-double -fstrict-aliasing -ffast-math # -Wall -Wextra
CUDA_FLAGS=$(COMMON_FLAGS) -std=c++11 # -Xcompiler -Wall -Xcompiler -Wextra
endif


ifeq ($(MODE),cpu)
CC=gcc
CXX=g++
LD=g++
CCFLAGS=$(COMMON_FLAGS) -fopenmp  -fomit-frame-pointer -malign-double -fstrict-aliasing -ffast-math -std=c++11
CXXFLAGS=$(COMMON_FLAGS) -fopenmp -fomit-frame-pointer -malign-double -fstrict-aliasing -ffast-math -std=c++11
LDFLAGS=$(COMMON_FLAGS) -fopenmp
endif




OBJS=bc_gpu_edge.o  bc_gpu_vertex.o  bc_gpu_vertex_multi.o  bc_gpu_virtual.o   preproc.o  ulib.o bc_gpu_edge_multi.o bc_gpu_virtual_multi.o cc_gpu_virtual.o cc_gpu_vertex_multi_spmv.o main.o fake_mic_for_gpu.o format.o cc_cpu.o bc_cpu.o graphIO.o #cc_virtual_multi_loop.o

OBJSMIC=ulib.o main.o fake_gpu_for_mic.o cc_mic.o format.o  #cc_cpu.o is not included becasue it is part of mic_spmv.o

OBJSCPU=ulib.o main.o fake_gpu_for_mic.o fake_mic_for_gpu.o format.o cc_cpu.o bc_cpu.o graphIO.o


all: $(OBJS) gpu_bc

%.o : %.cu
	nvcc $(CUDA_FLAGS) -c $< -arch sm_70

gpu_bc: $(OBJS)
	nvcc $(CUDA_FLAGS) -Xcompiler -fopenmp $(OBJS) -o gpu_bc

clean:
	-rm $(OBJS) $(OBJSMIC) $(OBJSCPU) gpu_bc mic_bc

mic_bc: $(OBJSMIC)
	icpc $(LDFLAGS) $(OBJSMIC) -o mic_bc

cpu_bc: $(OBJSCPU)
	$(LD) $(LDFLAGS) $(OBJSCPU) -o cpu_bc

depend:
	makedepend *.cpp *.hpp *.h -Y.

# DO NOT DELETE

cc-cpu.o: config.h util.h timestamp.hpp
fake_gpu_for_mic.o: debug.hpp
gpu-bc.o: ulib.h timestamp.hpp format.h config.h bc_cpu.hpp cc-cpu.hpp
gpu-bc.o: bc-gpu.hpp cc-gpu.hpp graphIO.hpp
graphIO.o: ulib.h timestamp.hpp graphIO.hpp
mic_spmv.o: cc-cpu.cpp config.h util.h timestamp.hpp
