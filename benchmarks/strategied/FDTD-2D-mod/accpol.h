#ifndef ACCESS_POLICY_H
#define ACCESS_POLICY_H

// System includes
#include <stdio.h>
#include <assert.h>
#include <iostream>
#include <stdint.h>

// CUDA runtime
#include <cuda_runtime.h>

// Helper functions and utilities to work with CUDA
//#include <helper_functions.h>
//#include <helper_cuda.h>
#include <unistd.h>
#include <fcntl.h>

#ifndef CUDA_CHECK
#define CUDA_CHECK(status) \
  if (status != cudaSuccess) \
  { \
    printf("%s:%d CudaError: %s\n", __FILE__, __LINE__, cudaGetErrorString(status)); \
    assert(0); \
  }
#endif
#ifndef CHECK_CUDA_ERROR
#define CHECK_CUDA_ERROR()                                                    \
{                                                                             \
    cudaError_t err = cudaGetLastError();                                     \
    if (err != cudaSuccess)                                                   \
    {                                                                         \
        printf("error=%d name=%s at "                                         \
               "ln: %d\n  ",err,cudaGetErrorString(err),__LINE__);            \
        exit(1);\
    }                                                                         \
}
#endif


class AccessPolicy{
private:
	void * mem_pressure = 0x0;
	int aoi = -1; 
	std::size_t aoi_size = 0;
	double perc = -1.0;
public:

	void setMemPressure(std::size_t size){
		if(size <= 1000) 
			return;
		cudaMalloc(&mem_pressure, size);
		CHECK_CUDA_ERROR();
		if(mem_pressure == 0x0){
			std::cout << "mem pressure failed\n";
			exit(-1);
		}
	      	double pressureGB = ((double)aoi_size * perc) / (1024. * 1024. * 1024.);
		fprintf(stdout, "Allocated %.2f GB (%.2f%% of allocation %d) of memory pressure\n",
		      pressureGB, perc, aoi);
	}

	void freeMemPressure(){
		cudaDeviceSynchronize();
		if(mem_pressure == 0x0) return;
		cudaFree(mem_pressure);
	}

	void usage(char * argv0){
		std::cout << "Usage: " << argv0 << "<benchmark args> <flag1> <arg1> <flag2...>\n"
			<< "Available flags: \n"
			<< "-p: policy\t| string \t| one character per allocation\n"
			<< "-aoi: alloc of interest\t| int\t| which alloc to apply mem pressure\n"
			<< "-r: pressure\t| float\t | fraction of aoi's size to block in memory as mem pressure\n"; 
	}
	//Checks for two things: 1) The mem policy of a given allocation 2) Checks and
	//implements memory pressure if applicable
	char parseCLA(std::size_t size, int alloc_num, int argc, char** argv){
	  char flag = '-';
	  bool args = false;
	  for (int i = 1; i < argc; i++){
	    if (argv[i][0] == '-' &&  argv[i][1] == 'p'){ // next arg sets policy
	      args = true;
	      flag = argv[++i][alloc_num];
	    }
	    if (strcmp(argv[i], "-aoi") == 0 && atoi(argv[i+1]) == alloc_num){
	      args = true;
	      aoi_size = size;
	      aoi = atoi(argv[i+1]);
	    }
	    if (argv[i][0] == '-' &&  argv[i][1] == 'r'){ // next arg sets pressure
	      args = true;
	      perc = atof(argv[++i]);
	      if(aoi == alloc_num)
		  setMemPressure((std::size_t)((double)aoi_size * perc));
	    }
	  }
	  if (!args) usage(argv[0]);
	  return flag;
	}


	//Set the allocation policy based on cl flags
	void setAllocationPolicy(void **a, std::size_t size, int alloc_num, int argc, char** argv) {
	  char flag = parseCLA(size, alloc_num, argc, argv);
    //std::cout << "Setting allocation " << alloc_num << " of size " << float(size) / (1024. * 1024. * 1024.) << "GB to '" << flag << "'\n";
	  switch(flag){
	    case 'm': //Policy is migrate; do nothing
	      break;
	    case 'd': //Pin to device; use cudaMemCopy
	      void * devptr;
	      cudaMalloc(&devptr, size);
	      CHECK_CUDA_ERROR();
	      cudaMemcpy(devptr, *a, size, cudaMemcpyHostToDevice);
	      CHECK_CUDA_ERROR();
	      CUDA_CHECK(cudaFree(*a));
	      CHECK_CUDA_ERROR();
	      *a = devptr;
	      break;
	    case 'a': //Pin to device, async variant
	      //First, set preferred location of everything to device
	      cudaMemAdvise(*a, size, cudaMemAdviseSetPreferredLocation, 0);
	      //Finally, move the pin device mem to device
	      cudaMemPrefetchAsync(*a, size, 0, 0);
	      break;
	    case 'h': //Pin to host; use cudaMemAdvise
	      CUDA_CHECK(cudaMemAdvise(*a, size, cudaMemAdviseSetPreferredLocation, cudaCpuDeviceId));
	      CHECK_CUDA_ERROR();
	      break;
	    case 's': //Split allocation between pin host and pin device
	      //Note that perc will be set already 
	      //First, set preferred location of everything to device
	      cudaMemAdvise(*a, size, cudaMemAdviseSetPreferredLocation, 0);
	      printf("Perc is %.2f\n", perc);
	      if(perc <= 0.0) break;
	      //Next, set preferred location of first (perc) bytes to host
	      cudaMemAdvise(*a, (perc*size), cudaMemAdviseSetPreferredLocation, cudaCpuDeviceId);
	      //Finally, move the pin device mem to device
	      //Assumes type is <=8 bytes
	      cudaMemPrefetchAsync(static_cast<uint64_t*>(*a)+((size_t)(size*perc)/sizeof(uint64_t)), (1.0-perc) * size, 0, 0);
	      break;
	    case 'x': //Split allocation between pin host and pin device with no prefetch 
	      //Note that perc will be set already 
	      //First, set preferred location of everything to device
	      cudaMemAdvise(*a, size, cudaMemAdviseSetPreferredLocation, 0);
	      //Next, set preferred location of first (perc) bytes to host
	      cudaMemAdvise(*a, (perc*size), cudaMemAdviseSetPreferredLocation, cudaCpuDeviceId);
	      break;
      default:
        std::cout << "Policy flag '" << flag << "' used on allocation " << alloc_num << " is not supported.\n";
        exit(1);
	  }
	  return;
	}

	
};

#endif
