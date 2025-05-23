/**
 * fdtd2d.cu: This file is part of the PolyBench/GPU 1.0 test suite.
 *
 *
 * Contact: Scott Grauer-Gray <sgrauerg@gmail.com>
 * Louis-Noel Pouchet <pouchet@cse.ohio-state.edu>
 * Web address: http://www.cse.ohio-state.edu/~pouchet/software/polybench/GPU
 */

#include <stdio.h>
#include <stdlib.h>
#include <math.h>
#include <assert.h>
#include <unistd.h>
#include <sys/time.h>
#include <cuda.h>
#include "accpol.h"
#include "../common/UVMBench/polybenchUtilFuncts.h"

//define the error threshold for the results "not matching"
#define PERCENT_DIFF_ERROR_THRESHOLD 10.05

#define GPU_DEVICE 0

/* Problem size */
#define tmax 10
#define NX PSIZE
#define NY PSIZE

/* Thread block dimensions */
#define DIM_THREAD_BLOCK_X 32
#define DIM_THREAD_BLOCK_Y 8

/* Can switch DATA_TYPE between float and double */
typedef float DATA_TYPE;
size_t PSIZE;


void init_arrays(DATA_TYPE* _fict_, DATA_TYPE* ex, DATA_TYPE* ey, DATA_TYPE* hz, DATA_TYPE* _fict_gpu, DATA_TYPE* ex_gpu, DATA_TYPE* ey_gpu, DATA_TYPE* hz_gpu)
{
	size_t i, j;

  	for (i = 0; i < tmax; i++)
	{
		_fict_[i] = (DATA_TYPE) i;
		_fict_gpu[i] = (DATA_TYPE) i;
	}
	
	for (i = 0; i < NX; i++)
	{
		for (j = 0; j < NY; j++)
		{
			ex[i*NY + j] = ((DATA_TYPE) i*(j+1) + 1) / NX;
			ey[i*NY + j] = ((DATA_TYPE) (i-1)*(j+2) + 2) / NX;
			hz[i*NY + j] = ((DATA_TYPE) (i-9)*(j+4) + 3) / NX;
			ex_gpu[i*NY + j] = ((DATA_TYPE) i*(j+1) + 1) / NX;
			ey_gpu[i*NY + j] = ((DATA_TYPE) (i-1)*(j+2) + 2) / NX;
			hz_gpu[i*NY + j] = ((DATA_TYPE) (i-9)*(j+4) + 3) / NX;
		}
	}
}


void runFdtd(DATA_TYPE* _fict_, DATA_TYPE* ex, DATA_TYPE* ey, DATA_TYPE* hz)
{
	size_t t, i, j;
	
	for (t=0; t < tmax; t++)  
	{
		for (j=0; j < NY; j++)
		{
			ey[0*NY + j] = _fict_[t];
		}
	
		for (i = 1; i < NX; i++)
		{
       		for (j = 0; j < NY; j++)
			{
       			ey[i*NY + j] = ey[i*NY + j] - 0.5*(hz[i*NY + j] - hz[(i-1)*NY + j]);
        		}
		}

		for (i = 0; i < NX; i++)
		{
       		for (j = 1; j < NY; j++)
			{
				ex[i*(NY+1) + j] = ex[i*(NY+1) + j] - 0.5*(hz[i*NY + j] - hz[i*NY + (j-1)]);
			}
		}

		for (i = 0; i < NX; i++)
		{
			for (j = 0; j < NY; j++)
			{
				hz[i*NY + j] = hz[i*NY + j] - 0.7*(ex[i*(NY+1) + (j+1)] - ex[i*(NY+1) + j] + ey[(i+1)*NY + j] - ey[i*NY + j]);
			}
		}
	}
}


void compareResults(DATA_TYPE* hz1, DATA_TYPE* hz2)
{
	size_t i, j, fail;
	fail = 0;
	
	for (i=0; i < NX; i++) 
	{
		for (j=0; j < NY; j++) 
		{
			if (percentDiff(hz1[i*NY + j], hz2[i*NY + j]) > PERCENT_DIFF_ERROR_THRESHOLD) 
			{
				fail++;
			}
		}
	}
	
	// Print results
	printf("Non-Matching CPU-GPU Outputs Beyond Error Threshold of %4.2f Percent: %zu\n", PERCENT_DIFF_ERROR_THRESHOLD, fail);
}


void GPU_argv_init()
{
	cudaDeviceProp deviceProp;
	cudaGetDeviceProperties(&deviceProp, GPU_DEVICE);
	printf("setting device %d with name %s\n",GPU_DEVICE,deviceProp.name);
	cudaSetDevice( GPU_DEVICE );
}



__global__ void fdtd_step1_kernel(DATA_TYPE* _fict_, DATA_TYPE *ex, DATA_TYPE *ey, DATA_TYPE *hz, size_t t, size_t PSIZE)
{
	size_t j = blockIdx.x * blockDim.x + threadIdx.x;
	size_t i = blockIdx.y * blockDim.y + threadIdx.y;

  double e = 0.000000001;
	if ((i < NX) && (j < NY))
	{
		if (i == 0) 
		{
      ey[i * NY + j] = _fict_[t] + (e * ex[i * NY + j]) + (e * hz[i * NY + j]);
		}
		else
		{ 
			ey[i * NY + j] = ey[i * NY + j] - 0.5f*(hz[i * NY + j] - hz[(i-1) * NY + j]) + (e * ex[i * NY + j]);
		}
	}
}



__global__ void fdtd_step2_kernel(DATA_TYPE *ex, DATA_TYPE *ey, DATA_TYPE *hz, size_t t, size_t PSIZE)
{
	size_t j = blockIdx.x * blockDim.x + threadIdx.x;
	size_t i = blockIdx.y * blockDim.y + threadIdx.y;
	
  double e = 0.000000001;
	if ((i < NX) && (j < NY) && (j > 0))
	{
		ex[i * (NY+1) + j] = ex[i * (NY+1) + j] - 0.5f*(hz[i * NY + j] - hz[i * NY + (j-1)]) + (e * ey[i * NY + j]);
	}
}


__global__ void fdtd_step3_kernel(DATA_TYPE *ex, DATA_TYPE *ey, DATA_TYPE *hz, size_t t, size_t PSIZE)
{
	size_t j = blockIdx.x * blockDim.x + threadIdx.x;
	size_t i = blockIdx.y * blockDim.y + threadIdx.y;
	
	if ((i < NX) && (j < NY))
	{	
		hz[i * NY + j] = hz[i * NY + j] - 0.7f*(ex[i * (NY+1) + (j+1)] - ex[i * (NY+1) + j] + ey[(i + 1) * NY + j] - ey[i * NY + j]);
	}
}


void fdtdCuda(DATA_TYPE* _fict_gpu, DATA_TYPE* ex_gpu, DATA_TYPE* ey_gpu, DATA_TYPE* hz_gpu)
{
	double t_start, t_end; //, t1;

	dim3 block(DIM_THREAD_BLOCK_X, DIM_THREAD_BLOCK_Y);
	dim3 grid( (size_t)ceil(((float)NY) / ((float)block.x)), (size_t)ceil(((float)NX) / ((float)block.y)));

	t_start = rtclock();

	for(int t = 0; t< tmax; t++)
	{
		//t1 = rtclock();
    fdtd_step1_kernel<<<grid,block>>>(_fict_gpu, ex_gpu, ey_gpu, hz_gpu, t, PSIZE);
		cudaDeviceSynchronize();
    //fprintf(stdout, "step1: %0.6lfs\n", rtclock() - t1);
    //t1 = rtclock();
		fdtd_step2_kernel<<<grid,block>>>(ex_gpu, ey_gpu, hz_gpu, t, PSIZE);
		cudaDeviceSynchronize();
    //fprintf(stdout, "step2: %0.6lfs\n", rtclock() - t1);
    //t1 = rtclock();
		fdtd_step3_kernel<<<grid,block>>>(ex_gpu, ey_gpu, hz_gpu, t, PSIZE);
		cudaDeviceSynchronize();
    //fprintf(stdout, "step3: %0.6lfs\n", rtclock() - t1);
	}
	
	t_end = rtclock();
    	fprintf(stdout, "GPU Runtime: %0.6lfs\n", t_end - t_start);
}


int main(int argc, char * argv[])
{

	//Set problem size with argv[1]
        if(argc>=2){
                if(strcmp(argv[1],"-h")==0){
                        printf("Usage: %s <psize (GB)> [1: cpu, 0: no cpu]\n",argv[0]);
                        exit(0);
                }
                double bytes = 1024. * 1024. * 1024. * atof(argv[1]);
                //printf("Bytes: %.2f\n", bytes);
                PSIZE = (size_t) (sqrt((bytes-tmax)/12));
        }
        else{
                PSIZE = 2048;
        }
        //printf("PSIZE: %zu\n", PSIZE)        printf("Problem size: %.2f GB\n", ((((double)(PSIZE * PSIZE * 3) + (TMAX)) * 4)/(1024. * 1024. * 1024.)));
        printf("Problem size: %.2f GB\n", ((((double)(PSIZE * PSIZE * 3) + (tmax)) * 4)/(1024. * 1024. * 1024.)));
        int cpu = 0;
	if(argc >= 3)
                cpu = atoi(argv[2]);	



	double t_start, t_end;

	DATA_TYPE* _fict_;
	DATA_TYPE* ex;
	DATA_TYPE* ey;
	DATA_TYPE* hz;

	DATA_TYPE *_fict_gpu;
	DATA_TYPE *ex_gpu;
	DATA_TYPE *ey_gpu;
	DATA_TYPE *hz_gpu;

	_fict_ = (DATA_TYPE*)malloc(tmax*sizeof(DATA_TYPE));
	ex = (DATA_TYPE*)malloc(NX*(NY+1)*sizeof(DATA_TYPE));
	ey = (DATA_TYPE*)malloc((NX+1)*NY*sizeof(DATA_TYPE));
	hz = (DATA_TYPE*)malloc(NX*NY*sizeof(DATA_TYPE));

	cudaMallocManaged(&_fict_gpu, sizeof(DATA_TYPE) * tmax);
	cudaMallocManaged(&ex_gpu, sizeof(DATA_TYPE) * NX * (NY + 1));
	cudaMallocManaged(&ey_gpu, sizeof(DATA_TYPE) * (NX + 1) * NY);
	cudaMallocManaged(&hz_gpu, sizeof(DATA_TYPE) * NX * NY);

	init_arrays(_fict_, ex, ey, hz, _fict_gpu, ex_gpu, ey_gpu, hz_gpu);

	GPU_argv_init();
	
	AccessPolicy acp;
	//acp.setAllocationPolicy((void**)&_fict_gpu, sizeof(DATA_TYPE) * tmax, 0, argc, argv);
	acp.setAllocationPolicy((void**)&ex_gpu, sizeof(DATA_TYPE) * NX * (NY + 1), 0, argc, argv);
	acp.setAllocationPolicy((void**)&ey_gpu, sizeof(DATA_TYPE) * (NX + 1) * NY, 1, argc, argv);
	acp.setAllocationPolicy((void**)&hz_gpu, sizeof(DATA_TYPE) * NX * NY, 2, argc, argv);

	
	fdtdCuda(_fict_gpu, ex_gpu, ey_gpu, hz_gpu);

	if(cpu){
		t_start = rtclock();
		runFdtd(_fict_, ex, ey, hz);
		t_end = rtclock();
		
		fprintf(stdout, "CPU Runtime: %0.6lfs\n", t_end - t_start);
		
		compareResults(hz, hz_gpu);
	}

	free(_fict_);
	free(ex);
	free(ey);
	free(hz);
			
	cudaFree(_fict_gpu);
	acp.freeMemPressure();
	cudaFree(ex_gpu);
	cudaFree(ey_gpu);
	cudaFree(hz_gpu);
	return 0;
}

