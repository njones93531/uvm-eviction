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


void init_arrays(DATA_TYPE* hz_gpu)
{
	int i, j;

  #pragma omp parallel for
	for (i = 0; i < NX; i++)
	{
		for (j = 0; j < NY; j++)
		{
			hz_gpu[i*NY + j] = ((DATA_TYPE) (i-9)*(j+4) + 3) / NX;
		}
	}
}


void runFdtd(DATA_TYPE* _fict_, DATA_TYPE* ex, DATA_TYPE* ey, DATA_TYPE* hz)
{
	int t, i, j;
	
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

__global__ void compareResultsKernel(DATA_TYPE* hz1, DATA_TYPE* hz2, int* fail, int PSIZE) {
    // Shared memory for partial sum within a block
    __shared__ int block_fail_count[256];  // assuming up to 256 threads per block

    // Get thread indexes
    int i = blockIdx.x * blockDim.x + threadIdx.x;  // Row index
    int j = blockIdx.y * blockDim.y + threadIdx.y;  // Column index

    // Initialize shared memory
    int thread_fail = 0;
    if (i < NX && j < NY) {
        if (percentDiff(hz1[i * NY + j], hz2[i * NY + j]) > PERCENT_DIFF_ERROR_THRESHOLD) {
            thread_fail = 1;
        }
    }

    // Perform reduction within a block
    int thread_id = threadIdx.x + blockDim.x * threadIdx.y;  // unique thread ID within block
    block_fail_count[thread_id] = thread_fail;
    __syncthreads();

    // Reduction: sum of block_fail_count
    for (int stride = blockDim.x * blockDim.y / 2; stride > 0; stride /= 2) {
        if (thread_id < stride) {
            block_fail_count[thread_id] += block_fail_count[thread_id + stride];
        }
        __syncthreads();
    }

    // Accumulate the result from this block to global memory (only by one thread)
    if (thread_id == 0) {
        atomicAdd(fail, block_fail_count[0]);
    }
}

void compareResultsCUDA(DATA_TYPE* hz1, DATA_TYPE* hz2) {
    int fail = 0;
    int *d_fail;

    // Allocate memory for fail counter on the device
    cudaMalloc((void **)&d_fail, sizeof(int));
    cudaMemcpy(d_fail, &fail, sizeof(int), cudaMemcpyHostToDevice);

    // Define block and grid sizes
    dim3 block(256, 256);  // 16x16 threads per block
    dim3 grid((NX + block.x - 1) / block.x, (NY + block.y - 1) / block.y);  // Ensure coverage of all elements

    // Launch the kernel
    compareResultsKernel<<<grid, block>>>(hz1, hz2, d_fail, PSIZE);

    // Copy the fail result back to host
    cudaMemcpy(&fail, d_fail, sizeof(int), cudaMemcpyDeviceToHost);

    // Free device memory
    cudaFree(d_fail);

    // Print results
    printf("Non-Matching CPU-GPU Outputs Beyond Error Threshold of %4.2f Percent: %d\n", PERCENT_DIFF_ERROR_THRESHOLD, fail);
}

void compareResults(DATA_TYPE* hz1, DATA_TYPE* hz2)
{
	int i, j, fail;
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
	printf("Non-Matching CPU-GPU Outputs Beyond Error Threshold of %4.2f Percent: %d\n", PERCENT_DIFF_ERROR_THRESHOLD, fail);
}


void GPU_argv_init()
{
	cudaDeviceProp deviceProp;
	cudaGetDeviceProperties(&deviceProp, GPU_DEVICE);
	printf("setting device %d with name %s\n",GPU_DEVICE,deviceProp.name);
	cudaSetDevice( GPU_DEVICE );
}

__global__ void fdtd_stream_kernel(DATA_TYPE* hz, size_t PSIZE)
{
  int j = blockIdx.x * blockDim.x + threadIdx.x;
	int i = blockIdx.y * blockDim.y + threadIdx.y;

	if ((i < NX) && (j < NY))
	{
		hz[j * NY + i]++;
  }
 
}
__global__ void fdtd_step1_kernel(DATA_TYPE *hz, int t, size_t PSIZE)
{
	int j = blockIdx.x * blockDim.x + threadIdx.x;
	int i = blockIdx.y * blockDim.y + threadIdx.y;

	if ((i < NX) && (j < NY))
	{
		if (i > 0)
		{ 
			 if( 0.5f*(hz[i * NY + j] - hz[(i-1) * NY + j]) == -123123123)
       {
         hz[i * NY + j] = 1;
       }
		}
	}
}



__global__ void fdtd_step2_kernel(DATA_TYPE *hz, int t, size_t PSIZE)
{
	int j = blockIdx.x * blockDim.x + threadIdx.x;
	int i = blockIdx.y * blockDim.y + threadIdx.y;
	
	if ((i < NX) && (j < NY) && (j > 0))
	{
		if( 0.5f*(hz[i * NY + j] - hz[i * NY + (j-1)]) == -123123123)
    {
      hz[i * NY + j]++;
    }
	}
}


__global__ void fdtd_step3_kernel(DATA_TYPE *hz, int t, size_t PSIZE)
{
	int j = blockIdx.x * blockDim.x + threadIdx.x;
	int i = blockIdx.y * blockDim.y + threadIdx.y;
	
	if ((i < NX) && (j < NY))
	{	
		hz[i * NY + j] = hz[i * NY + j] - 0.7f;
	}
}


// Function to round up to the next power of 2
__host__ __device__
size_t nextPowerOf2(size_t n) {
    if (n == 0) return 1; // Special case for 0
    n--;
    n |= n >> 1;
    n |= n >> 2;
    n |= n >> 4;
    n |= n >> 8;
    n |= n >> 16;
    n |= n >> 32;  // This works for 64-bit integers
    return n + 1;
}


void fdtdCuda(DATA_TYPE* hz_gpu)
{
	double t_start, t_end, t_total;

	dim3 block(DIM_THREAD_BLOCK_X, DIM_THREAD_BLOCK_Y);
	dim3 grid( nextPowerOf2((size_t)ceil(((float)NY) / ((float)block.x))), nextPowerOf2((size_t)ceil(((float)NX) / ((float)block.y))));


  //Ensure that the pages of hz_gpu are arranged IN ORDER on the device
  //cudaMemPrefetchAsync(hz_gpu, PSIZE*PSIZE * sizeof(DATA_TYPE), 0 /* device ID */, nullptr);
  //cudaDeviceSynchronize();
  //fdtd_stream_kernel<<<grid,block>>>(hz_gpu, PSIZE);
  //cudaDeviceSynchronize();

	t_total = rtclock();
	for(int t = 0; t< tmax; t++)
	{
	  t_start = rtclock();
		fdtd_step1_kernel<<<grid,block>>>(hz_gpu, t, PSIZE);
		cudaDeviceSynchronize();
	  t_end = rtclock();
    fprintf(stdout, "GPU Runtime: %0.6lfs\n", t_end - t_start);
	  t_start = rtclock();
		fdtd_step2_kernel<<<grid,block>>>(hz_gpu, t, PSIZE);
		cudaDeviceSynchronize();
	  t_end = rtclock();
    fprintf(stdout, "GPU Runtime: %0.6lfs\n", t_end - t_start);
	  t_start = rtclock();
		fdtd_step3_kernel<<<grid,block>>>(hz_gpu, t, PSIZE);
		cudaDeviceSynchronize();
	  t_end = rtclock();
    fprintf(stdout, "GPU Runtime: %0.6lfs\n", t_end - t_start);
	  t_start = rtclock();
	}
  t_total = rtclock() - t_total;
  fprintf(stdout, "GPU Runtime: %0.6lfs\n", t_total);
	
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

	DATA_TYPE *hz_gpu;

//	_fict_ = (DATA_TYPE*)malloc(tmax*sizeof(DATA_TYPE));
//	ex = (DATA_TYPE*)malloc(NX*(NY+1)*sizeof(DATA_TYPE));
//	ey = (DATA_TYPE*)malloc((NX+1)*NY*sizeof(DATA_TYPE));
	//hz = (DATA_TYPE*)malloc(NX*NY*sizeof(DATA_TYPE));

	//cudaMallocManaged(&_fict_gpu, sizeof(DATA_TYPE) * tmax);
	//cudaMallocManaged(&ex_gpu, sizeof(DATA_TYPE) * NX * (NY + 1));
	//cudaMallocManaged(&ey_gpu, sizeof(DATA_TYPE) * (NX + 1) * NY);
	cudaMallocManaged(&hz_gpu, sizeof(DATA_TYPE) * NX * NY);
	//cudaMallocManaged(&hz, sizeof(DATA_TYPE) * NX * NY);

	init_arrays(hz_gpu);

	GPU_argv_init();
	
	AccessPolicy acp;
	//acp.setAllocationPolicy((void**)&_fict_gpu, sizeof(DATA_TYPE) * tmax, 0, argc, argv);
	//Set fict_gpu to d
  size_t size = sizeof(DATA_TYPE) * tmax;
 // cudaMalloc(&devptr, size);
 // CHECK_CUDA_ERROR();
 // cudaMemcpy(devptr, *a, size, cudaMemcpyHostToDevice);
 // CHECK_CUDA_ERROR();
 // CUDA_CHECK(cudaFree(*a));
 // CHECK_CUDA_ERROR();
 // *a = devptr;

 // acp.setAllocationPolicy((void**)&ex_gpu, sizeof(DATA_TYPE) * NX * (NY + 1), 0, argc, argv);
	//acp.setAllocationPolicy((void**)&ey_gpu, sizeof(DATA_TYPE) * (NX + 1) * NY, 1, argc, argv);
	acp.setAllocationPolicy((void**)&hz_gpu, sizeof(DATA_TYPE) * NX * NY, 0, argc, argv);

	
	fdtdCuda(hz_gpu);

	acp.freeMemPressure();

	cudaFree(hz_gpu);
	return 0;
}

