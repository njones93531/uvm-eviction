/**
 * 2mm.cu: This file is part of the PolyBench/GPU 1.0 test suite.
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

#include "../common/UVMBench/polybenchUtilFuncts.h"

//define the error threshold for the results "not matching"
#define PERCENT_DIFF_ERROR_THRESHOLD 0.05

#define GPU_DEVICE 0

/* Problem size. */
# define NI PSIZE
# define NJ PSIZE
# define NK PSIZE
# define NL PSIZE

/* Thread block dimensions */
#define DIM_THREAD_BLOCK_X 32
#define DIM_THREAD_BLOCK_Y 8

/* Can switch DATA_TYPE between float and double */
typedef float DATA_TYPE;
size_t PSIZE;


void init_array(DATA_TYPE* A, DATA_TYPE* B, DATA_TYPE* C, DATA_TYPE* D, DATA_TYPE* A_gpu, DATA_TYPE* B_gpu, DATA_TYPE* C_gpu, DATA_TYPE* D_gpu)
{
	int i, j;

	for (i = 0; i < NI; i++)
	{
		for (j = 0; j < NK; j++)
		{
			A[i*NI + j] = ((DATA_TYPE) i*j) / NI;
			A_gpu[i*NI + j] = ((DATA_TYPE) i*j) / NI;
		}
	}

	for (i = 0; i < NK; i++)
	{
		for (j = 0; j < NJ; j++)
		{
			B[i*NK + j] = ((DATA_TYPE) i*(j+1)) / NJ;
			B_gpu[i*NK + j] = ((DATA_TYPE) i*(j+1)) / NJ;
		}
	}

	for (i = 0; i < NL; i++)
	{
		for (j = 0; j < NJ; j++)
		{
			C[i*NL + j] = ((DATA_TYPE) i*(j+3)) / NL;
			C_gpu[i*NL + j] = ((DATA_TYPE) i*(j+3)) / NL;
		}
	}

	for (i = 0; i < NI; i++)
	{
		for (j = 0; j < NL; j++)
		{
			D[i*NL + j] = ((DATA_TYPE) i*(j+2)) / NK;	
			D_gpu[i*NL + j] = ((DATA_TYPE) i*(j+2)) / NK;
		}
	}
}


void compareResults(DATA_TYPE *E, DATA_TYPE *E_outputFromGpu)
{
	int i,j,fail;
	fail = 0;

	for (i=0; i < NL; i++)
	{
		for (j=0; j < NI; j++)
		{
			if (percentDiff(E[i*NI + j], E_outputFromGpu[i*NI + j]) > PERCENT_DIFF_ERROR_THRESHOLD)
			{
				fail++;
			}
		}
	}
	
	// print results
	printf("Non-Matching CPU-GPU Outputs Beyond Error Threshold of %4.2f Percent: %d\n", PERCENT_DIFF_ERROR_THRESHOLD, fail);
}


void GPU_argv_init()
{
	cudaDeviceProp deviceProp;
	cudaGetDeviceProperties(&deviceProp, GPU_DEVICE);
	printf("setting device %d with name %s\n",GPU_DEVICE,deviceProp.name);
	cudaSetDevice( GPU_DEVICE );
}


__global__ void mm2_kernel1(DATA_TYPE *A, DATA_TYPE *B, DATA_TYPE *C, size_t PSIZE)
{
	int j = blockIdx.x * blockDim.x + threadIdx.x;
	int i = blockIdx.y * blockDim.y + threadIdx.y;

	if ((i < NI) && (j < NJ))
	{ 
		int k;
		for (k = 0; k < NK; k++)
		{
			C[i * NJ + j] += A[i * NK + k] * B[k * NJ + j];
		}
	}
}


__global__ void mm2_kernel2(DATA_TYPE *C, DATA_TYPE *D, DATA_TYPE *E, size_t PSIZE)
{
	int j = blockIdx.x * blockDim.x + threadIdx.x;
	int i = blockIdx.y * blockDim.y + threadIdx.y;

	if ((i < NI) && (j < NL))
	{ 
		int k;
		for (k = 0; k < NJ; k++)
		{
			E[i * NL + j] += C[i * NJ + k] * D[k * NL + j];
		}
	}
}


void mm2_cpu(DATA_TYPE* A, DATA_TYPE* B, DATA_TYPE* C, DATA_TYPE* D, DATA_TYPE* E)
{
	int i, j, k;
	
  	for (i = 0; i < NI; i++)
	{
		for (j = 0; j < NJ; j++)
		{
			C[i*NJ + j] = 0.0;
			for (k = 0; k < NK; ++k)
			{
				C[i*NJ + j] += A[i*NK + k] * B[k*NJ + j];
			}
		}
	}
	
	for (i = 0; i < NI; i++)
	{
		for (j = 0; j < NL; j++)
		{
			E[i*NL + j] = 0.0;
			for (k = 0; k < NJ; ++k)
			{
				E[i*NL + j] += C[i*NJ + k] * D[k*NL + j];
			}
		}
	}
}


void mm2Cuda(DATA_TYPE* A_gpu, DATA_TYPE* B_gpu, DATA_TYPE* C_gpu, DATA_TYPE* D_gpu, DATA_TYPE* E_gpu)
{
	double t_start, t_end;
	dim3 block(DIM_THREAD_BLOCK_X, DIM_THREAD_BLOCK_Y);
	dim3 grid1((size_t)ceil( ((float)NJ) / ((float)block.x) ), (size_t)ceil( ((float)NI) / ((float)block.y)) );
	dim3 grid2((size_t)ceil( ((float)NL) / ((float)block.x) ), (size_t)ceil( ((float)NI) / ((float)block.y)) );
	t_start = rtclock();
	mm2_kernel1<<<grid1,block>>>(A_gpu, B_gpu, C_gpu, PSIZE);
	cudaDeviceSynchronize();
	mm2_kernel2<<<grid2,block>>>(C_gpu, D_gpu, E_gpu, PSIZE);
	cudaDeviceSynchronize();
	t_end = rtclock();
	fprintf(stdout, "GPU Runtime: %0.6lfs\n", t_end - t_start);
}


int main(int argc, char** argv)
{
	//Set problem size with argv[1]
        if(argc>=2){
                if(strcmp(argv[1],"-h")==0){
                        printf("Usage: %s <psize (GB)> [1: cpu, 0: no cpu]\n",argv[0]);
                        exit(0);
                }
                double bytes = 1024. * 1024. * 1024. * atof(argv[1]);
                //printf("Bytes: %.2f\n", bytes);
                PSIZE = (size_t) (sqrt(bytes/20));
        }
        else{
                PSIZE = 2048;
        }
        //printf("PSIZE: %zu\n", PSIZE);
        printf("Problem size: %.2f GB\n", (((double)(PSIZE * PSIZE * 5) * 4)/(1024. * 1024. * 1024.)));
        int cpu = 0;
        if(argc >= 3)
                cpu = atoi(argv[2]);	


	
	double t_start, t_end;
	
	DATA_TYPE* C;
	DATA_TYPE* A;
	DATA_TYPE* B;
	DATA_TYPE* D;
	DATA_TYPE* E;

	DATA_TYPE *A_gpu;
	DATA_TYPE *B_gpu;
	DATA_TYPE *C_gpu;
	DATA_TYPE *D_gpu;
	DATA_TYPE *E_gpu;

	C = (DATA_TYPE*)malloc(NI*NJ*sizeof(DATA_TYPE));
	A = (DATA_TYPE*)malloc(NI*NK*sizeof(DATA_TYPE));
	B = (DATA_TYPE*)malloc(NK*NJ*sizeof(DATA_TYPE));
	D = (DATA_TYPE*)malloc(NJ*NL*sizeof(DATA_TYPE));
	E = (DATA_TYPE*)malloc(NI*NL*sizeof(DATA_TYPE));


	cudaMallocManaged(&A_gpu, sizeof(DATA_TYPE) * NI * NK);
	cudaMallocManaged(&B_gpu, sizeof(DATA_TYPE) * NK * NJ);
	cudaMallocManaged(&C_gpu, sizeof(DATA_TYPE) * NI * NJ);
	cudaMallocManaged(&D_gpu, sizeof(DATA_TYPE) * NJ * NL);
	cudaMallocManaged(&E_gpu, sizeof(DATA_TYPE) * NI * NL);


	printf("Start address of A_gpu:\t%p\n", &(A_gpu[0]));
	printf("Start address of B_gpu:\t%p\n", &(B_gpu[0]));
	printf("Start address of C_gpu:\t%p\n", &(C_gpu[0]));
	printf("Start address of D_gpu:\t%p\n", &(D_gpu[0]));
	printf("Start address of E_gpu:\t%p\n", &(E_gpu[0]));
  	
	init_array(A, B, C, D, A_gpu, B_gpu, C_gpu, D_gpu);
	GPU_argv_init();

	mm2Cuda(A_gpu, B_gpu, C_gpu, D_gpu, E_gpu);


	if(cpu){
		t_start = rtclock();
		mm2_cpu(A, B, C, D, E);
		t_end = rtclock();
		fprintf(stdout, "CPU Runtime: %0.6lfs\n", t_end - t_start);

		compareResults(E, E_gpu);
	}
	free(C);
	free(A);
	free(B);
	free(D);
	free(E);
	cudaFree(A_gpu);
	cudaFree(B_gpu);
	cudaFree(C_gpu);
	cudaFree(D_gpu);
	cudaFree(E_gpu);
  	return 0;
}

