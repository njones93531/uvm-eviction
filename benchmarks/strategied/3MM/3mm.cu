/**
 * 3mm.cu: This file is part of the PolyBench/GPU 1.0 test suite.
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

#define GPU_DEVICE 0

//define the error threshold for the results "not matching"
#define PERCENT_DIFF_ERROR_THRESHOLD 0.05

/* Problem size. */
# define NI PSIZE
# define NJ PSIZE
# define NK PSIZE
# define NL PSIZE
# define NM PSIZE

/* Thread block dimensions */
#define DIM_THREAD_BLOCK_X 32
#define DIM_THREAD_BLOCK_Y 8

/* Can switch DATA_TYPE between float and double */
typedef float DATA_TYPE;
size_t PSIZE;


void init_array(DATA_TYPE* A, DATA_TYPE* B, DATA_TYPE* C, DATA_TYPE* D, DATA_TYPE* A_gpu, DATA_TYPE* B_gpu, DATA_TYPE* C_gpu, DATA_TYPE* D_gpu )
{
	int i, j;

	for (i = 0; i < NI; i++)
	{
		for (j = 0; j < NK; j++)
		{
			A[i*NK + j] = ((DATA_TYPE) i*j) / NI;
			A_gpu[i*NK + j] = ((DATA_TYPE) i*j) / NI;
		}
	}
  
	for (i = 0; i < NK; i++)
	{
		for (j = 0; j < NJ; j++)
		{
			B[i*NJ + j] = ((DATA_TYPE) i*(j+1)) / NJ;
			B_gpu[i*NJ + j] = ((DATA_TYPE) i*(j+1)) / NJ;
		}
	}
  
	for (i = 0; i < NJ; i++)
	{
		for (j = 0; j < NM; j++)
		{
			C[i*NM + j] = ((DATA_TYPE) i*(j+3)) / NL;
			C_gpu[i*NM + j] = ((DATA_TYPE) i*(j+3)) / NL;
		}
	}
  
	for (i = 0; i < NM; i++)
	{
		for (j = 0; j < NL; j++)
		{
			D[i*NL + j] = ((DATA_TYPE) i*(j+2)) / NK;
			D_gpu[i*NL + j] = ((DATA_TYPE) i*(j+2)) / NK;
		}
	}
}


void compareResults(DATA_TYPE *G, DATA_TYPE *G_outputFromGpu)
{
	int i,j,fail;
	fail = 0;

	for (i=0; i < NI; i++)
	{
		for (j=0; j < NL; j++)
		{
			if (percentDiff(G[i*NL + j], G_outputFromGpu[i*NL + j]) > PERCENT_DIFF_ERROR_THRESHOLD)
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

	
__global__ void mm3_kernel1(DATA_TYPE *A, DATA_TYPE *B, DATA_TYPE *E, size_t PSIZE)
{
	int j = blockIdx.x * blockDim.x + threadIdx.x;
	int i = blockIdx.y * blockDim.y + threadIdx.y;

	if ((i < NI) && (j < NJ))
	{
		int k;
		for(k=0; k < NK; k++)
		{
			E[i * NJ + j] += A[i * NK + k] * B[k * NJ + j];
		}
	}
}

	
__global__ void mm3_kernel2(DATA_TYPE *C, DATA_TYPE *D, DATA_TYPE *F, size_t PSIZE)
{
	int j = blockIdx.x * blockDim.x + threadIdx.x;
	int i = blockIdx.y * blockDim.y + threadIdx.y;

	if ((i < NJ) && (j < NL))
	{
		int k;
		for(k=0; k < NM; k++)
		{
			F[i * NL + j] += C[i * NM + k] * D[k * NL +j];
		}
	}
}

	
__global__ void mm3_kernel3(DATA_TYPE *E, DATA_TYPE *F, DATA_TYPE *G, size_t PSIZE)
{
	int j = blockIdx.x * blockDim.x + threadIdx.x;
	int i = blockIdx.y * blockDim.y + threadIdx.y;

	if ((i < NI) && (j < NL))
	{
		int k;
		for(k=0; k < NJ; k++)
		{
			G[i * NL + j] += E[i * NJ + k] * F[k * NL + j];
		}
	}
}


void mm3_cpu(DATA_TYPE *A, DATA_TYPE *B, DATA_TYPE *C, DATA_TYPE *D, DATA_TYPE *E, DATA_TYPE *F, DATA_TYPE *G)
{
	int i,j,k;
	
	/* E := A*B */
	for (i = 0; i < NI; i++)
	{
		for (j = 0; j < NJ; j++)
		{
			E[i*NJ + j] = 0;
			for (k = 0; k < NK; ++k)
			{
				E[i*NJ + j] += A[i*NK + k] * B[k*NJ + j];
			}
		}
	}
		
	/* F := C*D */
	for (i = 0; i < NJ; i++)
	{
		for (j = 0; j < NL; j++)
		{
			F[i*NL + j] = 0;
			for (k = 0; k < NM; ++k)
			{
				F[i*NL + j] += C[i*NM + k] * D[k*NL + j];
			}
		}
	}

  	/* G := E*F */
	for (i = 0; i < NI; i++)
	{
		for (j = 0; j < NL; j++)
		{
			G[i*NL + j] = 0;
			for (k = 0; k < NJ; ++k)
			{
				G[i*NL + j] += E[i*NJ + k] * F[k*NL + j];
			}
		}
	}
}


void mm3Cuda(DATA_TYPE* A_gpu, DATA_TYPE* B_gpu, DATA_TYPE* C_gpu, DATA_TYPE* D_gpu, DATA_TYPE* E_gpu, DATA_TYPE* F_gpu, 
		DATA_TYPE* G_gpu)
{
	double t_start, t_end;

	
	dim3 block(DIM_THREAD_BLOCK_X, DIM_THREAD_BLOCK_Y);
	dim3 grid1((size_t)(ceil( ((float)NJ) / ((float)DIM_THREAD_BLOCK_X) )),(size_t)(ceil((float)NI/ ((float)DIM_THREAD_BLOCK_Y) )));
	dim3 grid2((size_t)(ceil( ((float)NL) / ((float)DIM_THREAD_BLOCK_X) )),(size_t)(ceil((float)NJ/ ((float)DIM_THREAD_BLOCK_Y) )));
	dim3 grid3((size_t)(ceil( ((float)NL) / ((float)DIM_THREAD_BLOCK_X) )),(size_t)(ceil((float)NI/ ((float)DIM_THREAD_BLOCK_Y) )));

	t_start = rtclock();
	mm3_kernel1<<<grid1,block>>>(A_gpu, B_gpu, E_gpu, PSIZE);
	cudaDeviceSynchronize();
	mm3_kernel2<<<grid2,block>>>(C_gpu, D_gpu, F_gpu, PSIZE);
	cudaDeviceSynchronize();
	mm3_kernel3<<<grid3,block>>>(E_gpu, F_gpu, G_gpu, PSIZE);
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
                PSIZE = (size_t) (sqrt(bytes/28));
        }
        else{
                PSIZE = 512;
        }
        //printf("PSIZE: %zu\n", PSIZE);
        printf("Problem size: %.2f GB\n", ((((double)(PSIZE * PSIZE *7)) * 4)/(1024. * 1024. * 1024.)));
        int cpu = 0;
        if(argc >= 3)
                cpu = atoi(argv[2]);	



	double t_start, t_end;

	DATA_TYPE* A;
	DATA_TYPE* B;
	DATA_TYPE* C;
	DATA_TYPE* D;
	DATA_TYPE* E;
	DATA_TYPE* F;
	DATA_TYPE* G;

	DATA_TYPE *A_gpu;
	DATA_TYPE *B_gpu;
	DATA_TYPE *C_gpu;
	DATA_TYPE *D_gpu;
	DATA_TYPE *E_gpu;
	DATA_TYPE *F_gpu;
	DATA_TYPE *G_gpu;


	A = (DATA_TYPE*)malloc(NI*NK*sizeof(DATA_TYPE));
	B = (DATA_TYPE*)malloc(NK*NJ*sizeof(DATA_TYPE));
	C = (DATA_TYPE*)malloc(NJ*NM*sizeof(DATA_TYPE));
	D = (DATA_TYPE*)malloc(NM*NL*sizeof(DATA_TYPE));
	E = (DATA_TYPE*)malloc(NI*NJ*sizeof(DATA_TYPE));
	F = (DATA_TYPE*)malloc(NJ*NL*sizeof(DATA_TYPE));
	G = (DATA_TYPE*)malloc(NI*NL*sizeof(DATA_TYPE));


	cudaMallocManaged(&A_gpu, sizeof(DATA_TYPE) * NI * NK);
	cudaMallocManaged(&B_gpu, sizeof(DATA_TYPE) * NK * NJ);
	cudaMallocManaged(&C_gpu, sizeof(DATA_TYPE) * NJ * NM);
	cudaMallocManaged(&D_gpu, sizeof(DATA_TYPE) * NM * NL);
	cudaMallocManaged(&E_gpu, sizeof(DATA_TYPE) * NI * NJ);
	cudaMallocManaged(&F_gpu, sizeof(DATA_TYPE) * NJ * NL);
	cudaMallocManaged(&G_gpu, sizeof(DATA_TYPE) * NI * NL);

	init_array(A, B, C, D, A_gpu, B_gpu, C_gpu, D_gpu);

	GPU_argv_init();

	AccessPolicy acp;
	acp.setAllocationPolicy((void**)&A_gpu, sizeof(DATA_TYPE) * NI * NK, 0, argc, argv);
	acp.setAllocationPolicy((void**)&B_gpu, sizeof(DATA_TYPE) * NK * NJ, 1, argc, argv);
	acp.setAllocationPolicy((void**)&C_gpu, sizeof(DATA_TYPE) * NJ * NM, 2, argc, argv);
	acp.setAllocationPolicy((void**)&D_gpu, sizeof(DATA_TYPE) * NM * NL, 3, argc, argv);
	acp.setAllocationPolicy((void**)&E_gpu, sizeof(DATA_TYPE) * NI * NJ, 4, argc, argv);
	acp.setAllocationPolicy((void**)&F_gpu, sizeof(DATA_TYPE) * NJ * NL, 5, argc, argv);
	acp.setAllocationPolicy((void**)&G_gpu, sizeof(DATA_TYPE) * NI * NL, 6, argc, argv);

	
	mm3Cuda(A_gpu, B_gpu, C_gpu, D_gpu, E_gpu, F_gpu, G_gpu);


	if(cpu){
		t_start = rtclock();

		mm3_cpu(A, B, C, D, E, F, G);
		
		t_end = rtclock();

		fprintf(stdout, "CPU Runtime: %0.6lfs\n", t_end - t_start);

		compareResults(G, G_gpu);
	}
	free(A);
	free(B);
	free(C);
	free(D);
	free(E);
	free(F);
	free(G);

	cudaFree(A_gpu);
	cudaFree(B_gpu);
	cudaFree(C_gpu);
	cudaFree(D_gpu);
	cudaFree(E_gpu);
	cudaFree(F_gpu);
	cudaFree(G_gpu);

	acp.freeMemPressure();

	return 0;
}

