/**
 * gemm.cu: This file is part of the PolyBench/GPU 1.0 test suite.
 *
 *
 * Contact: Scott Grauer-Gray <sgrauerg@gmail.com>
 * Louis-Noel Pouchet <pouchet@cse.ohio-state.edu>
 * Web address: http://www.cse.ohio-state.edu/~pouchet/software/polybench/GPU
 */

#include <unistd.h>
#include <stdio.h>
#include <time.h>
#include <sys/time.h>
#include <stdlib.h>
#include <stdarg.h>
#include <string.h>
#include <cuda.h>
#include <math.h>
#include "accpol.h"
#include "../common/UVMBench/polybenchUtilFuncts.h"

#define GPU_DEVICE 0

//define the error threshold for the results "not matching"
#define PERCENT_DIFF_ERROR_THRESHOLD 0.05

/* Problem size */
#define NI PSIZE
#define NJ PSIZE
#define NK PSIZE

/* Thread block dimensions */
#define DIM_THREAD_BLOCK_X 32
#define DIM_THREAD_BLOCK_Y 8

/* Declared constant values for ALPHA and BETA (same as values in PolyBench 2.0) */
#define ALPHA 32412.0f
#define BETA 2123.0f

/* Can switch DATA_TYPE between float and double */
typedef float DATA_TYPE;
size_t PSIZE;


void gemm(DATA_TYPE *A, DATA_TYPE *B, DATA_TYPE *C)
{
    size_t i,j,k;

    for (i = 0; i < NI; i++)
    {
        for (j = 0; j < NJ; j++)
        {
            C[i*NJ + j] *= BETA;

            for (k = 0; k < NK; ++k)
            {
                C[i*NJ + j] += ALPHA * A[i*NK + k] * B[k*NJ + j];
            }
        }
    }
}


void init(DATA_TYPE *A, DATA_TYPE *B, DATA_TYPE *C)
{
    size_t i = 0;
    size_t j = 0;

    #pragma omp parallel private(i,j) shared(A,B,C)
    {
        #pragma omp for simd collapse(2) nowait 
        for (i = 0; i < NI; i++)
        {
            for (j = 0; j < NK; j++)
            {
                A[i*NK + j] = ((DATA_TYPE) i*j) / NI;
            }
        }

        #pragma omp for simd collapse(2) nowait
        for (i = 0; i < NK; i++)
        {
            for (j = 0; j < NJ; j++)
            {
                B[i*NJ + j] = ((DATA_TYPE) i*j + 1) / NJ;
            }
        }

        #pragma omp for simd collapse(2) nowait
        for (i = 0; i < NI; i++)
        {
            for (j = 0; j < NJ; j++)
            {
                C[i*NJ + j] = ((DATA_TYPE) i*j + 2) / NJ;
            }
        }
    }
}



void compareResults(DATA_TYPE* C, DATA_TYPE* C_outputFromGpu)
{
    size_t i, j, fail;
    fail = 0;

    // Compare C1 and C2
    for (i=0; i < NI; i++) 
    {
        for (j=0; j < NJ; j++) 
        {
            if (percentDiff(C[i*NJ + j], C_outputFromGpu[i*NJ + j]) > PERCENT_DIFF_ERROR_THRESHOLD) 
            {
                fail++;
            }
        }
    }

    // Print results
    printf("Non-Matching CPU-GPU Outputs Beyond Error Threshold of %4.2f Percent: %lu\n", PERCENT_DIFF_ERROR_THRESHOLD, fail);
}


void GPU_argv_init()
{
    cudaDeviceProp deviceProp;
    cudaGetDeviceProperties(&deviceProp, GPU_DEVICE);
    printf("setting device %d with name %s\n",GPU_DEVICE,deviceProp.name);
    cudaSetDevice( GPU_DEVICE );
}


__global__ void gemm_kernel(DATA_TYPE *a, DATA_TYPE *b, DATA_TYPE *c, size_t PSIZE)
{
    int j = blockIdx.x * blockDim.x + threadIdx.x;
    int i = blockIdx.y * blockDim.y + threadIdx.y;

    if ((i < NI) && (j < NJ))
    {	
        c[i * NJ + j] *= BETA;
        int k;
        for(k=0; k < NK; k++)
        {
            c[i * NJ + j] += ALPHA * a[i * NK + k] * b[k * NJ +j];
        }
    }
}


void gemmCuda(DATA_TYPE* A_gpu, DATA_TYPE* B_gpu, DATA_TYPE* C_gpu)
{
    double t_start, t_end;

    dim3 block(DIM_THREAD_BLOCK_X, DIM_THREAD_BLOCK_Y);
    dim3 grid((size_t)(ceil( ((float)NI)/ ((float)block.x) )),(size_t)(ceil( ((float)NJ)/ ((float)block.y) )));

    t_start = rtclock();

    gemm_kernel<<< grid, block >>>(A_gpu, B_gpu, C_gpu, PSIZE);
    cudaDeviceSynchronize();
    CHECK_CUDA_ERROR();

    t_end = rtclock();
    fprintf(stdout, "GPU Runtime: %0.6lfs\n", t_end - t_start);   

}


int main(int argc, char *argv[])
{
    //Set problem size with argv[1]
    if(argc>=2){
        if(strcmp(argv[1],"-h")==0){
            printf("Usage: %s <psize (GB)> [1: cpu, 0: no cpu]\n",argv[0]);
            exit(0);
        }
        double bytes = 1024. * 1024. * 1024. * atof(argv[1]);
        //printf("Bytes: %.2f\n", bytes);
        PSIZE = (size_t) (sqrt(bytes/12));
    }
    else{
        PSIZE = 512;
    }
    //printf("PSIZE: %zu\n", PSIZE);
    printf("Problem size: %.2f GB\n", ((((double)(PSIZE * PSIZE * 3)) * 4)/(1024. * 1024. * 1024.)));
    int cpu = 0;
    if(argc >= 3)
        cpu = atoi(argv[2]);	



    double t_start, t_end;

    DATA_TYPE* A = NULL;
    DATA_TYPE* B = NULL;
    DATA_TYPE* C = NULL;
    DATA_TYPE *A_gpu = NULL;
    DATA_TYPE *B_gpu = NULL;
    DATA_TYPE *C_gpu = NULL;

    cudaMallocManaged(&A_gpu, sizeof(DATA_TYPE) * NI * NK);
    CHECK_CUDA_ERROR();
    cudaMallocManaged(&B_gpu, sizeof(DATA_TYPE) * NK * NJ);
    CHECK_CUDA_ERROR();
    cudaMallocManaged(&C_gpu, sizeof(DATA_TYPE) * NI * NJ);
    CHECK_CUDA_ERROR();

    init(A_gpu, B_gpu, C_gpu);

    GPU_argv_init();

    AccessPolicy acp; 
    acp.setAllocationPolicy((void**)&A_gpu, sizeof(DATA_TYPE) * NI * NK, 0, argc, argv);
    acp.setAllocationPolicy((void**)&B_gpu, sizeof(DATA_TYPE) * NK * NJ, 1, argc, argv);
    acp.setAllocationPolicy((void**)&C_gpu, sizeof(DATA_TYPE) * NI * NJ, 2,argc, argv);


    gemmCuda(A_gpu, B_gpu, C_gpu);

    if(cpu){
        A = (DATA_TYPE*)malloc(NI*NK*sizeof(DATA_TYPE)); 
        B = (DATA_TYPE*)malloc(NK*NJ*sizeof(DATA_TYPE));   
        C = (DATA_TYPE*)malloc(NI*NJ*sizeof(DATA_TYPE)); 
        init(A, B, C);
        t_start = rtclock();	
        gemm(A, B, C);
        t_end = rtclock();
        fprintf(stdout, "CPU Runtime: %0.6lfs\n", t_end - t_start);

        compareResults(C, C_gpu);
    }
    free(A);
    free(B);  
    free(C);  
    cudaFree(A_gpu);
    cudaFree(B_gpu);
    cudaFree(C_gpu);

    acp.freeMemPressure();

    return 0;
}

