/*
 * Copyright 1993-2017 NVIDIA Corporation.  All rights reserved.
 *
 * Please refer to the NVIDIA end user license agreement (EULA) associated
 * with this source code for terms and conditions that govern your use of
 * this software. Any use, reproduction, disclosure, or distribution of
 * this software and related documentation outside the terms of the EULA
 * is strictly prohibited.
 *
 */

/* Example showing the use of CUFFT for fast 1D-convolution using FFT. */

// includes, system
#include <math.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

// includes, project
#include <cuda_runtime.h>
#include <cufft.h>
#include <cufftXt.h>
#include <helper_cuda.h>
#include <helper_functions.h>

// Complex data type
typedef float2 Complex;
static __device__ __host__ inline Complex ComplexAdd(Complex, Complex);
static __device__ __host__ inline Complex ComplexScale(Complex, float);
static __device__ __host__ inline Complex ComplexMul(Complex, Complex);
static __global__ void ComplexPointwiseMulAndScale(Complex *, const Complex *, int, float);

// Filtering functions
void Convolve(const Complex *, int, const Complex *, int, Complex *);

// Padding functions
int PadData(const Complex *, Complex **, int, const Complex *, Complex **, int);

////////////////////////////////////////////////////////////////////////////////
// declaration, forward
void runTest(int argc, char **argv);

// The filter size is assumed to be a number smaller than the signal size
#ifndef SIGNAL_SIZE
#define SIGNAL_SIZE (int(80000000)*10)
#endif

#define FILTER_KERNEL_SIZE 1024
#define ITERS 150

////////////////////////////////////////////////////////////////////////////////
// Program main
////////////////////////////////////////////////////////////////////////////////
int main(int argc, char **argv) {
    srand(25890);
    runTest(argc, argv);
}

////////////////////////////////////////////////////////////////////////////////
//! Run a simple test for CUDA
////////////////////////////////////////////////////////////////////////////////
void runTest(int argc, char **argv) {
    printf("[simpleCUFFT] is starting...\n");

    findCudaDevice(argc, (const char **)argv);

    // Allocate host memory for the signal
    Complex *h_signal = reinterpret_cast<Complex *>(malloc(sizeof(Complex) * SIGNAL_SIZE));

    // Initialize the memory for the signal
    for (unsigned int i = 0; i < SIGNAL_SIZE; ++i) {
        h_signal[i].x = rand() / static_cast<float>(RAND_MAX);
        h_signal[i].y = 0;
    }

    // Allocate host memory for the filter
    Complex *h_filter_kernel = reinterpret_cast<Complex *>(malloc(sizeof(Complex) * FILTER_KERNEL_SIZE));

    // Initialize the memory for the filter
    for (unsigned int i = 0; i < FILTER_KERNEL_SIZE; ++i) {
        h_filter_kernel[i].x = rand() / static_cast<float>(RAND_MAX);
        h_filter_kernel[i].y = 0;
    }

    // Pad signal and filter kernel
    Complex *h_padded_signal;
    Complex *h_padded_filter_kernel;
    int new_size = PadData(h_signal, &h_padded_signal, SIGNAL_SIZE, h_filter_kernel, &h_padded_filter_kernel, FILTER_KERNEL_SIZE);
    int mem_size = sizeof(Complex) * new_size;

    // Allocate device memory for signal using cudaMallocManaged
    Complex *d_signal;
    checkCudaErrors(cudaMallocManaged(reinterpret_cast<void **>(&d_signal), mem_size));
    memcpy(d_signal, h_padded_signal, mem_size);

    // Allocate device memory for filter kernel using cudaMallocManaged
    Complex *d_filter_kernel;
    checkCudaErrors(cudaMallocManaged(reinterpret_cast<void **>(&d_filter_kernel), mem_size));
    memcpy(d_filter_kernel, h_padded_filter_kernel, mem_size);

    // CUFFT plan advanced API
    cufftHandle plan_adv;
    size_t workSize;
    long long int new_size_long = new_size;
    
    checkCudaErrors(cufftCreate(&plan_adv));
    checkCudaErrors(cufftXtMakePlanMany(plan_adv, 1, &new_size_long, NULL, 1, 1, CUDA_C_32F, NULL, 1, 1, CUDA_C_32F, 1, 0, CUDA_C_32F));

    // Allocate managed workspace for cuFFT
    void *d_workspace;
    checkCudaErrors(cudaMallocManaged(&d_workspace, workSize));
    checkCudaErrors(cufftSetWorkArea(plan_adv, d_workspace));


    // Timing events
    cudaEvent_t start, stop;
    cudaEventCreate(&start);
    cudaEventCreate(&stop);

    // Record the start event
    cudaEventRecord(start, NULL);

    for (int i = 0; i < ITERS; i++) {
        // Transform signal and filter kernel
        checkCudaErrors(cufftExecC2C(plan_adv, reinterpret_cast<cufftComplex *>(d_signal), reinterpret_cast<cufftComplex *>(d_signal), CUFFT_FORWARD));
        checkCudaErrors(cufftExecC2C(plan_adv, reinterpret_cast<cufftComplex *>(d_filter_kernel), reinterpret_cast<cufftComplex *>(d_filter_kernel), CUFFT_FORWARD));

        // Multiply the coefficients together and normalize the result
        ComplexPointwiseMulAndScale<<<32, 256>>>(d_signal, d_filter_kernel, new_size, 1.0f / new_size);
        getLastCudaError("Kernel execution failed [ ComplexPointwiseMulAndScale ]");

        // Transform signal back
        checkCudaErrors(cufftExecC2C(plan_adv, reinterpret_cast<cufftComplex *>(d_signal), reinterpret_cast<cufftComplex *>(d_signal), CUFFT_INVERSE));
    }

    // Record the stop event
    cudaEventRecord(stop, NULL);
    cudaEventSynchronize(stop);
    cudaDeviceSynchronize();

    float msecTotal = 0.0f;
    cudaEventElapsedTime(&msecTotal, start, stop);
    printf("Performance: %lf seconds\n", msecTotal / 1000.0);

    // Cleanup memory
    free(h_signal);
    free(h_filter_kernel);
    free(h_padded_signal);
    free(h_padded_filter_kernel);
    checkCudaErrors(cudaFree(d_signal));
    checkCudaErrors(cudaFree(d_filter_kernel));
    checkCudaErrors(cudaFree(d_workspace));
}

////////////////////////////////////////////////////////////////////////////////
// Padding function
////////////////////////////////////////////////////////////////////////////////
int PadData(const Complex *signal, Complex **padded_signal, int signal_size, const Complex *filter_kernel, Complex **padded_filter_kernel, int filter_kernel_size) {
    int minRadius = filter_kernel_size / 2;
    int maxRadius = filter_kernel_size - minRadius;
    int new_size = signal_size + maxRadius;

    Complex *new_data = reinterpret_cast<Complex *>(malloc(sizeof(Complex) * new_size));
    memcpy(new_data, signal, signal_size * sizeof(Complex));
    memset(new_data + signal_size, 0, (new_size - signal_size) * sizeof(Complex));
    *padded_signal = new_data;

    new_data = reinterpret_cast<Complex *>(malloc(sizeof(Complex) * new_size));
    memcpy(new_data, filter_kernel + minRadius, maxRadius * sizeof(Complex));
    memset(new_data + maxRadius, 0, (new_size - filter_kernel_size) * sizeof(Complex));
    memcpy(new_data + new_size - minRadius, filter_kernel, minRadius * sizeof(Complex));
    *padded_filter_kernel = new_data;

    return new_size;
}

////////////////////////////////////////////////////////////////////////////////
// Complex operations
////////////////////////////////////////////////////////////////////////////////
static __device__ __host__ inline Complex ComplexAdd(Complex a, Complex b) {
    Complex c;
    c.x = a.x + b.x;
    c.y = a.y + b.y;
    return c;
}

static __device__ __host__ inline Complex ComplexScale(Complex a, float s) {
    Complex c;
    c.x = s * a.x;
    c.y = s * a.y;
    return c;
}

static __device__ __host__ inline Complex ComplexMul(Complex a, Complex b) {
    Complex c;
    c.x = a.x * b.x - a.y * b.y;
    c.y = a.x * b.y + a.y * b.x;
    return c;
}

static __global__ void ComplexPointwiseMulAndScale(Complex *a, const Complex *b, int size, float scale) {
    const int numThreads = blockDim.x * gridDim.x;
    const int threadID = blockIdx.x * blockDim.x + threadIdx.x;

    for (int i = threadID; i < size; i += numThreads) {
        a[i] = ComplexScale(ComplexMul(a[i], b[i]), scale);
    }
}

