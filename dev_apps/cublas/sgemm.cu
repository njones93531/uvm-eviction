#include <iostream>
#include <chrono>
#include <cublas_v2.h>
#include <cblas.h>
#include <getopt.h>
#include <cuda_runtime.h>

using namespace std::chrono;

__global__ void warmup()
{
    return;
}

void cpu_multiply(float *A, float *B, float *C, size_t N, size_t iterations) {
    for (size_t i = 0; i < iterations; ++i) {
        cblas_sgemm(CblasRowMajor, CblasNoTrans, CblasNoTrans,
                N, N, N, 1.0f, A, N, B, N, 0.0f, C, N);
    }
}

void gpu_multiply(float *A, float *B, float *C, size_t N, size_t iterations) {
    cublasHandle_t handle;
    const float alpha = 1.0f;
    const float beta = 0.0f;

    warmup<<<1, 1>>>();
    cudaDeviceSynchronize();
    cublasCreate(&handle);

    cudaEvent_t start, stop;
    float elapsedTime;
    cudaEventCreate(&start);
    cudaEventCreate(&stop);
    cudaEventRecord(start, 0);

    for (size_t i = 0; i < iterations; ++i) {
        cublasSgemm(handle, CUBLAS_OP_N, CUBLAS_OP_N, N, N, N, &alpha, A, N, B, N, &beta, C, N);
    }

    cudaEventRecord(stop, 0);
    fprintf(stderr, "waiting on gpu kernel\n");
    cudaEventSynchronize(stop);
    cudaEventElapsedTime(&elapsedTime, start, stop);

    float gflops = iterations * (2.0f * N * N * N - N * N) / (elapsedTime / 1000.0f) / 1e9;
    printf("GPU,%zu,%f,%f\n", N, elapsedTime / 1000.0, gflops);

    //   cublasGetMatrix(N, N, sizeof(float), d_C, N, C, N);

    cudaEventDestroy(start);
    cudaEventDestroy(stop);
    cublasDestroy(handle);
}

int main(int argc, char **argv) {
    size_t N = 0;
    size_t iterations = 1;
    bool use_cpu = false;

    int opt;
    fprintf(stderr, "start\n");
    while ((opt = getopt(argc, argv, "n:ci:")) != -1) {
        switch (opt) {
            case 'n':
                N = std::stoull(optarg);
                break;
            case 'c':
                use_cpu = true;
                break;
            case 'i':
                iterations = std::stoull(optarg);
                break;
            default:
                std::cerr << "Usage: " << argv[0] << " -n N [-c] [-i iterations]" << std::endl;
                return 1;
        }
    }

    if (!N) {
        std::cerr << "Usage: " << argv[0] << " -n N [-c] [-i iterations]" << std::endl;
        return 1;
    }

    float* A;
    float* B;//= new float[N * N];
    float* C;// = new float[N * N];

    cudaMallocManaged(&A, sizeof(float) * N * N);
    cudaMallocManaged(&B, sizeof(float) * N * N);
    cudaMallocManaged(&C, sizeof(float) * N * N);
    fprintf(stderr, "memory alloced\n");

    unsigned int seed = 243;

    fprintf(stderr, "initializing arrays\n");
    #pragma omp parallel for simd private(seed)
    for (size_t i = 0; i < N * N; ++i) 
    {
        A[i] = static_cast<float>(rand_r(&seed)) / RAND_MAX;
        B[i] = static_cast<float>(rand_r(&seed)) / RAND_MAX;
    }
    fprintf(stderr, "arrays initialized\n");

    if (use_cpu) 
    {
        fprintf(stderr, "timing cpu\n");
        high_resolution_clock::time_point start = high_resolution_clock::now();
        cpu_multiply(A, B, C, N, iterations);
        high_resolution_clock::time_point end = high_resolution_clock::now();
        duration<float> elapsed_time = duration_cast<duration<float>>(end - start);
        float gflops = iterations * (2.0f * N * N * N - N * N) / elapsed_time.count() / 1e9;
        printf("CPU,%zu,%f,%f\n", N, elapsed_time.count(), gflops);
    } 
    else 
    {
        fprintf(stderr, "timing gpu\n");
        gpu_multiply(A, B, C, N, iterations);
    }

    cudaFree(A);
    cudaFree(B);
    cudaFree(C);

    return 0;
}

