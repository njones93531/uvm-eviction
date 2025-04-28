#include <cuda_runtime.h>
#include <iostream>

__global__ void incrementKernel(int *data, int size) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < size) {
        data[idx] += 1;
    }
}

int main(int argc, char **argv) {
    if (argc != 2) {
        std::cerr << "Usage: " << argv[0] << " <problem size>" << std::endl;
        return -1;
    }

    int size = atoi(argv[1]);
    if (size <= 0) {
        std::cerr << "Error: problem size must be a positive integer." << std::endl;
        return -1;
    }

    int *data;
    cudaMallocManaged(&data, size * sizeof(int));

    for (int i = 0; i < size; i++) {
        data[i] = i;
    }

    int blockSize = 128;
    int numBlocks = (size + blockSize - 1) / blockSize;

    incrementKernel<<<numBlocks, blockSize>>>(data, size);
    cudaDeviceSynchronize();

    cudaFree(data);
    return 0;
}

