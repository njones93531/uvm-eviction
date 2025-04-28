#include <cuda_runtime.h>
#include <stdio.h>

__global__ void warmup(int* foo)
{
    printf("init_val = %d\n", foo[0]);
    return;
}

int main(void)
{
    int* foo = NULL;
    cudaMallocManaged(&foo, 4);
    *foo = 7;
    warmup<<<1,1>>>(foo);
    cudaDeviceSynchronize();
    return 0;
}
