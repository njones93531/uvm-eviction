#include <cuda_runtime.h>
#include <stdio.h>

#define PAGE_SIZE 4096
#define ELE_IN_PAGE (PAGE_SIZE / sizeof(int))
#define NUM_PAGES 512
#define ALLOC_SIZE (PAGE_SIZE * NUM_PAGES)
#define NUM_ELE (ALLOC_SIZE/sizeof(int))

__global__ void loop(int* data)
{
    int i;
    int j;
    // Repeat 16 times; 16 iter * 16 pages per 64kb = 256 accesses
    for (i = 0; i < 1; ++i)
    {
        // touch each page; offset ele by 1 to offset compiler shenanigans (hopefully -O0 did this anyway...)
        for (j = 0; j < NUM_PAGES; ++j)
        {
            printf("virt, %p\n", &data[j * ELE_IN_PAGE]);
            data[j * ELE_IN_PAGE] = j;
        }
    }
}

int main(void)
{
    int* data;
    printf("alloc\n");
    cudaMallocManaged(&data, ALLOC_SIZE);
    cudaMemAdvise(data, ALLOC_SIZE, cudaMemAdviseSetPreferredLocation, cudaCpuDeviceId);
    printf("set\n");
    memset(data, 0, ALLOC_SIZE);

    printf("exec\n");
    loop<<<1,1>>>(data);
    cudaDeviceSynchronize();

    cudaFree(data);
    printf("done\n");
    return 0;
}
