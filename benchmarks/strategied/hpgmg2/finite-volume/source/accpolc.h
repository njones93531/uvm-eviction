#ifndef ACCESS_POLICY_H
#define ACCESS_POLICY_H

// System includes
#include <stdio.h>
#include <assert.h>
#include <stdint.h>
#include <string.h>
#include <stdlib.h>

// CUDA runtime
#include <cuda_runtime.h>
#include <unistd.h>
#include <fcntl.h>

#ifndef CUDA_CHECK
#define CUDA_CHECK(status) \
  if (status != cudaSuccess) \
  { \
    printf("%s:%d CudaError: %s\n", __FILE__, __LINE__, cudaGetErrorString(status)); \
    assert(0); \
  }
#endif

#ifndef CHECK_CUDA_ERROR
#define CHECK_CUDA_ERROR() \
do { \
    cudaError_t err = cudaGetLastError(); \
    if (err != cudaSuccess) \
    { \
        printf("error=%d name=%s at ln: %d\n", err, cudaGetErrorString(err), __LINE__); \
        exit(1); \
    } \
} while (0)
#endif

typedef struct {
    void *mem_pressure;
    int aoi;
    size_t aoi_size;
    double perc;
} AccessPolicy;

void initAccessPolicy(AccessPolicy *policy) {
    policy->mem_pressure = NULL;
    policy->aoi = -1;
    policy->aoi_size = 0;
    policy->perc = -1.0;
}

void setMemPressure(AccessPolicy *policy, size_t size) {
    if (size <= 1000) return;
    cudaMalloc(&policy->mem_pressure, size);
    CHECK_CUDA_ERROR();
    if (policy->mem_pressure == NULL) {
        printf("mem pressure failed\n");
        exit(-1);
    }
    double pressureGB = ((double)policy->aoi_size * policy->perc) / (1024.0 * 1024.0 * 1024.0);
    printf("Allocated %.2f GB (%.2f%% of allocation %d) of memory pressure\n",
           pressureGB, policy->perc, policy->aoi);
}

void freeMemPressure(AccessPolicy *policy) {
    cudaDeviceSynchronize();
    if (policy->mem_pressure == NULL) return;
    cudaFree(policy->mem_pressure);
}

void usage(const char *argv0) {
    printf("Usage: %s <benchmark args> <flag1> <arg1> <flag2...>\n", argv0);
    printf("Available flags:\n");
    printf("-p: policy\t| string \t| one character per allocation\n");
    printf("-aoi: alloc of interest\t| int\t| which alloc to apply mem pressure\n");
    printf("-r: pressure\t| float\t| fraction of aoi's size to block in memory as mem pressure\n");
}

char parseCLA(AccessPolicy *policy, size_t size, int alloc_num, int argc, char **argv) {
    char flag = '-';
    int args = 0;
    for (int i = 1; i < argc; i++) {
        if (argv[i][0] == '-' && argv[i][1] == 'p') {
            args = 1;
            flag = argv[++i][alloc_num];
        } else if (strcmp(argv[i], "-aoi") == 0 && atoi(argv[i + 1]) == alloc_num) {
            args = 1;
            policy->aoi_size = size;
            policy->aoi = atoi(argv[i + 1]);
        } else if (argv[i][0] == '-' && argv[i][1] == 'r') {
            args = 1;
            policy->perc = atof(argv[++i]);
            if (policy->aoi == alloc_num) {
                setMemPressure(policy, (size_t)((double)policy->aoi_size * policy->perc));
            }
        }
    }
    if (!args) usage(argv[0]);
    return flag;
}

void setAllocationPolicy(AccessPolicy *policy, void **a, size_t size, int alloc_num, int argc, char **argv) {
    char flag = parseCLA(policy, size, alloc_num, argc, argv);
    switch (flag) {
        case 'm':
            break;
        case 'd': {
            void *devptr;
            cudaMalloc(&devptr, size);
            CHECK_CUDA_ERROR();
            cudaMemcpy(devptr, *a, size, cudaMemcpyHostToDevice);
            CHECK_CUDA_ERROR();
            cudaFree(*a);
            *a = devptr;
            break;
        }
        case 'a': {
            cudaMemAdvise(*a, size, cudaMemAdviseSetPreferredLocation, 0);
            cudaMemPrefetchAsync(*a, size, 0, 0);
            break;
        }
        case 'h': {
            CUDA_CHECK(cudaMemAdvise(*a, size, cudaMemAdviseSetPreferredLocation, cudaCpuDeviceId));
            CUDA_CHECK(cudaMemAdvise(*a, size, cudaMemAdviseSetAccessedBy, 0));
            break;
        }
        case 's': {
            cudaMemAdvise(*a, size, cudaMemAdviseSetPreferredLocation, 0);
            printf("Perc is %.2f\n", policy->perc);
            if (policy->perc <= 0.0) break;
            cudaMemAdvise(*a, (policy->perc * size), cudaMemAdviseSetPreferredLocation, cudaCpuDeviceId);
            cudaMemPrefetchAsync((uint8_t *)*a + (size_t)(size * policy->perc), (1.0 - policy->perc) * size, 0, 0);
            break;
        }
        case 'x': {
            cudaMemAdvise(*a, size, cudaMemAdviseSetPreferredLocation, 0);
            cudaMemAdvise(*a, (policy->perc * size), cudaMemAdviseSetPreferredLocation, cudaCpuDeviceId);
            break;
        }
        default:
            printf("Policy flag '%c' used on allocation %d is not supported.\n", flag, alloc_num);
            exit(1);
    }
}

#endif

