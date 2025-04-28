#include <assert.h>
#include <stdio.h>
#include <unistd.h>
#include <string.h>
#include "safecuda.h"

// threads per block
#ifndef TPB
#warning "TPB not defined, using default"
#define TPB 64
#endif
// num float in 4k page
#ifndef PSIZE
#warning "PSIZE not defined, using default"
#define PSIZE 1024lu
#endif
// stride length
#ifndef STRIDE
#warning "STRIDE not defined, using default"
#define STRIDE 1
#endif
// Amount of virtual memory required, in GB 
#ifndef MEM
#warning "MEM not defined, using default"
#define MEM 12lu
#endif
// array location, 0:default malloc, 1:pinned to host, TODO 2:explicitly migrated to GPU 
#ifndef ARRAY_LOC
#warning "ARRAY_LOC not defined, using default"
#define ARRAY_LOC 1
#endif
// read flag (Do you want to read)
#ifndef READ
#warning "READ not defined, using default"
#define READ 1
#endif
// write flag (Do you want to write)
#ifndef WRITE
#warning "WRITE not defined, using default"
#define WRITE 0
#endif
// write val
#ifndef WRITE_VAL
#warning "WRITE_VAL not defined, using default"
#define WRITE_VAL 1.2
#endif
// Reuse flag
#ifndef REUSE 
#warning "REUSE not defined, using default"
#define REUSE 0
#endif
// Reuse stride
#ifndef REUSE_STRIDE 
#warning "REUSE_STRIDE not defined, using default"
#define REUSE_STRIDE 0
#endif 

//CALCULATED MACROS
#define NUM_FLOATS (MEM << 28) //2^30 bytes in a GB / 2^4 bytes in a float   


// meta stuff
void get_current_prop(cudaDeviceProp* prop)
{
    CHECK_CUDA_ERROR();
    int dev_id;
    int count;
    CHECK_CUDA_ERROR();
    cudaGetDeviceCount(&count);
    CHECK_CUDA_ERROR();
    if (!count)
    {
        fprintf(stderr, "No devices found.\n");
    }
    cudaGetDevice(&dev_id);
    CHECK_CUDA_ERROR();
    cudaGetDeviceProperties(prop, dev_id);
    CHECK_CUDA_ERROR();
}


int get_exact_hw_threads()
{
    CHECK_CUDA_ERROR();
    cudaDeviceProp prop;
    CHECK_CUDA_ERROR();
    get_current_prop(&prop);
    CHECK_CUDA_ERROR();
    //printf("prop.multiProcessorCount: %d\nprop.maxThreadsPerMultiProcessor: %d\n",  prop.multiProcessorCount, prop.maxThreadsPerMultiProcessor);
    return prop.multiProcessorCount * prop.maxThreadsPerMultiProcessor;
}


// dropping this for now but it's better to explicitly have the load instead of trying to trick the compiler not 
// to optimize it out
// Problem: compiler doesn't load parameter if only reference is (volatile?) inline asm
static __device__ __inline__ float __gload(const float* a)
{
    float val;
    asm volatile("ld.global.f32 %0, [%1];" : "=f"(val), "=l"(a));
    return val;
}



//Write to an array using the set stride length
extern "C"
__global__ void threads_write(float* a, size_t cycle_length)
{
    //Note: if gcd(stride, num_floats) is 1, there are no stride cycles
    //Else, the stride cycle length is num_floats / gcd(stride, num_floats)
    size_t idx = blockIdx.x * blockDim.x + threadIdx.x;
    size_t cycle_offset = idx / cycle_length;
    size_t stride_idx = (idx * (STRIDE) + cycle_offset) % NUM_FLOATS;
    a[stride_idx] = WRITE_VAL;
}

//Read from an array using the set stride length
extern "C"
__global__ void threads_read(float* a, float* b, size_t cycle_length)
{
 
    size_t idx = blockIdx.x * blockDim.x + threadIdx.x;
    size_t cycle_offset = idx / cycle_length;
    size_t stride_idx = (idx * (STRIDE) + cycle_offset) % NUM_FLOATS;
    float val = a[stride_idx];
    if (val == 0.35) 
    {
	b[idx] = val;
	a[idx] = b[idx + 73];
    }
}

//Write to an array using the set stride length and reuse
extern "C"
__global__ void threads_write_reuse(float* a, size_t cycle_length)
{
    //Note: if gcd(stride, num_floats) is 1, there are no stride cycles
    //Else, the stride cycle length is num_floats / gcd(stride, num_floats)
    size_t idx = blockIdx.x * blockDim.x + threadIdx.x;
    size_t cycle_offset = idx / cycle_length;
    size_t stride_idx = (idx * (STRIDE) + cycle_offset) % NUM_FLOATS;
    a[stride_idx] = WRITE_VAL;
    //Below adds the reuse functionality
    for(int i = 0; i < REUSE; i++){
	    stride_idx = (idx * STRIDE + cycle_offset + (REUSE_STRIDE*i)) % NUM_FLOATS;
	    a[stride_idx] = WRITE_VAL;
    }
}

//Read from an array using the set stride length and reuse
extern "C"
__global__ void threads_read_reuse(float* a, float* b, size_t cycle_length)
{
 
    size_t idx = blockIdx.x * blockDim.x + threadIdx.x;
    size_t cycle_offset = idx / cycle_length;
    size_t stride_idx = (idx * (STRIDE) + cycle_offset) % NUM_FLOATS;
    float val = a[stride_idx];
    if (val == 0.35) 
    {
	b[idx] = val;
	a[idx] = b[idx + 73];
    
    }
    //Below adds the reuse functionality
    for(int i = 0; i < REUSE; i++){
	    stride_idx = (idx * STRIDE + cycle_offset + (REUSE_STRIDE*i)) % NUM_FLOATS;
	    val = a[stride_idx];
	    if (val == 0.35){
		    b[idx] = val;
		    a[idx] = b[idx + 73];
	    }
    } 
}

//The length of the a cycle in modular addition is moduland/GCD(moduland, additive)
size_t get_cycle_length()
{
   size_t a = NUM_FLOATS;
   size_t b = STRIDE;
   //computes GCD(a,b) 
   while (b > 0) {
	   size_t c = a % b;
	   a = b;
	   b = c;
   }
   return NUM_FLOATS/a;
}


// INIT

// not using gpu init for now
__global__ void init_array(float* a, size_t size)
{
    size_t idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < size)
    {
        a[idx] = 0.5f;
    }
}

void cpu_init(float* a, float val)
{
    #pragma omp simd
    for (size_t i = 0; i < NUM_FLOATS; ++i)
    {
        a[i] = val;
    }
}

void cpu_init_zero(float* a)
{
    cpu_init(a, 0.0f);
}

// ALLOCS
float* managed_alloc()
{
    float* a;
    size_t alloc_size = NUM_FLOATS * sizeof(float);

    CUDA_CHECK(cudaMallocManaged(&a, alloc_size));
    return a;
}

float* managed_remote_alloc()
{
    float* a = managed_alloc();
    CUDA_CHECK(cudaMemAdvise(a, NUM_FLOATS * sizeof(float), cudaMemAdviseSetPreferredLocation, cudaCpuDeviceId));
    //CUDA_CHECK(cudaMemAdvise(a, NUM_FLOATS * sizeof(float), cudaMemAdviseSetAccessedBy, cudaGpuDeviceId));
    return a; 
}


// MAIN
void run_benchmarks()
{
    size_t numThreads = NUM_FLOATS;
    //if(REUSE) numThreads = get_exact_hw_threads();
    //else numThreads = NUM_FLOATS; 

    // TODO everything below here basically needs to be factored into its own benchmark function or similar design,
    // as we will have many different combinations of each function. 
    float* a;
    printf("ArrayLoc: %d\n", ARRAY_LOC); 
    printf("NumThreads: %zu\n", numThreads);
    printf("NumFloats: %zu\n", NUM_FLOATS);
    printf("MemSize: %d\n", MEM);
    switch(ARRAY_LOC){
	case 0: a = managed_alloc();
		break;
	case 1: a = managed_remote_alloc();
   		break;
	default: a = managed_alloc();
		break;
    }
    cpu_init_zero(a);
    size_t cycle_length = get_cycle_length();
    printf("cycle length: %zu\n", cycle_length);
    if(READ){ 
	if(REUSE) threads_read_reuse<<<numThreads/TPB, TPB>>>(a, a, cycle_length);
	else threads_read<<<numThreads/TPB, TPB>>>(a, a, cycle_length);
    	CUDA_CHECK(cudaDeviceSynchronize());    
    }
    if(WRITE){
	 if(REUSE) threads_write_reuse<<<numThreads/TPB, TPB>>>(a, cycle_length);
	 else threads_write<<<numThreads/TPB, TPB>>>(a, cycle_length);
         CUDA_CHECK(cudaDeviceSynchronize());
    }
}

int main(void)
{
    if(REUSE_STRIDE > 0 && REUSE > 1024/REUSE_STRIDE){
	    printf("Error: REUSE value out of bounds.");
	    exit(1);
    }
    CHECK_CUDA_ERROR();	
    printf("main start\n");
    run_benchmarks();
    int pid = getpid();
    char spid[100];
    sprintf(spid, "%d", pid);
    char cmd[100];

    snprintf(cmd, sizeof cmd, "%s%s%s", "sudo grep ^VmHWM /proc/", spid, "/status");
    int status = system(cmd);
    printf("main end\n");
}
