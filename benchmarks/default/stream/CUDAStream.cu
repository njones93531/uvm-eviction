
// Copyright (c) 2015-16 Tom Deakin, Simon McIntosh-Smith,
// University of Bristol HPC
//
// For full license terms please see the LICENSE file distributed with this
// source code


#include "CUDAStream.h"

void check_error(void)
{
  cudaError_t err = cudaGetLastError();
  if (err != cudaSuccess)
  {
    std::cerr << "Error: " << cudaGetErrorString(err) << std::endl;
    exit(err);
  }
}

template <class T>
CUDAStream<T>::CUDAStream(const unsigned int ARRAY_SIZE, const int device_index)
{

  // The array size must be divisible by TBSIZE for kernel launches
  if (ARRAY_SIZE % TBSIZE != 0)
  {
    std::stringstream ss;
    ss << "Array size must be a multiple of " << TBSIZE;
    throw std::runtime_error(ss.str());
  }

  // Set device
  int count;
  cudaGetDeviceCount(&count);
  check_error();
  if (device_index >= count)
    throw std::runtime_error("Invalid device index");
  cudaSetDevice(device_index);
  check_error();

  // Print out device information
  std::cout << "Using CUDA device " << getDeviceName(device_index) << std::endl;
  std::cout << "Driver: " << getDeviceDriver(device_index) << std::endl;

  array_size = ARRAY_SIZE;

  // Allocate the host array for partial sums for dot kernels
  sums = (T*)malloc(sizeof(T) * DOT_NUM_BLOCKS);

  // Check buffers fit on the device
  cudaDeviceProp props;
  cudaGetDeviceProperties(&props, 0);
//  if (props.totalGlobalMem < 3*ARRAY_SIZE*sizeof(T))
//    throw std::runtime_error("Device does not have enough memory for all 3 buffers");
/*
  // Create device buffers
  cudaMalloc(&d_a, ARRAY_SIZE*sizeof(T));
  check_error();
  cudaMalloc(&d_b, ARRAY_SIZE*sizeof(T));
  check_error();
  cudaMalloc(&d_c, ARRAY_SIZE*sizeof(T));
  check_error();
  cudaMalloc(&d_sum, DOT_NUM_BLOCKS*sizeof(T));
  check_error();
*/  
  cudaMallocManaged(&d_a, ARRAY_SIZE*sizeof(T));
  check_error();
  cudaMallocManaged(&d_b, ARRAY_SIZE*sizeof(T));
  check_error();
  cudaMallocManaged(&d_c, ARRAY_SIZE*sizeof(T));
  check_error();
  cudaMallocManaged(&d_sum, DOT_NUM_BLOCKS*sizeof(T));
  check_error();


  printf("Start address of d_a:\t%p\n", &(d_a[0]));
  printf("Start address of d_b:\t%p\n", &(d_b[0]));
  printf("Start address of d_c:\t%p\n", &(d_c[0]));
  printf("Start address of d_sum:\t%p\n", &(d_sum[0]));

  printf("Size of d_a:\t%.2f\n", (float)(ARRAY_SIZE * sizeof(T)) / (1024. * 1024. * 1024.));
  printf("Size of d_b:\t%.2f\n", (float)(ARRAY_SIZE * sizeof(T)) / (1024. * 1024. * 1024.));
  printf("Size of d_c:\t%.2f\n", (float)(ARRAY_SIZE * sizeof(T)) / (1024. * 1024. * 1024.));
  printf("Size of d_sum:\t%.2f\n", (float)(DOT_NUM_BLOCKS * sizeof(T)) / (1024. * 1024. * 1024.));



std::cout << "alloced," << ARRAY_SIZE*sizeof(T) * 3 << std::endl; 
//cudaMemAdvise(d_a, array_size, cudaMemAdviseSetPreferredLocation, 0);
//cudaMemAdvise(d_a, array_size, cudaMemAdviseSetAccessedBy, 0);
//cudaMemAdvise(d_a, array_size, cudaMemAdviseSetReadMostly, 0);
  
}


template <class T>
CUDAStream<T>::~CUDAStream()
{
  free(sums);

  cudaFree(d_a);
  check_error();
  cudaFree(d_b);
  check_error();
  cudaFree(d_c);
  check_error();
  cudaFree(d_sum);
  check_error();
}


template <typename T>
__global__ void init_kernel(T * a, T * b, T * c, T initA, T initB, T initC)
{
  const size_t i = blockDim.x * blockIdx.x + threadIdx.x;
  a[i] = initA;
  b[i] = initB;
  c[i] = initC;
}

template <class T>
void CUDAStream<T>::init_arrays(T initA, T initB, T initC)
{
  //init_kernel<<<array_size/TBSIZE, TBSIZE>>>(d_a, d_b, d_c, initA, initB, initC);
  for (size_t i = 0; i < array_size; i++)
  {
     d_a[i] = initA;
     d_b[i] = initB;
     d_c[i] = initC;
  }
  //check_error();
  //cudaDeviceSynchronize();
  //check_error();
}

template <class T>
void CUDAStream<T>::read_arrays(std::vector<T>& a, std::vector<T>& b, std::vector<T>& c)
{
  // Copy device memory to host
//  cudaMemcpy(a.data(), d_a, a.size()*sizeof(T), cudaMemcpyDeviceToHost);
//  check_error();
//  cudaMemcpy(b.data(), d_b, b.size()*sizeof(T), cudaMemcpyDeviceToHost);
//  check_error();
//  cudaMemcpy(c.data(), d_c, c.size()*sizeof(T), cudaMemcpyDeviceToHost);
//  check_error();
memcpy(a.data(), d_a, a.size() * sizeof(T));
memcpy(b.data(), d_b, b.size() * sizeof(T));
memcpy(c.data(), d_c, c.size() * sizeof(T));
}


template <typename T>
__global__ void copy_kernel(const T * a, T * c)
{
  const size_t i = blockDim.x * blockIdx.x + threadIdx.x;
  c[i] = a[i];
}

template <class T>
void CUDAStream<T>::copy()
{
  copy_kernel<<<array_size/TBSIZE, TBSIZE>>>(d_a, d_c);
  check_error();
  cudaDeviceSynchronize();
  check_error();
}

template <typename T>
__global__ void mul_kernel(T * b, const T * c)
{
  const T scalar = startScalar;
  const size_t i = blockDim.x * blockIdx.x + threadIdx.x;
  b[i] = scalar * c[i];
}

template <class T>
void CUDAStream<T>::mul()
{
  mul_kernel<<<array_size/TBSIZE, TBSIZE>>>(d_b, d_c);
  check_error();
  cudaDeviceSynchronize();
  check_error();
}

template <typename T>
__global__ void add_kernel(const T * a, const T * b, T * c)
{
  const size_t i = blockDim.x * blockIdx.x + threadIdx.x;
  c[i] = a[i] + b[i];
}

template <class T>
void CUDAStream<T>::add()
{
  add_kernel<<<array_size/TBSIZE, TBSIZE>>>(d_a, d_b, d_c);
  check_error();
  cudaDeviceSynchronize();
  check_error();
}

template <typename T>
__global__ void triad_kernel(T * a, const T * b, const T * c)
{
  const T scalar = startScalar;
  const size_t i = blockDim.x * blockIdx.x + threadIdx.x;
  a[i] = b[i] + scalar * c[i];
}

template <typename T>
__global__ void unalign_kernel(T * a, const T * b, const T * c)
{
  const T scalar = startScalar;
  const size_t i = blockDim.x * blockIdx.x + threadIdx.x;
  a[i] = b[i];
}


template <class T>
void CUDAStream<T>::triad()
{
  //Include to make UVM upset 
  //unalign_kernel<<<array_size/TBSIZE, TBSIZE>>>(d_a, d_b, d_c);
  triad_kernel<<<array_size/TBSIZE, TBSIZE>>>(d_a, d_b, d_c);
  check_error();
  cudaDeviceSynchronize();
  check_error();
}

template <class T>
__global__ void dot_kernel(const T * a, const T * b, T * sum, size_t array_size)
{
  __shared__ T tb_sum[TBSIZE];

  size_t i = blockDim.x * blockIdx.x + threadIdx.x;
  const size_t local_i = threadIdx.x;

  tb_sum[local_i] = 0.0;
  for (; i < array_size; i += blockDim.x*gridDim.x)
    tb_sum[local_i] += a[i] * b[i];

  for (size_t offset = blockDim.x / 2; offset > 0; offset /= 2)
  {
    __syncthreads();
    if (local_i < offset)
    {
      tb_sum[local_i] += tb_sum[local_i+offset];
    }
  }

  if (local_i == 0)
    sum[blockIdx.x] = tb_sum[local_i];
}

template <class T>
T CUDAStream<T>::dot()
{
  dot_kernel<<<DOT_NUM_BLOCKS, TBSIZE>>>(d_a, d_b, d_sum, array_size);
  cudaDeviceSynchronize();
  check_error();
  memcpy(sums, d_sum, DOT_NUM_BLOCKS*sizeof(T));
//  cudaMemcpy(sums, d_sum, DOT_NUM_BLOCKS*sizeof(T), cudaMemcpyDeviceToHost);
//  check_error();

  T sum = 0.0;
  for (size_t i = 0; i < DOT_NUM_BLOCKS; i++)
    sum += sums[i];

  return sum;
}

void listDevices(void)
{
  // Get number of devices
  int count;
  cudaGetDeviceCount(&count);
  check_error();

  // Print device names
  if (count == 0)
  {
    std::cerr << "No devices found." << std::endl;
  }
  else
  {
    std::cout << std::endl;
    std::cout << "Devices:" << std::endl;
    for (int i = 0; i < count; i++)
    {
      std::cout << i << ": " << getDeviceName(i) << std::endl;
    }
    std::cout << std::endl;
  }
}


std::string getDeviceName(const int device)
{
  cudaDeviceProp props;
  cudaGetDeviceProperties(&props, device);
  check_error();
  return std::string(props.name);
}


std::string getDeviceDriver(const int device)
{
  cudaSetDevice(device);
  check_error();
  int driver;
  cudaDriverGetVersion(&driver);
  check_error();
  return std::to_string(driver);
}

template class CUDAStream<float>;
template class CUDAStream<double>;
