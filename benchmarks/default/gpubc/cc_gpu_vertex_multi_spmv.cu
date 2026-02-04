/*
---------------------------------------------------------------------
 This file is a part of the source code for the paper "Betweenness
 Centrality on GPUs and Heterogeneous Architectures", published in
 GPGPU'13 workshop. If you use the code, please cite the paper.

 Copyright (c) 2013,
 By:    Ahmet Erdem Sariyuce,
        Kamer Kaya,
        Erik Saule,
        Umit V. Catalyurek
---------------------------------------------------------------------
 This file is licensed under the Apache License. For more licensing
 information, please see the README.txt and LICENSE.txt files in the
 main directory.
---------------------------------------------------------------------
*/

#include <assert.h>
#include <iostream>
#include <unistd.h>
#include <cstdlib>
#include <string>
#include <cmath>
#include <algorithm>
#include <list>
#include "common.h"
#include <sys/time.h>
#include <stdio.h>
#include "cuda_common.h"
#include <limits>

//#define TIMER

using namespace std;

//#define B64

#ifdef B64
#define TP unsigned long long
#define SZ 64
#else
#define TP unsigned int
#define SZ 32
#endif

template <typename VtxType, typename EdgeIndex>
__global__ void cc_forward_vertex_multi_spmv (EdgeIndex* d_vptrs, VtxType* d_vjs, TP* d_neighbor, TP* d_current, TP* d_visited, 
					      bool *d_continue, VtxType n, int TPV, int ipv) {
  VtxType tid = ((VtxType)blockIdx.x) * blockDim.x * gridDim.y + ((VtxType)blockIdx.y) * blockDim.x + threadIdx.x;
  
  if (tid < n * TPV) { /* there are n * TPV vertex in total */
    VtxType u = tid / TPV; /* there are TPV threads for each vertex. so u is the vertex id  */
    tid = tid % TPV; /* now tid is the local thread id for the vertex */
    
    EdgeIndex end = d_vptrs[u + 1]; 
    for (VtxType lloc = tid; lloc < ipv; lloc += TPV) {
#ifdef B64
      if (d_visited[u*ipv + lloc] == 0xFFFFFFFFFFFFFFFF) {
	continue;
      }
#else
      if (d_visited[u*ipv + lloc] == 0xFFFFFFFF) {
	continue;
      }
#endif
      TP out = 0; /* this is the variable used to store the bits for the SZ BFSs */
      for(EdgeIndex p = d_vptrs[u]; p < end; p++) {
	out = out | d_current[d_vjs[p] * ipv + lloc];
      }

      if (out != (TP)0) {
	*d_continue = 1;
	*(d_neighbor + u*ipv + lloc) = out;
      }
    }
  }
}

__device__ int bitCount_linear(TP x) {
  int i, res = 0; 
  TP mask;
  for(i = 0; i < SZ; i++) {
    mask = (TP)1 << i; 
    if (x & mask) {
      res++;
    }
  }
  return res;
}

__device__ int bitCount_log(TP x) {
#ifdef B64
  x = x - ((x >> 1) & 0x5555555555555555);
  x = (x & 0x3333333333333333) + ((x >> 2) & 0x3333333333333333);
  x = ((x + (x >> 4)) & 0x0F0F0F0F0F0F0F0F);
  return (x*(0x0101010101010101))>>56;
#else
  x -= ((x >> 1) & 0x55555555);
  x = (x & 0x33333333) + ((x >> 2) & 0x33333333);
  return (((x + (x >> 4)) & 0x0F0F0F0F) * 0x01010101) >> 24;
#endif
}

template <typename VtxType>
__global__ void cc_intermediate_vertex_multi_spmv (float *d_cc, VtxType n, TP* d_neighbor, TP* d_current, TP* d_visited, VtxType* d_dist, int TPV, int ipv) {
  VtxType tid = ((VtxType)blockIdx.x) * blockDim.x * gridDim.y + ((VtxType)blockIdx.y) * blockDim.x + threadIdx.x;
  
  if (tid < n * TPV) { /* there are n * TPV vertex in total */
    VtxType u = tid / TPV; /* there are TPV threads for each vertex. so u is the vertex id  */
    tid = tid % TPV; /* now tid is the local thread id for the vertex */
  
    VtxType vdist = *d_dist;  
    VtxType last = (u+1) * ipv; 
    for (VtxType loc = u * ipv + tid; loc < last; loc += TPV) {
      TP vis = *(d_visited + loc);
#ifdef B64
      if (vis == 0xFFFFFFFFFFFFFFFF) {
	*(d_current + loc) = (TP)0; 
	continue;
      }
#else
      if (vis == 0xFFFFFFFF) {
	*(d_current + loc) = (TP)0; 
	continue;
      }
#endif
      TP curr = *(d_neighbor + loc) & (~vis);
      *(d_current + loc) = curr;
      
      if(curr != (TP)0) {
	*(d_visited + loc) = vis | curr;
	//symetric assumption
	atomicAdd(d_cc + u, bitCount_log(curr) * 1.0 / vdist);
      }
    }
  }
}

/* this sets all the memory to 0 */
template <typename VtxType>
__global__ void cc_init_vertex_multi (TP* d_current, TP* d_neighbor, TP* d_visited, VtxType n, int TPV, int ipv) {
  VtxType tid = ((VtxType)blockIdx.x) * blockDim.x * gridDim.y + ((VtxType)blockIdx.y) * blockDim.x + threadIdx.x;
  
  if(tid < n * TPV) { /* there are n * TPV vertex in total */
    VtxType u = tid / TPV; /* there are TPV threads for each vertex. so u is the vertex id  */
    tid = tid % TPV; /* now tid is the local thread id for the vertex */
    
    VtxType last = (u+1) * ipv; 
    for(VtxType loc = u * ipv + tid; loc < last; loc += TPV) {
      d_neighbor[loc] = d_current[loc] = d_visited[loc] = (TP)0;  /* set the memory to 0*/
    }
  }
}

/* this sets the corresponding bits to the BFS sources in current and visited */
__global__ void cc_sources_vertex_multi (TP* d_current, TP* d_visited, int NB_BFS) {
  int u = blockIdx.x * blockDim.x * gridDim.y + blockIdx.y * blockDim.x + threadIdx.x;
  if(u < NB_BFS) { /* there are NB_BFS sources */
    int gintid = ((u+1)*NB_BFS/SZ - 1) - u/SZ; /* this is the global integer id 
					       (bits are in the reverse order: NB_BFS.... 4 3 2 1 0)*/
    int lbfsid = u % SZ; /* this is the local BFS id in the integer */
    d_current[gintid] = d_visited[gintid] = (TP)1 << lbfsid;   
  }
}

template <typename VtxType>
__global__ void cc_vertex_set_int_multi_spmv (VtxType* dest, VtxType val){
  *dest = val;
}

#ifdef NDEBUG
#error many side effects in asserts
#endif

template <typename VtxType, typename EdgeIndex>
int cc_gpu_vertex_multi_spmv (EdgeIndex* h_vptrs, VtxType* h_vjs, VtxType n, VtxType nb, float *h_cc) {
  EdgeIndex *d_vptrs;
  VtxType  *d_vjs, *d_dist, h_dist;
  float *d_cc;
  bool h_continue, *d_continue;
  

  printf("cc_gpu_vertex_multi_spmv\n");
  /*
  assert (cudaSuccess == cudaMalloc((void **)&d_vptrs, sizeof(EdgeIndex) * (n + 1)));
  assert (cudaSuccess == cudaMalloc((void **)&d_vjs, sizeof(VtxType) * h_vptrs[n]));
  assert (cudaSuccess == cudaMemcpy(d_vptrs, h_vptrs, sizeof(EdgeIndex) * (n + 1), cudaMemcpyHostToDevice));
  assert (cudaSuccess == cudaMemcpy(d_vjs, h_vjs, sizeof(VtxType) * h_vptrs[n], cudaMemcpyHostToDevice));
  */
  
  /* number of BFSs at one kernel execution */
  int NB_BFS = 1024;
  {
    char* str = getenv("NB_BFS");
    if (str != NULL) NB_BFS = atoi(str);
  }
  std::cout<<"NB_BFS="<<NB_BFS<<std::endl;	
  assert(min(nb, n) % NB_BFS == 0);
  
  /* no threads per vertex */	
  int TPV = 32; /* 32 threads will be responsible from a vertex	*/  
  {
    char* str = getenv("TPV");
    if (str != NULL) TPV = atoi(str);
  }
  std::cout<<"TPV="<<TPV<<std::endl;
  
  int ipv = NB_BFS / SZ; /* integers per vertex will be processed by TPV threads */
  std::cout<<"ipv="<<ipv<<std::endl;
  assert(ipv % TPV == 0); /* this is for simplification. to avoid idle threads */

  TP *d_current, *d_neighbor, *d_visited;
  /*
  assert (cudaSuccess == cudaMalloc((void **)&d_current, sizeof(TP) * ipv * n)); 
  assert (cudaSuccess == cudaMalloc((void **)&d_neighbor, sizeof(TP) * ipv * n));
  assert (cudaSuccess == cudaMalloc((void **)&d_visited, sizeof(TP) * ipv * n));
  */
  cudaMallocManaged((void **)&d_current, sizeof(TP) * ipv * n, cudaMemAttachGlobal); /* overall NB_BFS bits are allocated for all of these */
  cudaMallocManaged((void **)&d_neighbor, sizeof(TP) * ipv * n, cudaMemAttachGlobal);
  cudaMallocManaged((void **)&d_visited, sizeof(TP) * ipv * n, cudaMemAttachGlobal);

  //assert (cudaSuccess == cudaMalloc((void **)&d_dist, sizeof(VtxType)));
  cudaMallocManaged((void **)&d_dist, sizeof(VtxType), cudaMemAttachGlobal);
  
  //assert (cudaSuccess == cudaMalloc((void **)&d_cc, sizeof(float) * n));
  cudaMallocManaged((void **)&d_cc, sizeof(float) * n, cudaMemAttachGlobal);
  assert (cudaSuccess == cudaMemset(d_cc, 0, sizeof(float) * n));
  
  //assert (cudaSuccess == cudaMalloc((void **)&d_continue, sizeof(bool)));
  cudaMallocManaged((void **)&d_continue, sizeof(bool), cudaMemAttachGlobal);
  
  std::cout<<"n * TPV = "<<n*TPV<<"\t max limit "<<std::numeric_limits<VtxType>::max()<<std::endl; 
  assert (n  < std::numeric_limits<VtxType>::max()/TPV); //otherwise block thread size will go negative

  //this one is for n threads
  VtxType threads_per_block = n;
  VtxType blocks = 1;
  if(threads_per_block > MTS){
    blocks = (VtxType)ceil(threads_per_block/(double)MTS);
    blocks = (VtxType)ceil(sqrt((float)blocks));
    threads_per_block = MTS;
  }
  dim3 grid;
  grid.x = blocks;
  grid.y = blocks;
  dim3 threads(threads_per_block);
  
  //this one is for TPV * n threads
  VtxType threads_per_block2 = n * TPV;
  VtxType blocks2 = 1;
  if(threads_per_block2 > MTS){
    blocks2 = (VtxType)ceil(threads_per_block2/(double)MTS);
    blocks2 = (VtxType)ceil(sqrt((float)blocks2));
    threads_per_block2 = MTS;
  }
  dim3 grid2;
  grid2.x = blocks2;
  grid2.y = blocks2;
  dim3 threads2(threads_per_block2);
  
  //this one is for NB_BFS threads
  int threads_per_block3 = NB_BFS;
  int blocks3 = 1;
  if(threads_per_block3 > MTS){
    blocks3 = (int)ceil(threads_per_block3/(double)MTS);
    blocks3 = (int)ceil(sqrt((float)blocks3));
    threads_per_block3 = MTS;
  }
  dim3 grid3;
  grid3.x = blocks3;
  grid3.y = blocks3;
  dim3 threads3(threads_per_block3);
  
  cerr<<"cuda parameters: " <<blocks<<" "<<threads_per_block<<" "<<blocks2<<" "<<threads_per_block2<<" "<<blocks3<<" "<<threads_per_block3<<" "<<std::endl;
  
#ifdef TIMER
  struct timeval t1, t2, gt1, gt2; double time;
#endif
  
  VtxType diameter;
  for(VtxType i = 0; i < min(nb, n); i += NB_BFS){
#ifdef TIMER
    gettimeofday(&t1, 0);
#endif
    h_dist = 0;
    cc_vertex_set_int_multi_spmv<<<1,1>>> (d_dist, h_dist); /* set the current level to 0 */
    
    /* initialize the current, neighbor, and visited for all the vertices */    
    cc_init_vertex_multi<<<grid2,threads2>>>(d_current, d_neighbor, d_visited, n, TPV, ipv);

    /* set the appropriate bits for the sources in the visited and current arrays */
    cc_sources_vertex_multi<<<grid3,threads3>>> (d_current + i * ipv, d_visited + i * ipv, NB_BFS);
#ifdef TIMER
    cudaDeviceSynchronize();
    CudaCheckError();

    gettimeofday(&t2, 0);
    time = (1000000.0*(t2.tv_sec-t1.tv_sec) + t2.tv_usec-t1.tv_usec)/1000000.0;
    cerr << "initialization takes " << time << " secs"<<std::endl;
    gettimeofday(&gt1, 0);
#endif
   
    //cudaDeviceSynchronize();
    /*
    if(i == 0){
      TP* h_current = (TP*)malloc(sizeof(TP) * ipv * n);
      cudaMemcpy(h_current, d_current, sizeof(TP) * ipv * n, cudaMemcpyDeviceToHost);
      for(int j = 0; j < 40; j++) {
	int id = (j+1)*NB_BFS/SZ - j/SZ - 1;
	printf("%d -- %d -- %llu\n", j, id, d_current[id]); 
      }
      free(d_current);
    }
    */

    do{
#ifdef TIMER
      gettimeofday(&t1, 0);
#endif
      assert (cudaSuccess == cudaMemset(d_continue, 0, sizeof(bool)));

      cc_forward_vertex_multi_spmv<<<grid2, threads2>>> (h_vptrs, h_vjs, d_neighbor, d_current, d_visited, d_continue, n, TPV, ipv);
      cc_vertex_set_int_multi_spmv<<<1,1>>>(d_dist, ++h_dist);

      cc_intermediate_vertex_multi_spmv<<<grid2,threads2>>> (d_cc, n, d_neighbor, d_current, d_visited, d_dist, TPV, ipv);
      assert (cudaSuccess == cudaMemcpy(&h_continue, d_continue, sizeof(bool), cudaMemcpyDeviceToHost));
      
#ifdef TIMER
      cudaDeviceSynchronize();
      CudaCheckError();

      gettimeofday(&t2, 0);
      time = (1000000.0*(t2.tv_sec-t1.tv_sec) + t2.tv_usec-t1.tv_usec)/1000000.0;
      cerr << "level " <<  h_dist << " takes " << time << " secs"<<std::endl;
#endif
    }while(h_continue);
    diameter = h_dist;
#ifdef TIMER
    gettimeofday(&gt2, 0);
    time = (1000000.0*(gt2.tv_sec-gt1.tv_sec) + gt2.tv_usec-gt1.tv_usec)/1000000.0;
    cerr << "A single BFS takes " << time << " secs"<<std::endl;
    gettimeofday(&gt1, 0); // starts back propagation
#endif
  }
  
  assert (cudaSuccess == cudaMemcpy(h_cc, d_cc, sizeof(float) * n, cudaMemcpyDeviceToHost));
  
  std::cout<<"diameter: "<<diameter<<std::endl;
  
  cudaFree(d_vptrs);
  cudaFree(d_vjs);
  cudaFree(d_neighbor);
  cudaFree(d_current);
  cudaFree(d_visited);
  cudaFree(d_dist);
  cudaFree(d_cc);
  cudaFree(d_continue);
  return 0;
}


//explicit instanciation

template int cc_gpu_vertex_multi_spmv<int, int> (int* h_vptrs, int* h_vjs, int n, int nb, float *h_cc);
template int cc_gpu_vertex_multi_spmv<int, long int> (long int* h_vptrs, int* h_vjs, int n, int nb, float *h_cc);
template int cc_gpu_vertex_multi_spmv<long int, long int> (long int* h_vptrs, long int* h_vjs, long int n, long int nb, float *h_cc);
