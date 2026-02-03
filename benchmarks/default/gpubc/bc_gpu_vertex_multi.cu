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
#include <assert.h>
#include "cuda_common.h"
#include <limits>

using namespace std;

#define NB_BFS 8

/**
 * forward phase of the betweenness computation.
 * @param d_ptrs array of start of the indices of the columns in the d_js array (size V+1)
 * @param d_js array that give the columns id of the edges of the graph (size E (number of oriented edge)
 * @param d_d array that give the distance to the root of the BFS (size V*NB_BFS)
 * @param d_sigma array that give \sigma values of betweenness centrality: number of path to v (size V*NB_BFS)
 * @param d_continue [output] tells whether the front evolved
 * @param d_dist [input] current bfs level
 * @param n_count nbvertex
 */
template <typename VtxType, typename EdgeIndex>
__global__ void forward_vertex_multi (EdgeIndex *d_ptrs, VtxType *d_js, VtxType *d_d, int *d_sigma, bool *d_continue, VtxType *d_dist, VtxType n_count) {
  VtxType tid = ((VtxType)blockIdx.x) * blockDim.x * gridDim.y + ((VtxType)blockIdx.y) * blockDim.x + threadIdx.x;
	VtxType u = tid / NB_BFS;
	int bfs = tid % NB_BFS;
	if(u < n_count){
		/* for each edge (u, w) s.t. u is unvisited, w is in the current level */
		if(d_d[u*NB_BFS+bfs] == *d_dist) {
			EdgeIndex end = d_ptrs[u + 1];
			for(EdgeIndex p = d_ptrs[u]; p < end; p++) {
				VtxType w = d_js[p];
				if(d_d[w*NB_BFS+bfs] == -1) {
					d_d[w*NB_BFS+bfs] = *d_dist + 1;
					*d_continue = true;
				}
				if(d_d[w*NB_BFS+bfs] == *d_dist + 1) {
					atomicAdd(&d_sigma[w*NB_BFS+bfs], d_sigma[u*NB_BFS+bfs]);
				}
			}
		}
	}
}

/**
 * backward phase of the betweenness computation.
 * @param d_ptrs array of start of the indices of the columns in the d_js array (size V+1)
 * @param d_js array that give the columns id of the edges of the graph (size E (number of oriented edge)
 * @param d_d array that give the distance to the root of the BFS (size V*NB_BFS)
 * @param d_sigma array that give \sigma values of betweenness centrality: number of path to v (size V*NB_BFS)
 * @param d_delta array that give \delta values of betweenness centrality: contribution of paths from source to the BC of v (size V*NB_BFS)
 * @param d_bc array that partial value betweenness centrality (size V)
 * @param d_dist [input] current bfs level
 * @param n_count nbvertex
 */
template <typename VtxType, typename EdgeIndex>
__global__ void backward_vertex_multi (EdgeIndex *d_ptrs, VtxType* d_js, VtxType *d_d, int *d_sigma, float *d_delta, float* d_bc, VtxType *d_dist, VtxType n_count) {
  VtxType tid = ((VtxType)blockIdx.x) * blockDim.x * gridDim.y + ((VtxType)blockIdx.y) * blockDim.x + threadIdx.x;
        VtxType u = tid / NB_BFS;
        int bfs = tid % NB_BFS;
	if(u < n_count) {
		if(d_d[u*NB_BFS+bfs] == *d_dist - 1) {
			EdgeIndex end = d_ptrs[u + 1];
			float sum = 0;
			for(EdgeIndex p = d_ptrs[u]; p < end; p++) {
				VtxType w = d_js[p];
				if(d_d[w*NB_BFS+bfs] == *d_dist) {
					sum += 1.0f*d_sigma[u*NB_BFS+bfs]/d_sigma[w*NB_BFS+bfs]*(1.0f+d_delta[w*NB_BFS+bfs]);
				}
			}
			d_delta[u*NB_BFS+bfs] += sum;
		}
	}
}

/**
 * aggregate the delta values into bc
 * @param d_d array that give the distance to the root of the BFS (size V*NB_BFS)
 * @param d_delta array that give \delta values of betweenness centrality: contribution of paths from source to the BC of v (size V*NB_BFS)
 * @param d_bc array that partial value betweenness centrality (size V)
 * @param n_count nbvertex
 */
template <typename VtxType>
__global__ void backsum_vertex_multi (VtxType *d_d, float *d_delta, float *d_bc, VtxType n_count) {
  VtxType tid = ((VtxType)blockIdx.x) * blockDim.x * gridDim.y + ((VtxType)blockIdx.y) * blockDim.x + threadIdx.x;
	VtxType i = tid;
	if(i < n_count) {
		for (int bfs = 0; bfs < NB_BFS; ++bfs)
		  if ( d_d[i*NB_BFS+bfs] > 0 ) //this handle both unreachable vertices and source vertex
				d_bc[i] += d_delta[i*NB_BFS+bfs];
	}
}

/**
 * Initialize data for future execution of a BC iteration
 * @param s sources of the BFS
 * @param d_d array that give the distance to the root of the BFS (size V*NB_BFS)
 * @param d_sigma array that give \sigma values of betweenness centrality: number of path to v (size V*NB_BFS)
 * @param n_count nbvertex
 */
template <typename VtxType>
__global__ void init_vertex_multi (VtxType* s, VtxType *d_d, int *d_sigma, VtxType n_count){
  VtxType tid = ((VtxType)blockIdx.x) * blockDim.x * gridDim.y + ((VtxType)blockIdx.y) * blockDim.x + threadIdx.x;
	VtxType i = tid / NB_BFS;
	int bfs = tid%NB_BFS;
	if(i < n_count) {
		d_d[i*NB_BFS+bfs] = -1;
		d_sigma[i*NB_BFS+bfs] = 0;
		if(s[bfs] == i) {
			d_d[i*NB_BFS+bfs] = 0;
			d_sigma[i*NB_BFS+bfs] = 1;
		}
	}
}

/**
 * set an int on the GPU to a given value
 */

template <typename VtxType>
__global__ void set_int_vertex_m (VtxType* dest, VtxType val) { 
 	*dest = val; 
 }

template <typename VtxType, typename EdgeIndex>
int bc_gpu_vertex_multi (EdgeIndex *h_ptrs, VtxType* h_js, VtxType n_count, EdgeIndex e_count, VtxType nb, float *h_bc) {
  int *d_sigma;
  EdgeIndex *d_ptrs;
  VtxType*d_js, *d_d,  *d_dist, h_dist;
	float *d_delta, *d_bc;
	bool h_continue, *d_continue;

	cudaMalloc((void **)&d_ptrs, sizeof(EdgeIndex) * (n_count + 1));
	CudaCheckError();
	cudaMalloc((void **)&d_js, sizeof(VtxType) * e_count);
	CudaCheckError();

	cudaMemcpy(d_ptrs, h_ptrs, sizeof(EdgeIndex) * (n_count+1), cudaMemcpyHostToDevice); // xadj array
	CudaCheckError();
	cudaMemcpy(d_js, h_js, sizeof(VtxType) * e_count, cudaMemcpyHostToDevice); // adj array
	CudaCheckError();

	cudaMalloc((void **)&d_d, sizeof(VtxType) * n_count * NB_BFS);
	CudaCheckError();

	cudaMalloc((void **)&d_sigma, sizeof(int) * n_count * NB_BFS);
	CudaCheckError();
	cudaMalloc((void **)&d_delta, sizeof(float) * n_count * NB_BFS);
	CudaCheckError();
	cudaMalloc((void **)&d_dist, sizeof(VtxType));
	CudaCheckError();

	cudaMalloc((void **)&d_bc, sizeof(float) * n_count);
	CudaCheckError();
	cudaMemset(d_bc, 0, sizeof(float) * n_count);
	CudaCheckError();

	cudaMalloc((void **)&d_continue, sizeof(bool));
	CudaCheckError();

	VtxType* d_sources;
	cudaMalloc((void **)&d_sources, sizeof(VtxType)*NB_BFS);
	CudaCheckError();

	//for operations with V * NB_BFS threads
	VtxType threads_per_block = n_count * NB_BFS;
	assert (n_count < std::numeric_limits<VtxType>::max()/NB_BFS); //otherwise block thread size will go negative, and we will like OOM the card

	VtxType blocks = 1;
	if(threads_per_block > MTS){
		blocks = (VtxType)ceil(threads_per_block/(double)MTS);
		threads_per_block = MTS;
		blocks = (VtxType)ceil(sqrt((float)blocks)); //using 2d grid
	}
	dim3 grid;
	grid.x = blocks;
	grid.y = blocks;

	dim3 threads(threads_per_block);

	//for operations with V threads
	VtxType threads_per_block_reg = n_count;
	VtxType blocks_reg = 1;
	if(threads_per_block_reg > MTS){
		blocks_reg = (VtxType)ceil(threads_per_block_reg/(double)MTS);
		threads_per_block_reg = MTS;
	}

	dim3 grid_reg(blocks_reg);
	dim3 threads_reg(threads_per_block_reg);



#ifdef TIMER
	struct timeval t1, t2, gt1, gt2; double time;
#endif

	assert (min (nb, n_count)%NB_BFS == 0);
	
	for (VtxType i = 0; i < min (nb, n_count); i+=NB_BFS) {
#ifdef TIMER
		gettimeofday(&t1, 0);
#endif
		for (int x = 0; x<NB_BFS; ++x) {
		  set_int_vertex_m<<<1,1>>>(d_sources+x, i+x);
		  CudaCheckError();
		}
				

		h_dist = 0;
		set_int_vertex_m<<<1,1>>>(d_dist, h_dist);
		CudaCheckError();
		init_vertex_multi<<<grid,threads>>>(d_sources, d_d, d_sigma, n_count);
		CudaCheckError();

#ifdef TIMER
		gettimeofday(&t2, 0);
		time = (1000000.0*(t2.tv_sec-t1.tv_sec) + t2.tv_usec-t1.tv_usec)/1000000.0;
		cout << "initialization takes " << time << " secs\n";
		gettimeofday(&gt1, 0);
#endif

		// BFS
		do {
#ifdef TIMER
			gettimeofday(&t1, 0);
#endif

			cudaMemset(d_continue, 0, sizeof(bool));
			CudaCheckError();

			forward_vertex_multi<<<grid,threads>>>(d_ptrs, d_js, d_d, d_sigma, d_continue, d_dist, n_count);
			set_int_vertex_m<<<1,1>>>(d_dist, ++h_dist);
			cudaMemcpy(&h_continue, d_continue, sizeof(bool), cudaMemcpyDeviceToHost);
	

#ifdef TIMER
			cudaDeviceSynchronize();

			gettimeofday(&t2, 0);
			time = (1000000.0*(t2.tv_sec-t1.tv_sec) + t2.tv_usec-t1.tv_usec)/1000000.0;
			cout << "level " <<  h_dist << " takes " << time << " secs\n";
#endif
		} while (h_continue);

#ifdef TIMER
		gettimeofday(&gt2, 0);
		time = (1000000.0*(gt2.tv_sec-gt1.tv_sec) + gt2.tv_usec-gt1.tv_usec)/1000000.0;
		cout << "Phase 1 takes " << time << " secs\n";
		gettimeofday(&gt1, 0); // starts back propagation
#endif

		//Back propagation
		cudaMemset(d_delta, 0, sizeof(float) * n_count * NB_BFS);
		CudaCheckError();
		
		set_int_vertex_m<<<1,1>>>(d_dist, --h_dist);
		while (h_dist > 1) {
			backward_vertex_multi<<<grid, threads>>>(d_ptrs, d_js, d_d, d_sigma, d_delta, d_bc, d_dist, n_count);
			set_int_vertex_m<<<1,1>>>(d_dist, --h_dist);
		}
		backsum_vertex_multi<<<grid_reg, threads_reg>>>(d_d,  d_delta, d_bc, n_count);

#ifdef TIMER
                cudaDeviceSynchronize();
		gettimeofday(&gt2, 0);
		time = (1000000.0*(gt2.tv_sec-gt1.tv_sec) + gt2.tv_usec-gt1.tv_usec)/1000000.0;
		cout << "Phase 2 takes " << time << " secs\n";
#endif

	}

	cudaMemcpy(h_bc, d_bc, sizeof(float)*n_count, cudaMemcpyDeviceToHost);
	CudaCheckError();

	cudaFree(d_ptrs);
	cudaFree(d_js);
	cudaFree(d_d);
	cudaFree(d_sigma);
	cudaFree(d_delta);
	cudaFree(d_dist);
	cudaFree(d_bc);
	cudaFree(d_continue);

	return 0;
}

//explicit instanciation
template int bc_gpu_vertex_multi<int, int> (int *h_ptrs, int* h_js, int n_count, int e_count, int nb, float *h_bc);
template int bc_gpu_vertex_multi<int, long int> (long int *h_ptrs, int* h_js, int n_count, long int e_count, int nb, float *h_bc);
template int bc_gpu_vertex_multi<long int, long int> (long int *h_ptrs, long int* h_js, long int n_count, long int e_count, long int nb, float *h_bc);
