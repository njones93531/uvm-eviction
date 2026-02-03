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

using namespace std;

/**
 * forward phase of the betweenness computation.
 * @param d_ptrs array of start of the indices of the columns in the d_js array (size V+1)
 * @param d_js array that give the columns id of the edges of the graph (size E (number of oriented edge)
 * @param d_d array that give the distance to the root of the BFS (size V)
 * @param d_sigma array that give \sigma values of betweenness centrality: number of path to v (size V)
 * @param d_continue [output] tells whether the front evolved
 * @param d_dist [input] current bfs level
 * @param n_count nbvertex
 */
template <typename VtxType, typename EdgeIndex>
__global__ void forward_vertex (EdgeIndex *d_ptrs, VtxType *d_js, VtxType *d_d, int *d_sigma, bool *d_continue, VtxType *d_dist, int n_count) {
  VtxType u = ((VtxType)blockIdx.x) * blockDim.x + threadIdx.x;
	if(u < n_count){
		/* for each edge (u, w) s.t. u is unvisited, w is in the current level */
		if(d_d[u] == *d_dist) {
			EdgeIndex end = d_ptrs[u + 1];
			for(EdgeIndex p = d_ptrs[u]; p < end; p++) {
				VtxType w = d_js[p];
				if(d_d[w] == -1) {
					d_d[w] = *d_dist + 1;
					*d_continue = true;
				}
				if(d_d[w] == *d_dist + 1) {
					atomicAdd(&d_sigma[w], d_sigma[u]);
				}
			}
		}
	}
}

/**
 * backward phase of the betweenness computation.
 * @param d_ptrs array of start of the indices of the columns in the d_js array (size V+1)
 * @param d_js array that give the columns id of the edges of the graph (size E (number of oriented edge)
 * @param d_d array that give the distance to the root of the BFS (size V)
 * @param d_sigma array that give \sigma values of betweenness centrality: number of path to v (size V)
 * @param d_delta array that give \delta values of betweenness centrality: contribution of paths from source to the BC of v (size V)
 * @param d_bc array that partial value betweenness centrality (size V)
 * @param d_dist [input] current bfs level
 * @param n_count nbvertex
 */
template <typename VtxType, typename EdgeIndex>
__global__ void backward_vertex (EdgeIndex *d_ptrs, VtxType* d_js, VtxType *d_d, int *d_sigma, float *d_delta, float* d_bc, VtxType *d_dist, int n_count) {
  VtxType u = ((VtxType)blockIdx.x) * blockDim.x + threadIdx.x;
	if(u < n_count) {
		if(d_d[u] == *d_dist - 1) {
			EdgeIndex end = d_ptrs[u + 1];
			float sum = 0;
			for(EdgeIndex p = d_ptrs[u]; p < end; p++) {
				VtxType w = d_js[p];
				if(d_d[w] == *d_dist) {
					sum += 1.0f*d_sigma[u]/d_sigma[w]*(1.0f+d_delta[w]);
				}
			}
			d_delta[u] += sum;
		}
	}
}

/**
 * aggregate the delta values into bc
 * @param s source of the BFS
 * @param d_d array that give the distance to the root of the BFS (size V)
 * @param d_delta array that give \delta values of betweenness centrality: contribution of paths from source to the BC of v (size V)
 * @param d_bc array that partial value betweenness centrality (size V)
 * @param n_count nbvertex
 */
template <typename VtxType>
__global__ void backsum_vertex (VtxType s, VtxType *d_d, float *d_delta, float *d_bc, VtxType n_count) {
  VtxType tid = ((VtxType)blockIdx.x) * blockDim.x + threadIdx.x;
	if(tid < n_count && tid != s && d_d[tid] != -1) {
		d_bc[tid] += d_delta[tid];
	}
}

template <typename VtxType>
__global__ void backsum_vertex_deg1 (VtxType s, VtxType *d_d, float *d_delta, float *d_bc, VtxType n_count, VtxType* d_weight) {
  VtxType tid = ((VtxType)blockIdx.x) * blockDim.x + threadIdx.x;
	if(tid < n_count && tid != s && d_d[tid] != -1) {
		d_bc[tid] += d_delta[tid] * d_weight[s];
	}
}

/**
 * Initialize data for future execution of a BC iteration
 * @param s source of the BFS
 * @param d_d array that give the distance to the root of the BFS (size V)
 * @param d_sigma array that give \sigma values of betweenness centrality: number of path to v (size V)
 * @param n_count nbvertex
 * @param d_dist [input] current bfs level
 */
template <typename VtxType>
__global__ void init_vertex (VtxType s, VtxType *d_d, int *d_sigma, int n_count, VtxType* d_dist){
  VtxType i = ((VtxType)blockIdx.x) * blockDim.x + threadIdx.x;
	if(i < n_count) {
		d_d[i] = -1;
		d_sigma[i] = 0;
		if(s == i) {
			d_d[i] = 0;
			d_sigma[i] = 1;
			*d_dist = 0;
		}
	}
}

/**
 * set an int on the GPU to a given value
 */
template <typename VtxType>
__global__ void set_int_vertex (VtxType* dest, VtxType val){
	*dest = val;
}

template <typename VtxType>
__global__ void init_delta (VtxType *d_weight, float* d_delta, VtxType n_count) {
  VtxType i = ((VtxType)blockIdx.x)*blockDim.x + threadIdx.x;
	if(i < n_count) {
		d_delta[i] = d_weight[i]-1;
	}
}


template <typename VtxType, typename EdgeIndex>
int bc_gpu_vertex (EdgeIndex *h_ptrs, VtxType* h_js, VtxType n_count, EdgeIndex e_count, VtxType nb, float *h_bc) {

  EdgeIndex *d_ptrs;
  VtxType *d_js, *d_d,  *d_dist, h_dist;
	int *d_sigma;
	float *d_delta, *d_bc;
	bool h_continue, *d_continue;

	cudaMalloc((void **)&d_ptrs, sizeof(EdgeIndex) * (n_count + 1));
	cudaMalloc((void **)&d_js, sizeof(VtxType) * e_count);

	cudaMemcpy(d_ptrs, h_ptrs, sizeof(EdgeIndex) * (n_count+1), cudaMemcpyHostToDevice); // xadj array
	cudaMemcpy(d_js, h_js, sizeof(VtxType) * e_count, cudaMemcpyHostToDevice); // adj array

	cudaMalloc((void **)&d_d, sizeof(VtxType) * n_count);

	cudaMalloc((void **)&d_sigma, sizeof(int) * n_count);
	cudaMalloc((void **)&d_delta, sizeof(float) * n_count);
	cudaMalloc((void **)&d_dist, sizeof(VtxType));

	cudaMalloc((void **)&d_bc, sizeof(float) * n_count);
	cudaMemset(d_bc, 0, sizeof(float) * n_count);

	cudaMalloc((void **)&d_continue, sizeof(bool));

	VtxType threads_per_block = n_count;
	VtxType blocks = 1;
	if(n_count > MTS){
		blocks = (VtxType)ceil(n_count/(double)MTS);
		threads_per_block = MTS;
	}

	dim3 grid(blocks);
	dim3 threads(threads_per_block);


#ifdef TIMER
	struct timeval t1, t2, gt1, gt2; double time;
#endif

	for(VtxType i = 0; i < min (nb, n_count); i++) {
#ifdef TIMER
		gettimeofday(&t1, 0);
#endif

		h_dist = 0;
		init_vertex<<<grid,threads>>>(i, d_d, d_sigma, n_count, d_dist);

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
			forward_vertex<<<grid,threads>>>(d_ptrs, d_js, d_d, d_sigma, d_continue, d_dist, n_count);
			set_int_vertex<<<1,1>>>(d_dist, ++h_dist);
			cudaMemcpy(&h_continue, d_continue, sizeof(bool), cudaMemcpyDeviceToHost);

#ifdef TIMER
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
		cudaMemset(d_delta, 0, sizeof(float) * n_count);
		set_int_vertex<<<1,1>>>(d_dist, --h_dist);
		while (h_dist > 1) {
			backward_vertex<<<grid, threads>>>(d_ptrs, d_js, d_d, d_sigma, d_delta, d_bc, d_dist, n_count);
			set_int_vertex<<<1,1>>>(d_dist, --h_dist);
		}
		backsum_vertex<<<grid, threads>>>(i, d_d,  d_delta, d_bc, n_count);

#ifdef TIMER
		gettimeofday(&gt2, 0);
		time = (1000000.0*(gt2.tv_sec-gt1.tv_sec) + gt2.tv_usec-gt1.tv_usec)/1000000.0;
		cout << "Phase 2 takes " << time << " secs\n";
#endif

	}

	cudaMemcpy(h_bc, d_bc, sizeof(float)*n_count, cudaMemcpyDeviceToHost);
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

template <typename VtxType, typename EdgeIndex>
int bc_gpu_vertex_deg1 (EdgeIndex *h_ptrs, VtxType* h_js, VtxType n_count, EdgeIndex e_count, VtxType nb, float *h_bc, VtxType* h_weight) {

  EdgeIndex *d_ptrs;
  VtxType *d_js, *d_d, *d_dist, h_dist, *d_weight;
  int *d_sigma;
	float *d_delta, *d_bc;
	bool h_continue, *d_continue;

	cudaMalloc((void **)&d_ptrs, sizeof(EdgeIndex) * (n_count + 1));
	cudaMalloc((void **)&d_js, sizeof(VtxType) * e_count);

	cudaMemcpy(d_ptrs, h_ptrs, sizeof(EdgeIndex) * (n_count+1), cudaMemcpyHostToDevice); // xadj array
	cudaMemcpy(d_js, h_js, sizeof(VtxType) * e_count, cudaMemcpyHostToDevice); // adj array

	cudaMalloc((void **)&d_d, sizeof(VtxType) * n_count);

	cudaMalloc((void **)&d_sigma, sizeof(int) * n_count);
	cudaMalloc((void **)&d_delta, sizeof(float) * n_count);
	cudaMalloc((void **)&d_weight, sizeof(VtxType) * n_count);
	cudaMemcpy(d_weight, h_weight, sizeof(VtxType) * n_count, cudaMemcpyHostToDevice); // weight array
	cudaMalloc((void **)&d_dist, sizeof(VtxType));

	cudaMalloc((void **)&d_bc, sizeof(float) * n_count);
	cudaMemcpy(d_bc, h_bc, sizeof(float) * n_count, cudaMemcpyHostToDevice); // bc array

	cudaMalloc((void **)&d_continue, sizeof(bool));

	VtxType threads_per_block = n_count;
	VtxType blocks = 1;
	if(n_count > MTS){
		blocks = (VtxType)ceil(n_count/(double)MTS);
		threads_per_block = MTS;
	}

	dim3 grid(blocks);
	dim3 threads(threads_per_block);


#ifdef TIMER
	struct timeval t1, t2, gt1, gt2; double time;
#endif

	for(VtxType i = 0; i < min (nb, n_count); i++){
#ifdef TIMER
		gettimeofday(&t1, 0);
#endif

		h_dist = 0;
		init_vertex<<<grid,threads>>>(i, d_d, d_sigma, n_count, d_dist);

#ifdef TIMER
		gettimeofday(&t2, 0);
		time = (1000000.0*(t2.tv_sec-t1.tv_sec) + t2.tv_usec-t1.tv_usec)/1000000.0;
		cout << "initialization takes " << time << " secs\n";
		gettimeofday(&gt1, 0);
#endif

		// BFS
		do{
#ifdef TIMER
			gettimeofday(&t1, 0);
#endif

			cudaMemset(d_continue, 0, sizeof(bool));
			forward_vertex<<<grid,threads>>>(d_ptrs, d_js, d_d, d_sigma, d_continue, d_dist, n_count);
			set_int_vertex<<<1,1>>>(d_dist, ++h_dist);
			cudaMemcpy(&h_continue, d_continue, sizeof(bool), cudaMemcpyDeviceToHost);

#ifdef TIMER
			gettimeofday(&t2, 0);
			time = (1000000.0*(t2.tv_sec-t1.tv_sec) + t2.tv_usec-t1.tv_usec)/1000000.0;
			cout << "level " <<  h_dist << " takes " << time << " secs\n";
#endif

		} while(h_continue);

#ifdef TIMER
		gettimeofday(&gt2, 0);
		time = (1000000.0*(gt2.tv_sec-gt1.tv_sec) + gt2.tv_usec-gt1.tv_usec)/1000000.0;
		cout << "Phase 1 takes " << time << " secs\n";
		gettimeofday(&gt1, 0); // starts back propagation
#endif

		//Back propagation

		init_delta<<<grid, threads>>>(d_weight, d_delta, n_count); // deltas are initialized
		set_int_vertex<<<1,1>>>(d_dist, --h_dist);
		while(h_dist > 1) {
			backward_vertex<<<grid, threads>>>(d_ptrs, d_js, d_d, d_sigma, d_delta, d_bc, d_dist, n_count);
			set_int_vertex<<<1,1>>>(d_dist, --h_dist);
		}


		backsum_vertex_deg1<<<grid, threads>>>(i, d_d,  d_delta, d_bc, n_count, d_weight);
		

#ifdef TIMER
		cudaDeviceSynchronize(); //need to sync to get accurate timings
		gettimeofday(&gt2, 0);
		time = (1000000.0*(gt2.tv_sec-gt1.tv_sec) + gt2.tv_usec-gt1.tv_usec)/1000000.0;
		cout << "Phase 2 takes " << time << " secs\n";
#endif

	}

	cudaMemcpy(h_bc, d_bc, sizeof(float)*n_count, cudaMemcpyDeviceToHost);
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

//explicit specialization for common use case

template int bc_gpu_vertex<int, int> (int *h_ptrs, int* h_js, int n_count, int e_count, int nb, float *h_bc);
template int bc_gpu_vertex<int, long int> (long int *h_ptrs, int* h_js, int n_count, long int e_count, int nb, float *h_bc);
template int bc_gpu_vertex<long int, long int> (long int *h_ptrs, long int* h_js, long int n_count, long int e_count, long int nb, float *h_bc);

template int bc_gpu_vertex_deg1<int, int> (int *h_ptrs, int* h_js, int n_count, int e_count, int nb, float *h_bc, int* h_weight);
template int bc_gpu_vertex_deg1<int, long int> (long int *h_ptrs, int* h_js, int n_count, long int e_count, int nb, float *h_bc, int* h_weight);
template int bc_gpu_vertex_deg1<long int, long int> (long int *h_ptrs, long int* h_js, long int n_count, long int e_count, long int nb, float *h_bc, long int* h_weight);
