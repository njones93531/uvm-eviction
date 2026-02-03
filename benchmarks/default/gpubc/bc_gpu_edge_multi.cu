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
#include "cuda_common.h"
#include <assert.h>
#include <limits>

#define NB_BFS 8

using namespace std;

/**
 * Forward BFS phase
 * @param d_v array that gives the start of the edges (size E)
 * @param d_e array that gives the end of the edges (size E)
 * @param d_d array that gives the distance of vertices in each bfs (size V*NB_BFS)
 * @param d_sigma array that gives the \sigma value of betweenness centrality for each vertex in each bfs (size V*NB_BFS)
 * @param d_continue [output] tells whether the front evolved
 * @param d_dist [input] current bfs level
 * @param e_count number of edge (E)
 */
template <typename VtxType, typename EdgeIndex>
__global__ void forward_edge_multi (VtxType *d_v, VtxType *d_e, VtxType *d_d, int *d_sigma, bool *d_continue, VtxType *d_dist, EdgeIndex e_count) {
	EdgeIndex tid = blockIdx.x * blockDim.x * gridDim.y + blockIdx.y * blockDim.x + threadIdx.x;
	EdgeIndex edge = tid / NB_BFS;
	int bfs = tid % NB_BFS;
	if(edge < e_count) {
		/* for each edge (u, w) */
		VtxType u = d_v[edge];
		if(d_d[u*NB_BFS+bfs] == *d_dist) {
			VtxType w = d_e[edge];
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

/**
 * backward BFS phase
 * @param d_v array that gives the start of the edges (size E)
 * @param d_e array that gives the end of the edges (size E)
 * @param d_d array that gives the distance of vertices in each bfs (size V*NB_BFS)
 * @param d_sigma array that gives the \sigma value of betweenness centrality for each vertex in each bfs (size V*NB_BFS)
 * @param d_delta array that gives the \delta value of betweenness centrality for each vertex in each bfs (size V*NB_BFS)
 * @param d_dist [input] current bfs level
 * @param e_count number of edge (E)
 */
template <typename VtxType, typename EdgeIndex>
__global__ void backward_edge_multi (VtxType *d_v, VtxType *d_e, VtxType *d_d, int *d_sigma, float *d_delta, VtxType *d_dist, EdgeIndex e_count) {

  EdgeIndex tid = ((EdgeIndex)blockIdx.x) * blockDim.x * gridDim.y + ((EdgeIndex)blockIdx.y) * blockDim.x + threadIdx.x;
	EdgeIndex edge = tid / NB_BFS;
        int bfs = tid % NB_BFS;

	if(edge < e_count) {
		VtxType u = d_v[edge];
		if(d_d[u*NB_BFS+bfs] == *d_dist - 1) {
			VtxType w = d_e[edge];
			if(d_d[w*NB_BFS+bfs] == *d_dist) {
				atomicAdd(&d_delta[u*NB_BFS+bfs], 1.0f*d_sigma[u*NB_BFS+bfs]/d_sigma[w*NB_BFS+bfs]*(1.0f+d_delta[w*NB_BFS+bfs]));
			}
		}
	}
}

/**
 * aggregating the delta values into the actual betweenness centrality
 * @param s array that gives the sources of the bfs (size NB_BFS)
 * @param d_d array that gives the distance of vertices in each bfs (size V*NB_BFS)
 * @param d_delta array that gives the \delta value of betweenness centrality for each vertex in each bfs (size V*NB_BFS)
 * @param d_bc array that gives the BC value for each vertex (size V)
 * @param n_count number of vertices (V)
 */
template <typename VtxType>
__global__ void backsum_edge_multi (VtxType* s, VtxType *d_d, float *d_delta, float *d_bc, VtxType n_count) {

  VtxType vertex =  ((VtxType)blockIdx.x) * blockDim.x * gridDim.y + ((VtxType)blockIdx.y) * blockDim.x + threadIdx.x;
	if(vertex < n_count) {
	  for (int bfs = 0; bfs < NB_BFS; ++bfs) {
	    if (vertex != s[bfs] && d_d[vertex*NB_BFS+bfs] != -1) {
	      d_bc[vertex] += d_delta[vertex*NB_BFS+bfs];
	    }
	  }
	}
}

/**
 * aggregating the delta values into the actual betweenness centrality
 * @param s array that gives the sources of the bfs (size NB_BFS)
 * @param d_d array that gives the distance of vertices in each bfs (size V*NB_BFS)
 * @param d_sigma array that gives the \sigma value of betweenness centrality for each vertex in each bfs (size V*NB_BFS)
 * @param n_count number of vertices (V)
 */
template <typename VtxType>
__global__ void init_edge_multi (VtxType* s, VtxType *d_d, int *d_sigma, VtxType n_count) {
  VtxType tid =  ((VtxType)blockIdx.x) * blockDim.x * gridDim.y + ((VtxType)blockIdx.y) * blockDim.x + threadIdx.x;
  VtxType vertex = tid/NB_BFS;
	int bfs = tid%NB_BFS;
	if(vertex < n_count) {
		d_d[vertex*NB_BFS+bfs] = -1;
		d_sigma[vertex*NB_BFS+bfs] = 0;
		if(s[bfs] == vertex) {
			d_d[vertex*NB_BFS+bfs] = 0;
			d_sigma[vertex*NB_BFS+bfs] = 1;
		}
	}
}

template <typename VtxType>
__global__ void set_int_edge_multi (VtxType* dest, VtxType val) {
	*dest = val;
}

template <typename VtxType, typename EdgeIndex>
int bc_gpu_edge_multi (VtxType* h_v, VtxType *h_e, VtxType n_count, EdgeIndex e_count, VtxType nb, float *h_bc) {
  int *d_sigma;
	VtxType *d_v, *d_e, *d_d,  *d_dist, h_dist;
	float *d_delta, *d_bc;
	bool h_continue, *d_continue;

	assert (cudaSuccess == cudaMalloc((void **)&d_v, sizeof(VtxType)*e_count));
	assert (cudaSuccess == cudaMalloc((void **)&d_e, sizeof(VtxType)*e_count));

	assert (cudaSuccess == cudaMemcpy(d_v, h_v, sizeof(VtxType)*e_count, cudaMemcpyHostToDevice));
	assert (cudaSuccess == cudaMemcpy(d_e, h_e, sizeof(VtxType)*e_count, cudaMemcpyHostToDevice));

	assert (cudaSuccess == cudaMalloc((void **)&d_d, sizeof(VtxType)*n_count*NB_BFS));

	assert (cudaSuccess == cudaMalloc((void **)&d_sigma, sizeof(int)*n_count*NB_BFS));
	assert (cudaSuccess == cudaMalloc((void **)&d_delta, sizeof(float)*n_count*NB_BFS));
	assert (cudaSuccess == cudaMalloc((void **)&d_dist, sizeof(VtxType)));

	assert (cudaSuccess == cudaMalloc((void **)&d_bc, sizeof(float)*n_count));
	assert (cudaSuccess == cudaMemset(d_bc, 0, sizeof(float)*n_count));

	assert (cudaSuccess == cudaMalloc((void **)&d_continue, sizeof(bool)));


	VtxType* d_sources;
	cudaMalloc((void **)&d_sources, sizeof(VtxType)*NB_BFS);
	CudaCheckError();

	assert (e_count < std::numeric_limits<EdgeIndex>::max()/NB_BFS); //otherwise block thread size will go negative
	assert (n_count < std::numeric_limits<VtxType>::max()/NB_BFS); //otherwise block thread size will go negative

	//This one is for E*NB_BFS threads
	EdgeIndex threads_per_block = e_count*NB_BFS;
	EdgeIndex blocks = 1;
	if(threads_per_block > MTS) {
		blocks = (EdgeIndex)ceil(threads_per_block/(float)MTS);
		blocks = (EdgeIndex)ceil(sqrt((float)blocks));
		threads_per_block = MTS;
	}
	dim3 grid;
	grid.x = blocks;
	grid.y = blocks;
	dim3 threads(threads_per_block);

	//This one is for V*NB_BFS threads
	VtxType threads_per_block2=n_count * NB_BFS;
	VtxType blocks2 = 1;
	if(threads_per_block2 > MTS) {
		blocks2 = (VtxType)ceil(threads_per_block2/(double)MTS);
		blocks2 = (VtxType)ceil(sqrt((float)blocks2));
		threads_per_block2 = MTS;
	}
	dim3 grid2;
	grid2.x = blocks2;
	grid2.y = blocks2;
	dim3 threads2(threads_per_block2);


	//This one is for V threads
	VtxType threads_per_block3=n_count ;
	VtxType blocks3 = 1;
	if(threads_per_block3 > MTS) {
		blocks3 = (VtxType)ceil(threads_per_block3/(double)MTS);
		blocks3 = (VtxType)ceil(sqrt((float)blocks3));
		threads_per_block3 = MTS;
	}
	dim3 grid3;
	grid3.x = blocks3;
	grid3.y = blocks3;
	dim3 threads3(threads_per_block3);


#ifdef TIMER
	struct timeval t1, t2, gt1, gt2; double time;
#endif

	std::cout<<"cuda parameters: "
		 <<blocks<<" "<<threads_per_block<<" "
		 <<blocks2<<" "<<threads_per_block2<<" "
		 <<blocks3<<" "<<threads_per_block3<<std::endl;
	assert (min (nb, n_count)%NB_BFS == 0);

	for(VtxType i = 0; i < min(nb, n_count); i+=NB_BFS){
#ifdef TIMER
		gettimeofday(&t1, 0);
#endif

		for (int x = 0; x<NB_BFS; ++x) {
		  set_int_edge_multi<<<1,1>>>(d_sources+x, i+x);
		  CudaCheckError();
		}

		h_dist = 0;
		init_edge_multi <<<grid2,threads2>>>(d_sources, d_d, d_sigma, n_count);
		CudaCheckError();
		set_int_edge_multi <<<1,1>>>(d_dist, (VtxType)0);
		CudaCheckError();
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

			assert (cudaSuccess == cudaMemset(d_continue, 0, sizeof(bool)));
			forward_edge_multi <<<grid,threads>>>(d_v, d_e, d_d, d_sigma, d_continue, d_dist, e_count);
			CudaCheckError();
			set_int_edge_multi <<<1,1>>>(d_dist, ++h_dist);
			CudaCheckError();
			assert (cudaSuccess == cudaMemcpy(&h_continue, d_continue, sizeof(bool), cudaMemcpyDeviceToHost));

#ifdef TIMER
			cudaDeviceSynchronize();
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
		assert (cudaSuccess == cudaMemset(d_delta, 0, sizeof(float) * n_count * NB_BFS));
		set_int_edge_multi <<<1,1>>>(d_dist, --h_dist);
		CudaCheckError();
		while(h_dist > 1) {
			backward_edge_multi <<<grid, threads>>>(d_v, d_e, d_d, d_sigma, d_delta, d_dist, e_count);
			set_int_edge_multi <<<1,1>>>(d_dist, --h_dist);
			CudaCheckError();
		}
		backsum_edge_multi <<<grid3, threads3>>>(d_sources, d_d,  d_delta, d_bc, n_count);
		CudaCheckError();

#ifdef TIMER
		cudaDeviceSynchronize();
		gettimeofday(&gt2, 0);
		time = (1000000.0*(gt2.tv_sec-gt1.tv_sec) + gt2.tv_usec-gt1.tv_usec)/1000000.0;
		cout << "Phase 2 takes " << time << " secs\n";
#endif
	}

	assert (cudaSuccess == cudaMemcpy(h_bc, d_bc, sizeof(float)*n_count, cudaMemcpyDeviceToHost));
	cudaFree(d_v);
	cudaFree(d_e);
	cudaFree(d_d);
	cudaFree(d_sigma);
	cudaFree(d_delta);
	cudaFree(d_dist);
	cudaFree(d_bc);
	cudaFree(d_continue);
	return 0;
}


//explicit instanciation
template int bc_gpu_edge_multi<int, int> (int* h_v, int *h_e, int n_count, int e_count, int nb, float *h_bc);
template int bc_gpu_edge_multi<int, long int> (int* h_v, int *h_e, int n_count, long int e_count, int nb, float *h_bc);
template int bc_gpu_edge_multi<long int, long int> (long int* h_v, long int *h_e, long int n_count, long int e_count, long int nb, float *h_bc);
