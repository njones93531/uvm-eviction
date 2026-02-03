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

using namespace std;

template <typename VtxType, typename EdgeIndex>
__global__ void forward_edge (VtxType *d_v, VtxType *d_e, VtxType  *d_d, int *d_sigma, bool *d_continue, VtxType *d_dist, int e_count) {

  EdgeIndex tid = ((EdgeIndex)blockIdx.x) * blockDim.x * gridDim.y + ((EdgeIndex)blockIdx.y) * blockDim.x + threadIdx.x;
	if(tid < e_count) {
		/* for each edge (u, w) */
		VtxType u = d_v[tid];
		if(d_d[u] == *d_dist) {
			VtxType w = d_e[tid];
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

template <typename VtxType, typename EdgeIndex>
__global__ void backward_edge (VtxType *d_v, VtxType *d_e, VtxType *d_d, int *d_sigma, float *d_delta, VtxType *d_dist, int e_count) {

  EdgeIndex tid = ((EdgeIndex)blockIdx.x) * blockDim.x * gridDim.y + ((EdgeIndex)blockIdx.y) * blockDim.x + threadIdx.x;
	if(tid < e_count) {
		VtxType u = d_v[tid];
		if(d_d[u] == *d_dist - 1) {
			VtxType w = d_e[tid];
			if(d_d[w] == *d_dist) {
				atomicAdd(&d_delta[u], 1.0f*d_sigma[u]/d_sigma[w]*(1.0f+d_delta[w]));
			}
		}
	}
}

template <typename VtxType>
__global__ void backsum_edge (VtxType s, VtxType *d_d, float *d_delta, float *d_bc, VtxType n_count) {

  VtxType tid =  ((VtxType)blockIdx.x) * blockDim.x * gridDim.y + ((VtxType)blockIdx.y) * blockDim.x + threadIdx.x;
	if(tid < n_count && tid != s && d_d[tid] != -1) {
		d_bc[tid] += d_delta[tid];
	}
}

template <typename VtxType>
__global__ void init_edge (VtxType s, VtxType *d_d, int *d_sigma, VtxType n_count, VtxType* d_dist) {

  VtxType i =  ((VtxType)blockIdx.x) * blockDim.x * gridDim.y + ((VtxType)blockIdx.y) * blockDim.x + threadIdx.x;
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

template <typename VtxType>
__global__ void set_int_edge (VtxType* dest, VtxType val) {
	*dest = val;
}

template <typename VtxType, typename EdgeIndex>
int bc_gpu_edge (VtxType* h_v, VtxType *h_e, VtxType n_count, EdgeIndex e_count, VtxType nb, float *h_bc) {
	VtxType *d_v, *d_e, *d_d, *d_dist, h_dist;
	int *d_sigma;
	float *d_delta, *d_bc;
	bool h_continue, *d_continue;

	assert (cudaSuccess == cudaMalloc((void **)&d_v, sizeof(VtxType)*e_count));
	assert (cudaSuccess == cudaMalloc((void **)&d_e, sizeof(VtxType)*e_count));

	assert (cudaSuccess == cudaMemcpy(d_v, h_v, sizeof(VtxType)*e_count, cudaMemcpyHostToDevice));
	assert (cudaSuccess == cudaMemcpy(d_e, h_e, sizeof(VtxType)*e_count, cudaMemcpyHostToDevice));

	assert (cudaSuccess == cudaMalloc((void **)&d_d, sizeof(VtxType)*n_count));

	assert (cudaSuccess == cudaMalloc((void **)&d_sigma, sizeof(int)*n_count));
	assert (cudaSuccess == cudaMalloc((void **)&d_delta, sizeof(float)*n_count));
	assert (cudaSuccess == cudaMalloc((void **)&d_dist, sizeof(VtxType)));

	assert (cudaSuccess == cudaMalloc((void **)&d_bc, sizeof(float)*n_count));
	assert (cudaSuccess == cudaMemset(d_bc, 0, sizeof(float)*n_count));

	assert (cudaSuccess == cudaMalloc((void **)&d_continue, sizeof(bool)));

	EdgeIndex threads_per_block = e_count;
	EdgeIndex blocks = 1;
	if(e_count > MTS) {
		blocks = (EdgeIndex)ceil(e_count/(float)MTS);
		blocks = (EdgeIndex)ceil(sqrt((float)blocks));
		threads_per_block = MTS;
	}
	dim3 grid;
	grid.x = blocks;
	grid.y = blocks;
	dim3 threads(threads_per_block);
	EdgeIndex threads_per_block2=n_count;
	EdgeIndex blocks2 = 1;
	if(n_count > MTS) {
		blocks2 = (EdgeIndex)ceil(n_count/(double)MTS);
		blocks2 = (EdgeIndex)ceil(sqrt((float)blocks2));
		threads_per_block2 = MTS;
	}
	dim3 grid2;
	grid2.x = blocks2;
	grid2.y = blocks2;
	dim3 threads2(threads_per_block2);


	cout<<"cuda parameters: "<<blocks<<" "<<threads_per_block<<" "<<blocks2<<" "<<threads_per_block2<<endl;

#ifdef TIMER
	struct timeval t1, t2, gt1, gt2;
	double time;
#endif

	for(VtxType i = 0; i < min(nb, n_count); i++){
#ifdef TIMER
		gettimeofday(&t1, 0);
#endif

		h_dist = 0;
		init_edge <<<grid,threads>>>(i, d_d, d_sigma, n_count, d_dist);

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
			forward_edge<VtxType, EdgeIndex> <<<grid,threads>>>(d_v, d_e, d_d, d_sigma, d_continue, d_dist, e_count);
			set_int_edge <<<1,1>>>(d_dist, ++h_dist);
			CudaCheckError();
			assert (cudaSuccess == cudaMemcpy(&h_continue, d_continue, sizeof(bool), cudaMemcpyDeviceToHost));

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
		assert (cudaSuccess == cudaMemset(d_delta, 0, sizeof(int) * n_count));
		set_int_edge <<<1,1>>>(d_dist, --h_dist);
		while(h_dist > 1) {
		  backward_edge<VtxType, EdgeIndex> <<<grid, threads>>>(d_v, d_e, d_d, d_sigma, d_delta, d_dist, e_count);
			set_int_edge <<<1,1>>>(d_dist, --h_dist);
			CudaCheckError();
		}
		backsum_edge <<<grid2, threads2>>>(i, d_d,  d_delta, d_bc, n_count);


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


//explicit instanciation for common variant

template int bc_gpu_edge<int, int> (int* h_v, int *h_e, int n_count, int e_count, int nb, float *h_bc);

template int bc_gpu_edge<int, long int> (int* h_v, int *h_e, int n_count, long int e_count, int nb, float *h_bc);

template int bc_gpu_edge<long int, long int> (long int* h_v, long int *h_e, long int n_count, long int e_count, long int nb, float *h_bc);
