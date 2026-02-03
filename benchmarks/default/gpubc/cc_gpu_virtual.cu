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

using namespace std;

template <typename VtxType, typename EdgeIndex>
__global__ void cc_forward_virtual (VtxType* d_vmap, EdgeIndex* d_vptrs, VtxType* d_vjs, VtxType *d_d,  bool *d_continue, VtxType *d_dist, EdgeIndex virn_count) {
  EdgeIndex vu = ((EdgeIndex)blockIdx.x) * blockDim.x * gridDim.y + ((EdgeIndex)blockIdx.y) * blockDim.x + threadIdx.x;
	if(vu < virn_count) {
		VtxType u = d_vmap[vu];
		/* for each edge (u, w) s.t. u is unvisited, w is in the current level */
		if(d_d[u] == *d_dist) {
			EdgeIndex end = d_vptrs[vu + 1];
			for(EdgeIndex p = d_vptrs[vu]; p < end; p++) {
				VtxType w = d_vjs[p];
				if(d_d[w] == -1) {
					d_d[w] = *d_dist + 1;
					*d_continue = 1;
				}
			}
		}
	}
}

template <typename VtxType>
__global__ void cc_backsum_virtual (VtxType s, VtxType *d_d, float *d_cc, VtxType n_count){
  VtxType tid = ((VtxType)blockIdx.x) * blockDim.x * gridDim.y + ((VtxType)blockIdx.y) * blockDim.x + threadIdx.x;
	if(tid < n_count && tid != s && d_d[tid] != -1) {
	  //	  d_cc[tid] += 1; //incorrect.
	  VtxType di = d_d[tid];
	  if (di > 0)
	    d_cc[tid] += 1.0 / di;//symetric assumption
	}
}

template <typename VtxType>
__global__ void cc_init_virtual (VtxType s, VtxType *d_d, VtxType n_count, VtxType* d_dist){
  VtxType i = ((VtxType)blockIdx.x) * blockDim.x * gridDim.y + ((VtxType)blockIdx.y) * blockDim.x + threadIdx.x;
	if(i < n_count) {
		d_d[i] = -1;
		if(s == i) {
			d_d[i] = 0;
			*d_dist = 0;
		}
	}
}

template <typename VtxType>
__global__ void cc_set_int (VtxType* dest, VtxType val){
	*dest = val;
}

template <typename VtxType, typename EdgeIndex>
int cc_gpu_virtual (VtxType* h_vmap, EdgeIndex* h_vptrs, VtxType* h_vjs, VtxType n_count, EdgeIndex e_count, EdgeIndex virn_count, VtxType nb, float *h_cc) {
  EdgeIndex* d_vptrs;
	VtxType *d_vmap, *d_vjs, *d_d, *d_dist, h_dist;
	float *d_cc;
	bool h_continue, *d_continue;

	assert (cudaSuccess == cudaMalloc((void **)&d_vmap, sizeof(VtxType) *  virn_count));
	assert (cudaSuccess == cudaMalloc((void **)&d_vptrs, sizeof(EdgeIndex) * (virn_count + 1)));
	assert (cudaSuccess == cudaMalloc((void **)&d_vjs, sizeof(VtxType) * e_count));

	assert (cudaSuccess == cudaMemcpy(d_vmap, h_vmap, sizeof(VtxType) * virn_count, cudaMemcpyHostToDevice));
	assert (cudaSuccess == cudaMemcpy(d_vptrs, h_vptrs, sizeof(EdgeIndex) * (virn_count + 1), cudaMemcpyHostToDevice));
	assert (cudaSuccess == cudaMemcpy(d_vjs, h_vjs, sizeof(VtxType) * e_count, cudaMemcpyHostToDevice));

	assert (cudaSuccess == cudaMalloc((void **)&d_d, sizeof(VtxType)*n_count));

	assert (cudaSuccess == cudaMalloc((void **)&d_dist, sizeof(VtxType)));

	assert (cudaSuccess == cudaMalloc((void **)&d_cc, sizeof(float)*n_count));
	assert (cudaSuccess == cudaMemset(d_cc, 0, sizeof(float)*n_count));

	assert (cudaSuccess == cudaMalloc((void **)&d_continue, sizeof(bool)));

	EdgeIndex threads_per_block = virn_count;
	EdgeIndex blocks = 1;
	if(virn_count > MTS){
		blocks = (EdgeIndex)ceil(virn_count/(double)MTS);
		blocks = (EdgeIndex)ceil(sqrt((float)blocks));
		threads_per_block = MTS;
	}
	dim3 grid;
	grid.x = blocks;
	grid.y = blocks;
	dim3 threads(threads_per_block);

	VtxType threads_per_block2 = n_count;
	VtxType blocks2 = 1;
	if(n_count > MTS){
		blocks2 = (VtxType)ceil(n_count/(double)MTS);
		blocks2 = (VtxType)ceil(sqrt((float)blocks2));
		threads_per_block2 = MTS;
	}
	dim3 grid2;
	grid2.x = blocks2;
	grid2.y = blocks2;
	dim3 threads2(threads_per_block2);

	cout<<"cuda parameters: "<<blocks<<" "<<threads_per_block<<" "<<blocks2<<" "<<threads_per_block2<<endl;

#ifdef TIMER
	struct timeval t1, t2, gt1, gt2; double time;
#endif

	VtxType diameter;
	for(VtxType i = 0; i < min(nb, n_count); i++){
#ifdef TIMER
		gettimeofday(&t1, 0);
#endif

		h_dist = 0;
		cc_init_virtual<<<grid2,threads2>>>(i, d_d, n_count, d_dist);

		CudaCheckError();
#ifdef TIMER
		cudaDeviceSynchronize();
		gettimeofday(&t2, 0);
		time = (1000000.0*(t2.tv_sec-t1.tv_sec) + t2.tv_usec-t1.tv_usec)/1000000.0;
		cout << "initialization takes " << time << " secs\n";
		gettimeofday(&gt1, 0);
#endif


		do{
#ifdef TIMER
			gettimeofday(&t1, 0);
#endif

			assert (cudaSuccess == cudaMemset(d_continue, 0, sizeof(bool)));
			cc_forward_virtual<<<grid,threads>>> (d_vmap, d_vptrs, d_vjs, d_d, d_continue, d_dist, virn_count);
			CudaCheckError();
			cc_set_int<<<1,1>>>(d_dist, ++h_dist);
			assert (cudaSuccess == cudaMemcpy(&h_continue, d_continue, sizeof(bool), cudaMemcpyDeviceToHost));

#ifdef TIMER
			cudaDeviceSynchronize();
			gettimeofday(&t2, 0);
			time = (1000000.0*(t2.tv_sec-t1.tv_sec) + t2.tv_usec-t1.tv_usec)/1000000.0;
			cout << "level " <<  h_dist << " takes " << time << " secs\n";
#endif
		}while(h_continue);
		diameter = h_dist;
#ifdef TIMER
		gettimeofday(&gt2, 0);
		time = (1000000.0*(gt2.tv_sec-gt1.tv_sec) + gt2.tv_usec-gt1.tv_usec)/1000000.0;
		cout << "Phase 1 takes " << time << " secs\n";
		gettimeofday(&gt1, 0); // starts back propagation
#endif

		cc_backsum_virtual<<<grid2, threads2>>>(i, d_d, d_cc, n_count);
#ifdef TIMER
		cudaDeviceSynchronize();
		CudaCheckError();
		
		gettimeofday(&gt2, 0);
		time = (1000000.0*(gt2.tv_sec-gt1.tv_sec) + gt2.tv_usec-gt1.tv_usec)/1000000.0;
		cout << "backsum takes " << time << " secs\n";
#endif
	}


	std::cout<<"diameter: "<<diameter<<std::endl;

	assert (cudaSuccess == cudaMemcpy(h_cc, d_cc, sizeof(float)*n_count, cudaMemcpyDeviceToHost));
	cudaFree(d_vmap);
	cudaFree(d_vptrs);
	cudaFree(d_vjs);
	cudaFree(d_d);
	cudaFree(d_dist);
	cudaFree(d_cc);
	cudaFree(d_continue);
	CudaCheckError();
	return 0;
}


//explicit instanciation
template int cc_gpu_virtual<int,  int>  (int* h_vmap,  int* h_vptrs, int* h_vjs, int n_count,  int e_count,  int virn_count, int nb, float *h_cc);
template int cc_gpu_virtual<int, long int>  (int* h_vmap, long int* h_vptrs, int* h_vjs, int n_count, long int e_count, long int virn_count, int nb, float *h_cc);
template int cc_gpu_virtual<long int, long int>  (long int* h_vmap, long int* h_vptrs, long int* h_vjs, long int n_count, long int e_count, long int virn_count, long int nb, float *h_cc);
