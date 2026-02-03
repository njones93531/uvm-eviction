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




template <typename VtxType, typename EdgeIndex> //The type of virtual vertices should be edgeindex since there is typically about |E|/constant of them
__global__ void forward_virtual (VtxType* d_vmap, EdgeIndex* d_vptrs, VtxType* d_vjs, VtxType *d_d, int *d_sigma, bool *d_continue, VtxType *d_dist, EdgeIndex virn_count) {
  EdgeIndex vu = ((EdgeIndex)blockIdx.x) * blockDim.x * gridDim.y + blockIdx.y * blockDim.x + threadIdx.x;
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
				if(d_d[w] == *d_dist + 1) {
					atomicAdd(&d_sigma[w], d_sigma[u]);
				}
			}
		}
	}
}

template <typename VtxType, typename EdgeIndex>
__global__ void forward_virtual_coalesced (VtxType* d_vmap, VtxType* d_vjs, VtxType *d_d, int *d_sigma, bool *d_continue, VtxType *d_dist, EdgeIndex virn_count, VtxType *d_stride, EdgeIndex *d_startoffset, EdgeIndex *d_xadj) {
  EdgeIndex vu = ((EdgeIndex)blockIdx.x) * blockDim.x * gridDim.y + ((EdgeIndex)blockIdx.y) * blockDim.x + threadIdx.x;
	if(vu < virn_count) {
		VtxType u = d_vmap[vu];
		/* for each edge (u, w) s.t. u is unvisited, w is in the current level */
		if(d_d[u] == *d_dist) {
			EdgeIndex end = d_xadj[u + 1];
			VtxType stride = d_stride[u];
			for(EdgeIndex p = d_startoffset[vu]; p < end; p+=stride) {
				VtxType w = d_vjs[p];
				if(d_d[w] == -1) {
					d_d[w] = *d_dist + 1;
					*d_continue = 1;
				}
				if(d_d[w] == *d_dist + 1) {
					atomicAdd(&d_sigma[w], d_sigma[u]);
				}
			}
		}
	}
}

template <typename VtxType, typename EdgeIndex>
__global__ void backward_virtual_coalesced (VtxType* d_vmap, VtxType* d_vjs, VtxType *d_d, float *d_delta, VtxType *d_dist, EdgeIndex virn_count, VtxType *d_stride, EdgeIndex *d_startoffset, EdgeIndex *d_xadj){
  EdgeIndex vu = ((EdgeIndex)blockIdx.x) * blockDim.x * gridDim.y + ((EdgeIndex)blockIdx.y) * blockDim.x + threadIdx.x;
	if(vu < virn_count) {
		VtxType u = d_vmap[vu];
		if(d_d[u] == *d_dist - 1) {
			EdgeIndex end = d_xadj[u + 1];
			VtxType stride = d_stride[u];
			float sum = 0;
			for(EdgeIndex p = d_startoffset[vu]; p < end; p+=stride) {
				VtxType w = d_vjs[p];
				if(d_d[w] == *d_dist ) {
					sum += d_delta[w];
				}
			}
			atomicAdd(&d_delta[u], sum);
		}
	}
}


template <typename VtxType, typename EdgeIndex>
__global__ void backward_virtual (VtxType* d_vmap, EdgeIndex* d_vptrs, VtxType* d_vjs, VtxType *d_d, float *d_delta, VtxType *d_dist, EdgeIndex virn_count){
  EdgeIndex vu = ((EdgeIndex)blockIdx.x) * blockDim.x * gridDim.y + ((EdgeIndex)blockIdx.y) * blockDim.x + threadIdx.x;
	if(vu < virn_count) {
		VtxType u = d_vmap[vu];
		if(d_d[u] == *d_dist - 1) {
			EdgeIndex end = d_vptrs[vu+1];
			float sum = 0;
			for(EdgeIndex p = d_vptrs[vu]; p < end; p++) {
				VtxType w = d_vjs[p];
				if(d_d[w] == *d_dist ) {
					sum += d_delta[w];
				}
			}
			atomicAdd(&d_delta[u], sum);
		}
	}
}

template <typename VtxType>
__global__ void intermediate_virtual (VtxType *d_d, int *d_sigma, float *d_delta, VtxType *d_dist, VtxType n_count) {
  VtxType u = ((VtxType)blockIdx.x) * blockDim.x * gridDim.y + ((VtxType)blockIdx.y) * blockDim.x + threadIdx.x;
	if(u < n_count) {
		d_delta[u] = 1.0f / d_sigma[u];
	}
}

template <typename VtxType>
__global__ void intermediate_virtual_deg1 (VtxType *d_d, int *d_sigma, float *d_delta, VtxType *d_dist, VtxType n_count, VtxType* d_weight) {
  VtxType u = ((VtxType)blockIdx.x) * blockDim.x * gridDim.y + ((VtxType)blockIdx.y) * blockDim.x + threadIdx.x;
	if(u < n_count) {
		d_delta[u] = d_weight[u] / (float)d_sigma[u];
	}
}

template <typename VtxType>
__global__ void backsum_virtual_deg1 (VtxType s, VtxType *d_d, float *d_delta, int *d_sigma, float *d_bc, VtxType n_count, VtxType* d_weight){
  VtxType tid = ((VtxType)blockIdx.x) * blockDim.x * gridDim.y + ((VtxType)blockIdx.y) * blockDim.x + threadIdx.x;
	if(tid < n_count && tid != s && d_d[tid] != -1) {
		d_bc[tid] += (d_delta[tid] * d_sigma[tid] - 1) * d_weight[s];
	}
}


template <typename VtxType>
__global__ void backsum_virtual (VtxType s, VtxType *d_d, float *d_delta, int *d_sigma, float *d_bc, VtxType n_count){
  VtxType tid = ((VtxType)blockIdx.x) * blockDim.x * gridDim.y + ((VtxType)blockIdx.y) * blockDim.x + threadIdx.x;
	if(tid < n_count && tid != s && d_d[tid] != -1) {
		d_bc[tid] += d_delta[tid] * d_sigma[tid] - 1;
	}
}

template <typename VtxType>
__global__ void init_virtual (VtxType s, VtxType *d_d, int *d_sigma, VtxType n_count, VtxType* d_dist){
  VtxType i = ((VtxType)blockIdx.x) * blockDim.x * gridDim.y + ((VtxType)blockIdx.y) * blockDim.x + threadIdx.x;
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
__global__ void set_int (VtxType* dest, VtxType val){
	*dest = val;
}


template <typename VtxType, typename EdgeIndex>
int bc_gpu_virtual (VtxType* h_vmap, EdgeIndex* h_vptrs, VtxType* h_vjs, VtxType n_count, EdgeIndex e_count, EdgeIndex virn_count, VtxType nb, float *h_bc) {
	VtxType *d_vmap, *d_vjs, *d_d, *d_dist, h_dist;
	EdgeIndex *d_vptrs;
	int *d_sigma;
	float *d_delta, *d_bc;
	bool h_continue, *d_continue;


	assert (cudaSuccess == cudaMalloc((void **)&d_vmap, sizeof(VtxType) *  virn_count));
	assert (cudaSuccess == cudaMalloc((void **)&d_vptrs, sizeof(EdgeIndex) * (virn_count + 1)));
	assert (cudaSuccess == cudaMalloc((void **)&d_vjs, sizeof(VtxType) * e_count));

	assert (cudaSuccess == cudaMemcpy(d_vmap, h_vmap, sizeof(VtxType) * virn_count, cudaMemcpyHostToDevice));
	assert (cudaSuccess == cudaMemcpy(d_vptrs, h_vptrs, sizeof(EdgeIndex) * (virn_count + 1), cudaMemcpyHostToDevice));
	assert (cudaSuccess == cudaMemcpy(d_vjs, h_vjs, sizeof(VtxType) * e_count, cudaMemcpyHostToDevice));

	assert (cudaSuccess == cudaMalloc((void **)&d_d, sizeof(VtxType)*n_count));

	assert (cudaSuccess == cudaMalloc((void **)&d_sigma, sizeof(int)*n_count));
	assert (cudaSuccess == cudaMalloc((void **)&d_delta, sizeof(float)*n_count));
	assert (cudaSuccess == cudaMalloc((void **)&d_dist, sizeof(VtxType)));

	assert (cudaSuccess == cudaMalloc((void **)&d_bc, sizeof(float)*n_count));
	assert (cudaSuccess == cudaMemset(d_bc, 0, sizeof(float)*n_count));

	assert (cudaSuccess == cudaMalloc((void **)&d_continue, sizeof(bool)));

	EdgeIndex threads_per_block = virn_count;
	int blocks = 1;
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

	for(VtxType i = 0; i < min(nb, n_count); i++){
#ifdef TIMER
		gettimeofday(&t1, 0);
#endif

		h_dist = 0;
		init_virtual<<<grid2,threads2>>>(i, d_d, d_sigma, n_count, d_dist);

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
			forward_virtual<<<grid,threads>>>(d_vmap, d_vptrs, d_vjs, d_d, d_sigma, d_continue, d_dist, virn_count);
			CudaCheckError();
			set_int<<<1,1>>>(d_dist, ++h_dist);
			assert (cudaSuccess == cudaMemcpy(&h_continue, d_continue, sizeof(bool), cudaMemcpyDeviceToHost));

#ifdef TIMER
			cudaDeviceSynchronize();
			gettimeofday(&t2, 0);
			time = (1000000.0*(t2.tv_sec-t1.tv_sec) + t2.tv_usec-t1.tv_usec)/1000000.0;
			cout << "level " <<  h_dist << " takes " << time << " secs\n";
#endif
		}while(h_continue);
#ifdef TIMER
		gettimeofday(&gt2, 0);
		time = (1000000.0*(gt2.tv_sec-gt1.tv_sec) + gt2.tv_usec-gt1.tv_usec)/1000000.0;
		cout << "Phase 1 takes " << time << " secs\n";
		gettimeofday(&gt1, 0); // starts back propagation
#endif

		set_int<<<1,1>>>(d_dist, --h_dist);
		intermediate_virtual<<<grid2, threads2>>>(d_d, d_sigma, d_delta, d_dist, n_count);
		while(h_dist > 1) {
			backward_virtual<<<grid, threads>>>(d_vmap, d_vptrs, d_vjs, d_d, d_delta, d_dist, virn_count);
			set_int<<<1,1>>>(d_dist, --h_dist);
		}
		backsum_virtual<<<grid2, threads2>>>(i, d_d,  d_delta, d_sigma, d_bc, n_count);
#ifdef TIMER
		cudaDeviceSynchronize();
		CudaCheckError();

		gettimeofday(&gt2, 0);
		time = (1000000.0*(gt2.tv_sec-gt1.tv_sec) + gt2.tv_usec-gt1.tv_usec)/1000000.0;
		cout << "Phase 2 takes " << time << " secs\n";
#endif
	}

	assert (cudaSuccess == cudaMemcpy(h_bc, d_bc, sizeof(float)*n_count, cudaMemcpyDeviceToHost));
	cudaFree(d_vmap);
	cudaFree(d_vptrs);
	cudaFree(d_vjs);
	cudaFree(d_d);
	cudaFree(d_sigma);
	cudaFree(d_delta);
	cudaFree(d_dist);
	cudaFree(d_bc);
	cudaFree(d_continue);
	CudaCheckError();
	return 0;
}


template <typename VtxType, typename EdgeIndex>
int bc_gpu_virtual_coalesced (VtxType* h_vmap, EdgeIndex* h_xadj, VtxType* h_vjs, VtxType n_count, EdgeIndex* h_startoffset, VtxType* h_stride, EdgeIndex e_count, EdgeIndex virn_count, VtxType nb, float *h_bc) {
  int *d_sigma;
	VtxType *d_vmap, *d_vjs, *d_d,  *d_dist, h_dist;
	float *d_delta, *d_bc;
	bool h_continue, *d_continue;

	EdgeIndex *d_xadj;
	EdgeIndex *d_startoffset;
	VtxType *d_stride;

	assert (cudaSuccess == cudaMalloc((void **)&d_vmap, sizeof(VtxType) *  virn_count));
	assert (cudaSuccess == cudaMemcpy(d_vmap, h_vmap, sizeof(VtxType) * virn_count, cudaMemcpyHostToDevice));

	assert (cudaSuccess == cudaMalloc((void **)&d_vjs, sizeof(VtxType) * e_count));
	assert (cudaSuccess == cudaMemcpy(d_vjs, h_vjs, sizeof(VtxType) * e_count, cudaMemcpyHostToDevice));

	assert (cudaSuccess == cudaMalloc((void **)&d_xadj, sizeof(EdgeIndex) * (n_count + 1)));
	assert (cudaSuccess == cudaMemcpy(d_xadj, h_xadj, sizeof(EdgeIndex) * (n_count + 1), cudaMemcpyHostToDevice));

	assert (cudaSuccess == cudaMalloc((void **)&d_startoffset, sizeof(EdgeIndex) * (virn_count)));
	assert (cudaSuccess == cudaMemcpy(d_startoffset, h_startoffset, sizeof(EdgeIndex) * (virn_count), cudaMemcpyHostToDevice));

	assert (cudaSuccess == cudaMalloc((void **)&d_stride, sizeof(VtxType) * (n_count)));
	assert (cudaSuccess == cudaMemcpy(d_stride, h_stride, sizeof(VtxType) * (n_count), cudaMemcpyHostToDevice));


	assert (cudaSuccess == cudaMalloc((void **)&d_d, sizeof(VtxType)*n_count));

	assert (cudaSuccess == cudaMalloc((void **)&d_sigma, sizeof(int)*n_count));
	assert (cudaSuccess == cudaMalloc((void **)&d_delta, sizeof(float)*n_count));
	assert (cudaSuccess == cudaMalloc((void **)&d_dist, sizeof(VtxType)));

	assert (cudaSuccess == cudaMalloc((void **)&d_bc, sizeof(float)*n_count));
	assert (cudaSuccess == cudaMemset(d_bc, 0, sizeof(float)*n_count));

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

	cout<<"coalesced"<<std::endl;

#ifdef TIMER
	struct timeval t1, t2, gt1, gt2; double time;
#endif
	VtxType diameter;

	for(VtxType i = 0; i < min(nb, n_count); i++){
#ifdef TIMER
		gettimeofday(&t1, 0);
#endif

		h_dist = 0;
		init_virtual<<<grid,threads>>>(i, d_d, d_sigma, n_count, d_dist);
#ifdef TIMER
		cudaDeviceSynchronize();
		CudaCheckError();
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
			forward_virtual_coalesced<<<grid,threads>>>(d_vmap,  d_vjs, d_d, d_sigma, d_continue, d_dist, virn_count, d_stride, d_startoffset, d_xadj);
			set_int<<<1,1>>>(d_dist, ++h_dist);
			assert (cudaSuccess == cudaMemcpy(&h_continue, d_continue, sizeof(bool), cudaMemcpyDeviceToHost));

#ifdef TIMER
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

		set_int<<<1,1>>>(d_dist, --h_dist);
		intermediate_virtual<<<grid2, threads2>>>(d_d, d_sigma, d_delta, d_dist, n_count);
		while(h_dist > 1) {
			backward_virtual_coalesced<<<grid, threads>>>(d_vmap, d_vjs, d_d, d_delta, d_dist, virn_count, d_stride, d_startoffset, d_xadj);
			set_int<<<1,1>>>(d_dist, --h_dist);
		}
		backsum_virtual<<<grid2, threads2>>>(i, d_d,  d_delta, d_sigma, d_bc, n_count);

#ifdef TIMER
		cudaDeviceSynchronize();
		CudaCheckError();
		
		gettimeofday(&gt2, 0);
		time = (1000000.0*(gt2.tv_sec-gt1.tv_sec) + gt2.tv_usec-gt1.tv_usec)/1000000.0;
		cout << "Phase 2 takes " << time << " secs\n";
#endif
	}

	CudaCheckError();

	std::cout<<"diameter: "<<diameter<<std::endl;
	
	assert (cudaSuccess == cudaMemcpy(h_bc, d_bc, sizeof(float)*n_count, cudaMemcpyDeviceToHost));
	cudaFree(d_vmap);
	cudaFree(d_vjs);
	cudaFree(d_d);
	cudaFree(d_sigma);
	cudaFree(d_delta);
	cudaFree(d_dist);
	cudaFree(d_bc);
	cudaFree(d_continue);
	CudaCheckError();
	return 0;
}


template <typename VtxType, typename EdgeIndex>
int bc_gpu_virtual_deg1 (VtxType* h_vmap, EdgeIndex* h_vptrs, VtxType* h_vjs, VtxType n_count, EdgeIndex e_count, EdgeIndex virn_count, VtxType nb, float *h_bc, VtxType* h_weight) {
  int *d_sigma;
  EdgeIndex *d_vptrs;
	VtxType *d_vmap, *d_vjs, *d_d,  *d_dist, h_dist;
	float *d_delta, *d_bc;
	VtxType* d_weight;
	bool h_continue, *d_continue;



	assert (cudaSuccess == cudaMalloc((void **)&d_vmap, sizeof(VtxType) *  virn_count));
	assert (cudaSuccess == cudaMalloc((void **)&d_vptrs, sizeof(EdgeIndex) * (virn_count + 1)));
	assert (cudaSuccess == cudaMalloc((void **)&d_vjs, sizeof(VtxType) * e_count));

	assert (cudaSuccess == cudaMemcpy(d_vmap, h_vmap, sizeof(VtxType) * virn_count, cudaMemcpyHostToDevice));
	assert (cudaSuccess == cudaMemcpy(d_vptrs, h_vptrs, sizeof(EdgeIndex) * (virn_count + 1), cudaMemcpyHostToDevice));
	assert (cudaSuccess == cudaMemcpy(d_vjs, h_vjs, sizeof(VtxType) * e_count, cudaMemcpyHostToDevice));

	assert (cudaSuccess == cudaMalloc((void **)&d_d, sizeof(VtxType)*n_count));

	assert (cudaSuccess == cudaMalloc((void **)&d_sigma, sizeof(int)*n_count));
	assert (cudaSuccess == cudaMalloc((void **)&d_delta, sizeof(float)*n_count));
	assert (cudaSuccess == cudaMalloc((void **)&d_weight, sizeof(VtxType) * n_count));
	assert (cudaSuccess == cudaMemcpy(d_weight, h_weight, sizeof(VtxType) * n_count, cudaMemcpyHostToDevice)); // weight array
	assert (cudaSuccess == cudaMalloc((void **)&d_dist, sizeof(VtxType)));

	assert (cudaSuccess == cudaMalloc((void **)&d_bc, sizeof(float) * n_count));
	assert (cudaSuccess == cudaMemcpy(d_bc, h_bc, sizeof(int) * n_count, cudaMemcpyHostToDevice)); // bc array

	assert (cudaSuccess == cudaMalloc((void **)&d_continue, sizeof(bool)));

	EdgeIndex threads_per_block = virn_count;
	EdgeIndex blocks = 1;
	if(virn_count > MTS){
		blocks = (EdgeIndex)ceil(virn_count/(double)(MTS));
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
		blocks2 = (VtxType)ceil(n_count/(double)(MTS));
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
	for(VtxType i = 0; i < min (nb, n_count); i++){
#ifdef TIMER
		gettimeofday(&t1, 0);
#endif

		h_dist = 0;
		init_virtual<<<grid,threads>>>(i, d_d, d_sigma, n_count, d_dist);
		CudaCheckError();

#ifdef TIMER
		gettimeofday(&t2, 0);
		time = (1000000.0*(t2.tv_sec-t1.tv_sec) + t2.tv_usec-t1.tv_usec)/1000000.0;
		cout << "initialization takes " << time << " secs\n";
		gettimeofday(&gt1, 0);
#endif
		do{
#ifdef TIMER
			gettimeofday(&t1, 0);
#endif

			cudaMemset(d_continue, 0, sizeof(bool));
			forward_virtual<<<grid,threads>>>(d_vmap, d_vptrs, d_vjs, d_d, d_sigma, d_continue, d_dist, virn_count);
			set_int<<<1,1>>>(d_dist, ++h_dist);
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

		set_int<<<1,1>>>(d_dist, --h_dist);
		intermediate_virtual_deg1<<<grid2, threads2>>>(d_d, d_sigma, d_delta, d_dist, n_count, d_weight);
		while(h_dist > 1) {
			backward_virtual<<<grid, threads>>>(d_vmap, d_vptrs, d_vjs, d_d, d_delta, d_dist, virn_count);
			set_int<<<1,1>>>(d_dist, --h_dist);
		}

		backsum_virtual_deg1<<<grid2, threads2>>>(i, d_d,  d_delta, d_sigma, d_bc, n_count, d_weight);
#ifdef TIMER
		cudaDeviceSynchronize();
		gettimeofday(&gt2, 0);
		time = (1000000.0*(gt2.tv_sec-gt1.tv_sec) + gt2.tv_usec-gt1.tv_usec)/1000000.0;
		cout << "Phase 2 takes " << time << " secs\n";
#endif
	}

	assert (cudaSuccess == cudaMemcpy(h_bc, d_bc, sizeof(float)*n_count, cudaMemcpyDeviceToHost));
	cudaFree(d_vmap);
	cudaFree(d_vptrs);
	cudaFree(d_vjs);
	cudaFree(d_d);
	cudaFree(d_sigma);
	cudaFree(d_delta);
	cudaFree(d_dist);
	cudaFree(d_bc);
	cudaFree(d_continue);
	CudaCheckError();
	return 0;
}

//explicit instanciation
template int bc_gpu_virtual<int, int> (int* h_vmap, int* h_vptrs, int* h_vjs, int n_count, int e_count, int virn_count, int nb, float *h_bc);
template int bc_gpu_virtual<int, long int> (int* h_vmap, long int* h_vptrs, int* h_vjs, int n_count, long int e_count, long int virn_count, int nb, float *h_bc);
template int bc_gpu_virtual<long int, long int> (long int* h_vmap, long int* h_vptrs, long int* h_vjs, long int n_count, long int e_count, long int virn_count, long int nb, float *h_bc);


template int bc_gpu_virtual_coalesced<int,  int> (int* h_vmap, int* h_xadj, int* h_vjs, int n_count, int* h_startoffset, int* h_stride, int e_count, int virn_count, int nb, float *h_bc);
template int bc_gpu_virtual_coalesced<int,  long int> (int* h_vmap, long int* h_xadj, int* h_vjs, int n_count, long int* h_startoffset, int* h_stride, long int e_count, long int virn_count, int nb, float *h_bc);
template int bc_gpu_virtual_coalesced<long int, long int> (long int* h_vmap, long int* h_xadj, long int* h_vjs, long int n_count, long int* h_startoffset, long int* h_stride, long int e_count, long int virn_count, long int nb, float *h_bc);


template int bc_gpu_virtual_deg1<int, int> (int* h_vmap, int* h_vptrs, int* h_vjs, int n_count, int e_count, int virn_count, int nb, float *h_bc, int* h_weight);
template int bc_gpu_virtual_deg1<int, long int> (int* h_vmap, long int* h_vptrs, int* h_vjs, int n_count, long int e_count, long int virn_count, int nb, float *h_bc, int* h_weight);
template int bc_gpu_virtual_deg1<long int, long int> (long int* h_vmap, long int* h_vptrs, long int* h_vjs, long int n_count, long int e_count, long int virn_count, long int nb, float *h_bc, long int* h_weight);
