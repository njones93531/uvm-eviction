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
#include <stdio.h>
#include "common.h"
#include "cuda_common.h"
#include <sstream>
using namespace std;

#define DEBUG

template<typename VtxType, typename EdgeIndex>
__global__ void orderEdges_kernel(EdgeIndex* d_xadj, VtxType* d_adj, VtxType n) { /* erases  -1s from adj list */
  VtxType u = (VtxType)blockIdx.x * (VtxType)blockDim.x + (VtxType)threadIdx.x;
	if(u < n) {
		auto wp = d_xadj[u];
		auto end = d_xadj[u+1];
		for(auto p = wp; p < end; p++) {
			auto j = d_adj[p];
			if(j != -1) {
				d_adj[p] = -1;
				d_adj[wp++] = j;
			}
		}
	}
}

template<typename VtxType, typename EdgeIndex>
__global__ void degreeSet_kernel(EdgeIndex* d_xadj, VtxType* d_degrees, VtxType n) {
  VtxType u = (VtxType)blockIdx.x * (VtxType)blockDim.x + (VtxType)threadIdx.x;
	if(u < n) {
		d_degrees[u] = d_xadj[u+1] - d_xadj[u];
	}
}

template <typename T>
__device__ T atomicAdd_int (T* ptr,T v) {
  return atomicAdd (ptr, v);
}

template <>
__device__ long int atomicAdd_int (long int * ptr, long int v) {
  return atomicAdd ((unsigned long long int*)ptr, (unsigned long long int) v);
}



template<typename VtxType, typename EdgeIndex>
__global__ void degree1_kernel(EdgeIndex* d_xadj, VtxType* d_adj, VtxType* d_tadj, VtxType n, float* d_bc, VtxType* d_weight, bool *d_continue, VtxType* d_degrees) {
	VtxType u = (VtxType)blockIdx.x * (VtxType)blockDim.x + (VtxType)threadIdx.x;

	if(u < n) {
		if(d_degrees[u] == 1) { /* degree 1 vertex is found */
		  //int p, v, end, remwght;
			*d_continue = true;
			d_degrees[u] = 0;
			EdgeIndex end = d_xadj[u + 1];
			for(EdgeIndex p = d_xadj[u]; p < end; p++) {
				VtxType v = d_adj[p];
				if(v != -1) {
					d_adj[p] = -1;
					d_adj[d_tadj[p]] = -1; /* bu satiri basit haliyle yazinca ne kaybediyoruz bakalim */

					VtxType remwght = n - d_weight[u];
					d_bc[u] += (d_weight[u] - 1) * remwght;

					atomicAdd (d_bc + v, d_weight[u] * (remwght - 1));
					atomicAdd_int (d_weight + v, d_weight[u]);
					atomicAdd_int (d_degrees + v, (VtxType) -1);
					break;
				}
			}
		}
	}
}

void init () {
	int* tmp;
	cudaMalloc((void **)&tmp, sizeof(int));
	cudaFree(tmp);
}

template<typename VtxType, typename EdgeIndex>
int preprocess(EdgeIndex *xadj, VtxType* adj, VtxType* tadj, VtxType *np, float* bc, VtxType* weight, VtxType* map_for_order, VtxType* reverse_map_for_order, FILE* ofp) {

	VtxType n = *np;
	EdgeIndex nz = xadj[n];
	//	fflush(0);

	EdgeIndex *d_xadj;
	VtxType *d_adj, *d_tadj, *d_weight;
	VtxType *d_degrees;
	float *d_bc;
	bool h_continue, *d_continue;
	cudaMalloc((void **)&d_xadj, sizeof(EdgeIndex)*(n+1));
	cudaMalloc((void **)&d_adj, sizeof(VtxType)* nz);
	cudaMalloc((void **)&d_tadj, sizeof(VtxType)* nz);
	cudaMalloc((void **)&d_weight, sizeof(VtxType)* n);
	cudaMalloc((void **)&d_bc, sizeof(float)* n);
	cudaMalloc((void **)&d_degrees, sizeof(VtxType)* n);
	cudaMalloc((void **)&d_continue, sizeof(bool));

	cudaMemcpy(d_xadj, xadj, sizeof(EdgeIndex) * (n+1), cudaMemcpyHostToDevice);
	cudaMemcpy(d_adj, adj, sizeof(VtxType) * nz, cudaMemcpyHostToDevice);
	cudaMemcpy(d_tadj, tadj, sizeof(VtxType) * nz, cudaMemcpyHostToDevice);
	cudaMemset(d_bc, 0, sizeof(float) * n);
	cudaMemcpy(d_weight, weight, sizeof(VtxType) * n, cudaMemcpyHostToDevice);

	VtxType threads_per_block = n;
	VtxType blocks = 1;
	if(n > MTS){
		blocks = (VtxType)ceil(n / (float)MTS);
		threads_per_block = MTS;
	}
	dim3 grid(blocks);
	dim3 threads(threads_per_block);

	// degree1 removal
	degreeSet_kernel<<<grid,threads>>>(d_xadj, d_degrees, n);
	do{
		h_continue = false;
		cudaMemcpy(d_continue, &h_continue, sizeof(bool), cudaMemcpyHostToDevice);
		degree1_kernel<<<grid,threads>>>(d_xadj, d_adj, d_tadj, n, d_bc, d_weight, d_continue, d_degrees);
		cudaMemcpy(&h_continue, d_continue, sizeof(bool), cudaMemcpyDeviceToHost);
	} while(h_continue);

	//shrink the pointers and reconstruct xadj and adj
	orderEdges_kernel<<<grid,threads>>>(d_xadj, d_adj, n);

	cudaMemcpy(bc, d_bc, sizeof(float) * n, cudaMemcpyDeviceToHost);
	cudaMemcpy(weight, d_weight, sizeof(VtxType) * n, cudaMemcpyDeviceToHost);
	cudaMemcpy(adj, d_adj, sizeof(VtxType) * nz, cudaMemcpyDeviceToHost);

	cudaFree(d_xadj);
	cudaFree(d_adj);
	cudaFree(d_tadj);
	cudaFree(d_bc);
	cudaFree(d_weight);
	cudaFree(d_continue);

	VtxType idx = 0;

	EdgeIndex ptr = 0;

	for (VtxType i = 0; i < n; i++) {
		bool flag = false;
		for (auto j = xadj[i]; j < xadj[i+1]; j++) {
			if (adj[j] != -1) {
				adj[ptr++] = adj[j];
			}
			else {
				flag = true;
				xadj[idx++] = ptr;
				break;
			}
		}
		if (!flag)
			xadj[idx++] = ptr;
	}

	for (VtxType i = idx; i > 0; i--) {
		xadj[i] = xadj[i-1];
	}
	xadj[0] = 0;

	VtxType vcount = 0;
	for (VtxType i = 0; i < n; i++) {
		if(xadj[i+1] != xadj[i]) {
			bc[vcount] = bc[i];
			weight[vcount] = weight[i];
			map_for_order[i] = vcount;
			reverse_map_for_order[vcount] = i;
			vcount++;
			xadj[vcount] = xadj[i+1];
		}
		else {
		  std::stringstream ss;
		  ss<<"bc["<<i<<"]: "<<bc[i];
		  fprintf(ofp, "%s\n", ss.str().c_str());
		}
	}
	for (EdgeIndex i = 0; i < xadj[vcount]; i++) {
		adj[i] = map_for_order[adj[i]];
	}
	*np = vcount;

	return 0;
}

//specialization
template
int preprocess<int, int> (int *xadj, int* adj, int* tadj, int *np, float* bc, int* weight, int* map_for_order, int* reverse_map_for_order, FILE* ofp);

template
int preprocess<int, long int> (long int *xadj, int* adj, int* tadj, int *np, float* bc, int* weight, int* map_for_order, int* reverse_map_for_order, FILE* ofp);

template
int preprocess<long int, long  int> (long int *xadj, long int* adj, long int* tadj, long int *np, float* bc, long int* weight, long int* map_for_order, long int* reverse_map_for_order, FILE* ofp);
