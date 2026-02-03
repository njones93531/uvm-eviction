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

//#define NB_BFS 8
//#define TIMER
#define ANALYSIS

using namespace std;

template <typename VtxType, typename EdgeIndex, int NB_BFS>
__global__ void forward_virtual_multi (VtxType* d_vmap, EdgeIndex* d_vptrs, VtxType* d_vjs, VtxType *d_d, int *d_sigma, bool *d_continue, VtxType *d_dist, VtxType virn_count) {
  EdgeIndex tid = ((EdgeIndex)blockIdx.x) * blockDim.x * gridDim.y + ((EdgeIndex)blockIdx.y) * blockDim.x + threadIdx.x;
  EdgeIndex vu = tid / NB_BFS;
  int bfs = tid % NB_BFS;
	if(vu < virn_count) {
		VtxType u = d_vmap[vu];
		/* for each edge (u, w) s.t. u is unvisited, w is in the current level */
		if(d_d[u*NB_BFS+bfs] == *d_dist) {
			EdgeIndex end = d_vptrs[vu + 1];
			for(EdgeIndex p = d_vptrs[vu]; p < end; p++) {
				VtxType w = d_vjs[p];
				if(d_d[w*NB_BFS+bfs] == -1) {
					d_d[w*NB_BFS+bfs] = *d_dist + 1;
					*d_continue = 1;
				}
				if(d_d[w*NB_BFS+bfs] == *d_dist + 1) {
					atomicAdd(&d_sigma[w*NB_BFS+bfs], d_sigma[u*NB_BFS+bfs]);
				}
			}
		}
	}
}

template <typename VtxType, typename EdgeIndex, int NB_BFS>
__global__ void backward_virtual_multi (VtxType* d_vmap, EdgeIndex* d_vptrs, VtxType* d_vjs, VtxType *d_d, float *d_delta, VtxType *d_dist, EdgeIndex virn_count){
  EdgeIndex tid = ((EdgeIndex)blockIdx.x) * blockDim.x * gridDim.y + ((EdgeIndex)blockIdx.y) * blockDim.x + threadIdx.x;
	EdgeIndex vu = tid / NB_BFS;
	int bfs = tid % NB_BFS;
	if(vu < virn_count) {
		VtxType u = d_vmap[vu];
		if(d_d[u*NB_BFS+bfs] == *d_dist - 1) {
			EdgeIndex end = d_vptrs[vu+1];
			float sum = 0;
			for(EdgeIndex p = d_vptrs[vu]; p < end; p++) {
				VtxType w = d_vjs[p];
				if(d_d[w*NB_BFS+bfs] == *d_dist ) {
					sum += d_delta[w*NB_BFS+bfs];
				}
			}
			atomicAdd(&d_delta[u*NB_BFS+bfs], sum);
		}
	}
}

template <typename VtxType, int NB_BFS>
__global__ void intermediate_virtual_multi (VtxType *d_d, int *d_sigma, float *d_delta, VtxType *d_dist, VtxType n_count) {
  VtxType tid = ((VtxType)blockIdx.x) * blockDim.x * gridDim.y + ((VtxType)blockIdx.y) * blockDim.x + threadIdx.x;
	VtxType u = tid / NB_BFS;
	int bfs = tid % NB_BFS;
	if(u < n_count) {
		d_delta[u*NB_BFS+bfs] = 1.0f / d_sigma[u*NB_BFS+bfs];
	}
}

template <typename VtxType, int NB_BFS>
__global__ void backsum_virtual_multi (VtxType* s, VtxType *d_d, float *d_delta, int *d_sigma, float *d_bc, VtxType n_count){
  VtxType tid = ((VtxType)blockIdx.x) * blockDim.x * gridDim.y + ((VtxType)blockIdx.y) * blockDim.x + threadIdx.x;
	VtxType u = tid;
	if(u < n_count) {
	  for (int bfs = 0; bfs < NB_BFS; ++bfs)
	    if (u != s[bfs] && d_d[u*NB_BFS+bfs] != -1) {
		d_bc[u] += d_delta[u*NB_BFS+bfs] * d_sigma[u*NB_BFS+bfs] - 1;
	    }
	}
}

template <typename VtxType, int NB_BFS>
__global__ void init_virtual_multi (VtxType* s, VtxType *d_d, int *d_sigma, VtxType n_count) {
  VtxType tid = ((VtxType)blockIdx.x) * blockDim.x * gridDim.y + ((VtxType)blockIdx.y) * blockDim.x + threadIdx.x;
	VtxType i = tid / NB_BFS;
	int bfs = tid % NB_BFS;
	if(i < n_count) {
		d_d[i*NB_BFS+bfs] = -1;
		d_sigma[i*NB_BFS+bfs] = 0;
		if(s[bfs] == i) {
			d_d[i*NB_BFS+bfs] = 0;
			d_sigma[i*NB_BFS+bfs] = 1;
		}
	}
}

template <typename VtxType>
__global__ void set_int_multi (VtxType* dest, VtxType val){
	*dest = val;
}

template <typename VtxType, typename EdgeIndex>
int bc_gpu_virtual_multi (VtxType* h_vmap, EdgeIndex* h_vptrs, VtxType* h_vjs, VtxType n_count, EdgeIndex e_count, EdgeIndex virn_count, VtxType nb, float *h_bc) {

  EdgeIndex *d_vptrs;
  VtxType *d_vmap,  *d_vjs, *d_d, *d_dist, h_dist;
	int *d_sigma;
	float *d_delta, *d_bc;
	bool h_continue, *d_continue;


	int NB_BFS = 8;

	{
	  char* str = getenv("NB_BFS");
	  if (str != NULL)
	    NB_BFS = atoi(str);
	}

 	std::cout<<"NB_BFS="<<NB_BFS<<std::endl;

	assert (cudaSuccess == cudaMalloc((void **)&d_vmap, sizeof(VtxType) *  virn_count));
	assert (cudaSuccess == cudaMalloc((void **)&d_vptrs, sizeof(EdgeIndex) * (virn_count + 1)));
	assert (cudaSuccess == cudaMalloc((void **)&d_vjs, sizeof(VtxType) * e_count));

	assert (cudaSuccess == cudaMemcpy(d_vmap, h_vmap, sizeof(VtxType) * virn_count, cudaMemcpyHostToDevice));
	assert (cudaSuccess == cudaMemcpy(d_vptrs, h_vptrs, sizeof(EdgeIndex) * (virn_count + 1), cudaMemcpyHostToDevice));
	assert (cudaSuccess == cudaMemcpy(d_vjs, h_vjs, sizeof(VtxType) * e_count, cudaMemcpyHostToDevice));

	assert (cudaSuccess == cudaMalloc((void **)&d_d, sizeof(VtxType)*n_count*NB_BFS));

	assert (cudaSuccess == cudaMalloc((void **)&d_sigma, sizeof(int)*n_count*NB_BFS));
	assert (cudaSuccess == cudaMalloc((void **)&d_delta, sizeof(float)*n_count*NB_BFS));
	assert (cudaSuccess == cudaMalloc((void **)&d_dist, sizeof(VtxType)));

	assert (cudaSuccess == cudaMalloc((void **)&d_bc, sizeof(float)*n_count));
	assert (cudaSuccess == cudaMemset(d_bc, 0, sizeof(float)*n_count));

	assert (cudaSuccess == cudaMalloc((void **)&d_continue, sizeof(bool)));

	VtxType *h_d = (VtxType*) malloc (sizeof(VtxType)*(n_count*NB_BFS+32));

	VtxType* d_sources;
	cudaMalloc((void **)&d_sources, sizeof(VtxType)*NB_BFS);
	CudaCheckError();

	assert (virn_count < std::numeric_limits<int>::max()/NB_BFS); //otherwise block thread size will go negative

	//this one is for VN * NB_BFS threads
	EdgeIndex threads_per_block = virn_count * NB_BFS;
	EdgeIndex blocks = 1;
	if(threads_per_block > MTS){
		blocks = (EdgeIndex)ceil(threads_per_block/(double)MTS);
		blocks = (EdgeIndex)ceil(sqrt((float)blocks));
		threads_per_block = MTS;
	}
	dim3 grid;
	grid.x = blocks;
	grid.y = blocks;
	dim3 threads(threads_per_block);

	//this one is for V * NB_BFS threads
	VtxType threads_per_block2 = n_count * NB_BFS;
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


	//this one is for V threads
	VtxType threads_per_block3 = n_count;
	VtxType blocks3 = 1;
	if(threads_per_block3 > MTS){
		blocks3 = (VtxType)ceil(threads_per_block3/(double)MTS);
		blocks3 = (VtxType)ceil(sqrt((float)blocks3));
		threads_per_block3 = MTS;
	}
	dim3 grid3;
	grid3.x = blocks3;
	grid3.y = blocks3;
	dim3 threads3(threads_per_block3);


	cerr<<"cuda parameters: "
	    <<blocks<<" "<<threads_per_block<<" "
	    <<blocks2<<" "<<threads_per_block2<<" "
	    <<blocks3<<" "<<threads_per_block3<<" "
	    <<std::endl;

#ifdef TIMER
	struct timeval t1, t2, gt1, gt2; double time;
#endif

	assert (min (nb, n_count)%NB_BFS == 0);


	long int active = 0;
	long int non_simu_traverse = 0;

	for(VtxType i = 0; i < min(nb, n_count); i+= NB_BFS){
#ifdef TIMER
		gettimeofday(&t1, 0);
#endif

		for (int x = 0; x<NB_BFS; ++x) {
		  set_int_multi<<<1,1>>>(d_sources+x, i+x);
		  CudaCheckError();
		}

		h_dist = 0;
		set_int_multi<<<1,1>>>(d_dist, h_dist);
		switch (NB_BFS) 
		  {
#define CASE(X)	    case X: init_virtual_multi<VtxType,X> <<<grid2,threads2>>>(d_sources, d_d, d_sigma, n_count); break;
		    CASE(1);CASE(2);CASE(3);CASE(4);CASE(5);CASE(6);CASE(7);CASE(8);
		    CASE(16);CASE(32);CASE(64);CASE(128);
		  default: assert(0);
#undef CASE
		  }
#ifdef TIMER
		cudaDeviceSynchronize();
		CudaCheckError();

		gettimeofday(&t2, 0);
		time = (1000000.0*(t2.tv_sec-t1.tv_sec) + t2.tv_usec-t1.tv_usec)/1000000.0;
		cerr << "initialization takes " << time << " secs"<<std::endl;
		gettimeofday(&gt1, 0);
#endif


		do{
#ifdef TIMER
			gettimeofday(&t1, 0);
#endif

			assert (cudaSuccess == cudaMemset(d_continue, 0, sizeof(bool)));

			switch (NB_BFS)
			  {
#define CASE(X) case X: forward_virtual_multi<VtxType,EdgeIndex,X><<<grid,threads>>>(d_vmap, d_vptrs, d_vjs, d_d, d_sigma, d_continue, d_dist, virn_count); break;
			    CASE(1);CASE(2);CASE(3);CASE(4);CASE(5);CASE(6);CASE(7);CASE(8);
			    CASE(16);CASE(32);CASE(64);CASE(128);			    
			  default: assert(0);
#undef CASE
			  }

			set_int_multi<<<1,1>>>(d_dist, ++h_dist);
			assert (cudaSuccess == cudaMemcpy(&h_continue, d_continue, sizeof(bool), cudaMemcpyDeviceToHost));

#ifdef TIMER
			gettimeofday(&t2, 0);
			time = (1000000.0*(t2.tv_sec-t1.tv_sec) + t2.tv_usec-t1.tv_usec)/1000000.0;
			cerr << "level " <<  h_dist << " takes " << time << " secs"<<std::endl;
#endif
		}while(h_continue);
#ifdef TIMER
		gettimeofday(&gt2, 0);
		time = (1000000.0*(gt2.tv_sec-gt1.tv_sec) + gt2.tv_usec-gt1.tv_usec)/1000000.0;
		cerr << "Phase 1 takes " << time << " secs"<<std::endl;
		gettimeofday(&gt1, 0); // starts back propagation
#endif

		set_int_multi<<<1,1>>>(d_dist, --h_dist);

		switch (NB_BFS)
		  {
#define CASE(X) case X:	intermediate_virtual_multi<VtxType,X><<<grid2, threads2>>>(d_d, d_sigma, d_delta, d_dist, n_count); break;
		    CASE(1);CASE(2);CASE(3);CASE(4);CASE(5);CASE(6);CASE(7);CASE(8);
		    CASE(16);CASE(32);CASE(64);CASE(128);
		  default: assert(0);
#undef CASE
		  }

		while(h_dist > 1) {
		  switch (NB_BFS)
		    {
#define CASE(X) case X: backward_virtual_multi<VtxType,EdgeIndex,X><<<grid, threads>>>(d_vmap, d_vptrs, d_vjs, d_d, d_delta, d_dist, virn_count); break;
			CASE(1);CASE(2);CASE(3);CASE(4);CASE(5);CASE(6);CASE(7);CASE(8);
			CASE(16);CASE(32);CASE(64);CASE(128);
		    default: assert(0);
#undef CASE
		    }

			set_int_multi<<<1,1>>>(d_dist, --h_dist);
		}

                switch (NB_BFS)
                  {
#define CASE(X) case X:	backsum_virtual_multi<VtxType,X><<<grid3, threads3>>>(d_sources, d_d,  d_delta, d_sigma, d_bc, n_count); break;
                    CASE(1);CASE(2);CASE(3);CASE(4);CASE(5);CASE(6);CASE(7);CASE(8);
                    CASE(16);CASE(32);CASE(64);CASE(128);
                  default: assert(0);
#undef CASE
                  }


#ifdef TIMER
		cudaDeviceSynchronize();
		CudaCheckError();

		gettimeofday(&gt2, 0);
		time = (1000000.0*(gt2.tv_sec-gt1.tv_sec) + gt2.tv_usec-gt1.tv_usec)/1000000.0;
		cerr << "Phase 2 takes " << time << " secs"<<std::endl;
#endif

#ifdef ANALYSIS
		//analysis
		//std::cout<<"copying results for analysis"<<std::endl;
		assert (cudaSuccess == cudaMemcpy(h_d, d_d, sizeof(int)*n_count*NB_BFS, cudaMemcpyDeviceToHost));
		//std::cout<<"starting analysis"<<std::endl;
		for (int baseth=0; baseth < virn_count*NB_BFS; baseth += 32) { //32 is warpsize
#define MAX_LEVEL 64
		  int count[MAX_LEVEL];
		  for (int i=0; i< MAX_LEVEL; ++i) {
		    count[i] = 0;
		  }
		  for (int th = baseth; th < baseth+32; ++th) {
		    int vu = th/NB_BFS;
		    if (vu >= virn_count) continue;
		    int bfs = th%NB_BFS;
		    int off = h_vmap[vu]*NB_BFS+bfs;
		    if (h_d[off] > 0 && h_d[off] < MAX_LEVEL)
		      count[h_d[off]] ++;
		  }
		  for (int i=0; i< MAX_LEVEL; ++i) {
		    if (count[i] > 0)
		      active ++;
		  }
#undef MAX_LEVEL
		}
		//		std::cout<<"active warps: "<<active<<std::endl;

		for (int vu = 0; vu< virn_count; ++vu) {
		  for (int wa = 0; wa<NB_BFS/32 + ((NB_BFS%32)!=0); wa++) {
		    int minbfs = wa*32;
		    int maxbfs = (wa+1)*32;
		    maxbfs = std::min (NB_BFS, maxbfs);
#define MAX_LEVEL 64
		    int count[MAX_LEVEL];
		    for (int i=0; i< MAX_LEVEL; ++i) {
		      count[i] = 0;
		    }
		    for (int th = vu*NB_BFS+minbfs; th < vu*NB_BFS+maxbfs; ++th) {
		      if (vu >= virn_count) continue;
		      int bfs = th%NB_BFS;
		      int off = h_vmap[vu]*NB_BFS+bfs;
		      
		      if (h_d[off] > 0 && h_d[off] < MAX_LEVEL)
			count[h_d[off]] ++;
		    }
		    for (int i=0; i< MAX_LEVEL; ++i) {
		      if (count[i] > 0) {
			non_simu_traverse ++;
		      }
		    }
		  }
#undef MAX_LEVEL
		}
#endif
	}

	std::cout<<"active warps: "<<active<<std::endl;
	std::cout<<"non simultaneous virtual vertex traversal: "<<non_simu_traverse<<std::endl;

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
	return 0;
}



//explicit instanciation
template int bc_gpu_virtual_multi<int, int> (int* h_vmap, int* h_vptrs, int* h_vjs, int n_count, int e_count, int virn_count, int nb, float *h_bc);
template int bc_gpu_virtual_multi<int, long int> (int* h_vmap, long int* h_vptrs, int* h_vjs, int n_count, long int e_count, long int virn_count, int nb, float *h_bc);
template int bc_gpu_virtual_multi<long int, long int> (long int* h_vmap, long int* h_vptrs, long int* h_vjs, long int n_count, long int e_count, long int virn_count, long int nb, float *h_bc);
