#include <iostream>
#include <stdlib.h>
#include "debug.hpp"

void undefined_gpu() {
  std::cerr<<"UNDEFINED"<<std::endl;
  print_backtrace();
  exit (-1);
}

void init(){
#pragma offload target(mic)
   {
     size_t size = 64*1024L*1024L;
     char* p = (char*)malloc(size);
     #pragma omp parallel
#pragma omp parallel for
     for (size_t i = 0; i<size; ++i) {
	 p[i] = 0;
     }
     free(p);
   }
}

template <typename VtxType, typename EdgeIndex>
int cc_gpu_virtual(VtxType* h_vmap, EdgeIndex* h_vptrs, VtxType* h_vjs, VtxType n_count, EdgeIndex e_count, EdgeIndex virn_count, VtxType nb, float *h_cc)
{undefined_gpu();return 0;}  

template <typename VtxType, typename EdgeIndex>
int cc_gpu_vertex_multi_spmv (EdgeIndex* h_vptrs, VtxType* h_vjs, VtxType n, VtxType nb, float *h_cc)
{undefined_gpu();return 0;}  

template <typename VtxType, typename EdgeIndex>
int bc_gpu_vertex_deg1 (EdgeIndex *h_ptrs, VtxType* h_js, VtxType n_count, EdgeIndex e_count, VtxType nb, float *h_bc, VtxType* h_weight){undefined_gpu();return 0;}

template <typename VtxType, typename EdgeIndex>
int bc_gpu_virtual_deg1(VtxType* h_vmap, EdgeIndex* h_vptrs, VtxType* h_vjs, VtxType n_count, EdgeIndex e_count, EdgeIndex virn_count, VtxType nb, float *h_bc, VtxType* weight){undefined_gpu();return 0;}

template <typename VtxType, typename EdgeIndex>
int bc_gpu_vertex(EdgeIndex *h_ptrs, VtxType* h_js, VtxType n_count, EdgeIndex e_count, VtxType nb, float *h_bc){undefined_gpu();return 0;}

template <typename VtxType, typename EdgeIndex>
int bc_gpu_vertex_multi(EdgeIndex *h_ptrs, VtxType* h_js, VtxType n_count, EdgeIndex e_count, VtxType nb, float *h_bc){undefined_gpu();return 0;}

template <typename VtxType, typename EdgeIndex>
int bc_gpu_edge(VtxType* h_v, VtxType *h_e, VtxType n_count, EdgeIndex e_count, VtxType nb, float *h_bc){undefined_gpu();return 0;}

template <typename VtxType, typename EdgeIndex>
int bc_gpu_edge_multi(VtxType* h_v, VtxType *h_e, VtxType n_count, EdgeIndex e_count, VtxType nb, float *h_bc){undefined_gpu();return 0;}

template <typename VtxType, typename EdgeIndex>
int bc_gpu_virtual(VtxType* h_vmap, EdgeIndex* h_vptrs, VtxType* h_vjs, VtxType n_count, EdgeIndex e_count, EdgeIndex virn_count, VtxType nb, float *h_bc){undefined_gpu();return 0;}

template <typename VtxType, typename EdgeIndex>
int bc_gpu_virtual_multi(VtxType* h_vmap, EdgeIndex* h_vptrs, VtxType* h_vjs, VtxType n_count, EdgeIndex e_count, EdgeIndex virn_count, VtxType nb, float *h_bc){undefined_gpu();return 0;}

template <typename VtxType, typename EdgeIndex>
int bc_gpu_virtual_coalesced (VtxType* h_vmap, EdgeIndex* h_xadj, VtxType* h_vjs, VtxType n_count, EdgeIndex* h_startoffset, VtxType* h_stride, EdgeIndex e_count, EdgeIndex virn_count, VtxType nb, float *h_bc){undefined_gpu();return 0;}

template<typename VtxType, typename EdgeIndex>
int preprocess(EdgeIndex *xadj, VtxType* adj, VtxType* tadj, VtxType *np, float* bc, VtxType* weight, VtxType* map_for_order, VtxType* reverse_map_for_order, FILE* ofp)
{undefined_gpu(); return 0;}


void order_graph (int* xadj, int* adj, int* weight, float* bc, int n, int vcount, int deg1, int* map_for_order, int* reverse_map_for_order){undefined_gpu();}


//explicit instanciation
//int, int
template 
int cc_gpu_virtual<int,  int>(int* h_vmap, int* h_vptrs, int* h_vjs, int n_count, int e_count, int virn_count, int nb, float *h_cc);  

template
int cc_gpu_vertex_multi_spmv <int,  int>(int* h_vptrs, int* h_vjs, int n, int nb, float *h_cc);  

template
int bc_gpu_vertex_deg1<int,  int> (int *h_ptrs, int* h_js, int n_count, int e_count, int nb, float *h_bc, int* h_weight);

template
int bc_gpu_virtual_deg1<int,  int>(int* h_vmap, int* h_vptrs, int* h_vjs, int n_count, int e_count, int virn_count, int nb, float *h_bc, int* weight);

template
int bc_gpu_vertex<int,  int>(int *h_ptrs, int* h_js, int n_count, int e_count, int nb, float *h_bc);

template
int bc_gpu_vertex_multi<int,  int>(int *h_ptrs, int* h_js, int n_count, int e_count, int nb, float *h_bc);

template
int bc_gpu_edge<int,  int>(int* h_v, int *h_e, int n_count, int e_count, int nb, float *h_bc);

template
int bc_gpu_edge_multi<int,  int>(int* h_v, int *h_e, int n_count, int e_count, int nb, float *h_bc);

template
int bc_gpu_virtual<int,  int>(int* h_vmap, int* h_vptrs, int* h_vjs, int n_count, int e_count, int virn_count, int nb, float *h_bc);

template
int bc_gpu_virtual_multi<int,  int>(int* h_vmap, int* h_vptrs, int* h_vjs, int n_count, int e_count, int virn_count, int nb, float *h_bc);

template
int bc_gpu_virtual_coalesced <int,  int>(int* h_vmap, int* h_xadj, int* h_vjs, int n_count, int* h_startoffset, int* h_stride, int e_count, int virn_count, int nb, float *h_bc);

template
int preprocess<int, int>(int *xadj, int* adj, int* tadj, int *np, float* bc, int* weight, int* map_for_order, int* reverse_map_for_order, FILE* ofp);

//long int long int

template
int cc_gpu_virtual<long int, long int>(long int* h_vmap, long int* h_vptrs, long int* h_vjs, long int n_count, long int e_count, long int virn_count, long int nb, float *h_cc);  

template
int cc_gpu_vertex_multi_spmv<long int, long int> (long int* h_vptrs, long int* h_vjs, long int n, long int nb, float *h_cc);  

template
int bc_gpu_vertex_deg1<long int, long int> (long int *h_ptrs, long int* h_js, long int n_count, long int e_count, long int nb, float *h_bc, long int* h_weight);

template
int bc_gpu_virtual_deg1<long int, long int>(long int* h_vmap, long int* h_vptrs, long int* h_vjs, long int n_count, long int e_count, long int virn_count, long int nb, float *h_bc, long int* weight);

template
int bc_gpu_vertex<long int, long int>(long int *h_ptrs, long int* h_js, long int n_count, long int e_count, long int nb, float *h_bc);

template
int bc_gpu_vertex_multi<long int, long int>(long int *h_ptrs, long int* h_js, long int n_count, long int e_count, long int nb, float *h_bc);

template
int bc_gpu_edge<long int, long int>(long int* h_v, long int *h_e, long int n_count, long int e_count, long int nb, float *h_bc);

template
int bc_gpu_edge_multi<long int, long int>(long int* h_v, long int *h_e, long int n_count, long int e_count, long int nb, float *h_bc);

template
int bc_gpu_virtual<long int, long int>(long int* h_vmap, long int* h_vptrs, long int* h_vjs, long int n_count, long int e_count, long int virn_count, long int nb, float *h_bc);

template
int bc_gpu_virtual_multi<long int, long int>(long int* h_vmap, long int* h_vptrs, long int* h_vjs, long int n_count, long int e_count, long int virn_count, long int nb, float *h_bc);

template
int bc_gpu_virtual_coalesced<long int, long int> (long int* h_vmap, long int* h_xadj, long int* h_vjs, long int n_count, long int* h_startoffset, long int* h_stride, long int e_count, long int virn_count, long int nb, float *h_bc);

template
int preprocess<long int, long int>(long int *xadj, long int* adj, long int* tadj, long int *np, float* bc, long int* weight, long int* map_for_order, long int* reverse_map_for_order, FILE* ofp);

//int long int

template 
int cc_gpu_virtual<int,  long int>(int* h_vmap, long int* h_vptrs, int* h_vjs, int n_count, long int e_count, long int virn_count, int nb, float *h_cc);  

template 
int cc_gpu_vertex_multi_spmv<int,  long int> (long int* h_vptrs, int* h_vjs, int n, int nb, float *h_cc);  

template 
int bc_gpu_vertex_deg1<int,  long int> (long int *h_ptrs, int* h_js, int n_count, long int e_count, int nb, float *h_bc, int* h_weight);

template 
int bc_gpu_virtual_deg1<int,  long int>(int* h_vmap, long int* h_vptrs, int* h_vjs, int n_count, long int e_count, long int virn_count, int nb, float *h_bc, int* weight);

template 
int bc_gpu_vertex<int,  long int>(long int *h_ptrs, int* h_js, int n_count, long int e_count, int nb, float *h_bc);

template 
int bc_gpu_vertex_multi<int,  long int>(long int *h_ptrs, int* h_js, int n_count, long int e_count, int nb, float *h_bc);

template 
int bc_gpu_edge<int,  long int> (int* h_v, int *h_e, int n_count, long int e_count, int nb, float *h_bc);

template 
int bc_gpu_edge_multi<int,  long int>(int* h_v, int *h_e, int n_count, long int e_count, int nb, float *h_bc);

template 
int bc_gpu_virtual<int,  long int>(int* h_vmap, long int* h_vptrs, int* h_vjs, int n_count, long int e_count, long int virn_count, int nb, float *h_bc);

template 
int bc_gpu_virtual_multi<int,  long int>(int* h_vmap, long int* h_vptrs, int* h_vjs, int n_count, long int e_count, long int virn_count, int nb, float *h_bc);

template 
int bc_gpu_virtual_coalesced<int,  long int> (int* h_vmap, long int* h_xadj, int* h_vjs, int n_count, long int* h_startoffset, int* h_stride, long int e_count, long int virn_count, int nb, float *h_bc);

template
int preprocess<int,  long int>(long int *xadj, int* adj, int* tadj, int *np, float* bc, int* weight, int* map_for_order, int* reverse_map_for_order, FILE* ofp);
