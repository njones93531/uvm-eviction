#ifndef BC_GPU_HPP
#define BC_GPU_HPP

template <typename VtxType, typename EdgeIndex>
int bc_gpu_vertex_deg1 (EdgeIndex *h_ptrs, VtxType* h_js, VtxType n_count, EdgeIndex e_count, VtxType nb, float *h_bc, VtxType* h_weight);

template <typename VtxType, typename EdgeIndex>
int bc_gpu_virtual_deg1(VtxType* h_vmap, EdgeIndex* h_vptrs, VtxType* h_vjs, VtxType n_count, EdgeIndex e_count, EdgeIndex virn_count, VtxType nb, float *h_bc, VtxType* weight);

template <typename VtxType, typename EdgeIndex>
int bc_gpu_vertex(EdgeIndex *h_ptrs, VtxType* h_js, VtxType n_count, EdgeIndex e_count, VtxType nb, float *h_bc);

template <typename VtxType, typename EdgeIndex>
int bc_gpu_vertex_multi(EdgeIndex *h_ptrs, VtxType* h_js, VtxType n_count, EdgeIndex e_count, VtxType nb, float *h_bc);

template <typename VtxType, typename EdgeIndex>
int bc_gpu_edge(VtxType* h_v, VtxType *h_e, VtxType n_count, EdgeIndex e_count, VtxType nb, float *h_bc);

template <typename VtxType, typename EdgeIndex>
int bc_gpu_edge_multi(VtxType* h_v, VtxType *h_e, VtxType n_count, EdgeIndex e_count, VtxType nb, float *h_bc);

template <typename VtxType, typename EdgeIndex>
int bc_gpu_virtual(VtxType* h_vmap, EdgeIndex* h_vptrs, VtxType* h_vjs, VtxType n_count, EdgeIndex e_count, EdgeIndex virn_count, VtxType nb, float *h_bc);

template <typename VtxType, typename EdgeIndex>
int bc_gpu_virtual_multi(VtxType* h_vmap, EdgeIndex* h_vptrs, VtxType* h_vjs, VtxType n_count, EdgeIndex e_count, EdgeIndex virn_count, VtxType nb, float *h_bc);

template <typename VtxType, typename EdgeIndex>
int bc_gpu_virtual_coalesced (VtxType* h_vmap, EdgeIndex* h_xadj, VtxType* h_vjs, VtxType n_count, EdgeIndex* h_startoffset, VtxType* h_stride, EdgeIndex e_count, EdgeIndex virn_count, VtxType nb, float *h_bc);

#endif

