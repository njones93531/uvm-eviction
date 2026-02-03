#ifndef CC_GPU_HPP
#define CC_GPU_HPP

template <typename VtxType, typename EdgeIndex>
int cc_gpu_virtual(VtxType* h_vmap, EdgeIndex* h_vptrs, VtxType* h_vjs, VtxType n_count, EdgeIndex e_count, EdgeIndex virn_count, VtxType nb, float *h_cc);

template <typename VtxType, typename EdgeIndex>
int cc_gpu_vertex_multi_spmv (EdgeIndex* h_vptrs, VtxType* h_vjs, VtxType n, VtxType nb, float *h_cc);

#endif
