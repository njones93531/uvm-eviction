#ifndef CC_CPU_HPP
#define CC_CPU_HPP

template <typename VtxType, typename EdgeIndex>
void cc_mic (EdgeIndex* xadj, VtxType* adj, VtxType nVtx, VtxType nb, float* cc);
template <typename VtxType, typename EdgeIndex>
void cc_mic_hybrid (EdgeIndex* xadj, VtxType* adj, VtxType nVtx, VtxType nb, float* cc);

template <typename VtxType, typename EdgeIndex>
void cc_cpu (EdgeIndex* xadj, VtxType* adj, VtxType nVtx, VtxType nb, float* cc);
template <typename VtxType, typename EdgeIndex>
void cc_cpu_hybrid (EdgeIndex* xadj, VtxType* adj, VtxType nVtx, VtxType nb, float* cc);

template <typename VtxType, typename EdgeIndex>
void cc_cpu_spmv_soft_vec_opt (EdgeIndex* xadj, VtxType* adj, VtxType nVtx, VtxType nb, float* cc);
template <typename VtxType, typename EdgeIndex>
void cc_mic_spmv_soft_vec_opt (EdgeIndex* xadj, VtxType* adj, VtxType nVtx, VtxType nb, float* cc);

#endif
