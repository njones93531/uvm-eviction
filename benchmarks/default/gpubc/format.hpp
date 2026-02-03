#ifndef FORMAT_GRAPH_HPP
#define FORMAT_GRAPH_HPP


template <typename VtxType, typename EdgeIndex>
EdgeIndex createVirtualCSR(EdgeIndex* ptrs, VtxType* js, VtxType nov, VtxType* vmap, EdgeIndex* virptrs, VtxType maxload, bool permuteAdj);

template <typename VtxType, typename EdgeIndex>
EdgeIndex createVirtualCoalescedCSR(EdgeIndex* ptrs, VtxType* js, VtxType nov, VtxType* vmap, EdgeIndex* virptrs, EdgeIndex* startoffset, VtxType* stride, VtxType maxload, bool permuteAdj);

template <typename VtxType, typename EdgeIndex>
void order_graph (EdgeIndex* xadj, VtxType* adj, VtxType* weight, float* bc, VtxType n, VtxType vcount, int deg1, VtxType* map_for_order, VtxType* reverse_map_for_order);





#endif
