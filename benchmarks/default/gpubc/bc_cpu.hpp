#ifndef BC_CPU_HPP
#define BC_CPU_HPP


/**
 *
 * @param xadj first array of CRS representation. (offset in adj)
 * @param adj second array of CRS representation. (adjacency lists)
 * @param nVtx number of vertices
 * @param nb number of sources to compute
 * @param bc betwenness centrality values that are computed
 **/
template <typename VtxType, typename EdgeIndex>
void bc_cpu (EdgeIndex* xadj, VtxType* adj, VtxType nVtx, int , VtxType nb, float* bc);


/**
 *
 * @param xadj first array of CRS representation. (offset in adj)
 * @param adj second array of CRS representation. (adjacency lists)
 * @param nVtx number of vertices
 * @param nb number of sources to compute
 * @param weight representatino count (caused by degree 1 removal)
 * @param bc betwenness centrality values that are computed
 **/
template <typename VtxType, typename EdgeIndex>
void bc_cpu_deg1 (EdgeIndex* xadj, VtxType* adj, VtxType nVtx, int , VtxType nb, float* bc, VtxType* weight);


#endif
