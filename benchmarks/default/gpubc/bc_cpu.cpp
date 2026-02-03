
#include <omp.h>
#include <algorithm>


/**
 *
 * @param xadj first array of CRS representation. (offset in adj)
 * @param adj second array of CRS representation. (adjacency lists)
 * @param nVtx number of vertices
 * @param nb number of sources to compute
 * @param bc betwenness centrality values that are computed
 **/
template <typename VtxType, typename EdgeIndex>
void bc_cpu (EdgeIndex* xadj, VtxType* adj, VtxType nVtx, int , VtxType nb, float* bc) {
  typedef int PathCount;

  for (VtxType i = 0; i < nVtx; i++)
    bc[i] = 0.;

  int nthreads = omp_get_max_threads();
  // printf("no threads is %d\n", nthreads);
#pragma omp parallel
  {

    VtxType* bfsorder = new VtxType[nVtx];
    VtxType* Pred = new VtxType[xadj[nVtx]];
    EdgeIndex* endpred = new EdgeIndex[nVtx];
    VtxType* __restrict__ level = new VtxType[nVtx];
    PathCount* sigma = new PathCount[nVtx];
    float* delta = new float[nVtx];
    float* bclocal = new float[nVtx];
    for (VtxType i = 0; i < nVtx; i++)
      bclocal[i] = 0.;

#pragma omp for schedule(dynamic, 16)
    for (VtxType source = 0; source < std::min (nb, nVtx); source++) {
      int endofbfsorder = 1;
      bfsorder[0] = source;

      for (VtxType i = 0; i < nVtx; i++)
	endpred[i] = xadj[i];

      for (VtxType i = 0; i < nVtx; i++)
	level[i] = -2;
      level[source] = 0;

      for (VtxType i = 0; i < nVtx; i++)
	sigma[i] = 0;
      sigma[source] = 1;

      //step 1: build shortest path graph
      VtxType cur = 0;
      while (cur != endofbfsorder) {
	VtxType v = bfsorder[cur];
	for (EdgeIndex j = xadj[v]; j < xadj[v+1]; j++) {
	  VtxType w = adj[j];
	  if (level[w] < 0) {
	    level[w] = level[v]+1;
	    bfsorder[endofbfsorder++] = w;
	  }
	  if (level[w] == level[v]+1) {
	    sigma[w] += sigma[v];
	    //assert (sigma[w] > 0); //check for overflow
	    //assert (isfinite(sigma[w]));
	  }
	  else if (level[w] == level[v] - 1) {
	    Pred[endpred[v]++] = w;
	  }
	}
	cur++;
      }

      for (VtxType i = 0; i < nVtx; i++) {
	delta[i] = 0.;
      }

      //step 2: compute betweenness
      for (VtxType i = endofbfsorder - 1; i > 0; i--) {
	VtxType w = bfsorder[i];
	for (EdgeIndex j = xadj[w]; j < endpred[w]; j++) {
	  VtxType v = Pred[j];
	  delta[v] += (sigma[v] * (1 + delta[w])) / sigma[w];
	}
	bclocal[w] += delta[w];
      }
    }

    delete[] bfsorder;
    delete[] Pred;
    delete[] level;
    delete[] sigma;
    delete[] delta;
    delete[] endpred;

#pragma omp critical
    {
      for (int i=0; i<nVtx; ++i)
	bc[i] += bclocal[i];
    }
    delete[] bclocal;

  }

}

template <typename VtxType, typename EdgeIndex, typename PathCount>
void bc_cpu_deg1_one_source (EdgeIndex* xadj, VtxType* adj, VtxType nVtx,
			     int , float* bc, VtxType* weight, VtxType source, VtxType* bfsorder,
			     VtxType* Pred, EdgeIndex* endpred, VtxType* level, PathCount* sigma, float* delta ) {
  VtxType endofbfsorder = 1;
  bfsorder[0] = source;

  for (VtxType i = 0; i < nVtx; i++)
    endpred[i] = xadj[i];

  for (VtxType i = 0; i < nVtx; i++)
    level[i] = -2;
  level[source] = 0;

  for (VtxType i = 0; i < nVtx; i++)
    sigma[i] = 0;
  sigma[source] = 1;

  //step 1: build shortest path graph
  VtxType cur = 0;
  while (cur != endofbfsorder) {
    VtxType v = bfsorder[cur];
    for (EdgeIndex j = xadj[v]; j < xadj[v+1]; j++) {
      VtxType w = adj[j];
      if (level[w] < 0) {
	level[w] = level[v]+1;
	bfsorder[endofbfsorder++] = w;
      }
      if (level[w] == level[v]+1) {
	sigma[w] += sigma[v];
      }
      else if (level[w] == level[v] - 1) {
	Pred[endpred[v]++] = w;
      }
    }
    cur++;
  }

  for (VtxType i = 0; i < nVtx; i++) {
    delta[i] = weight[i] - 1;
  }

  //step 2: compute betweenness
  for (VtxType i = endofbfsorder - 1; i > 0; i--) {
    VtxType w = bfsorder[i];
    for (EdgeIndex j = xadj[w]; j < endpred[w]; j++) {
      VtxType v = Pred[j];
      delta[v] += (sigma[v] * (1 + delta[w])) / sigma[w];
    }
    bc[w] += delta[w] * weight[source];
  }

}

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
void bc_cpu_deg1 (EdgeIndex* xadj, VtxType* adj, VtxType nVtx, int , VtxType nb, float* bc, VtxType* weight) {
  typedef int PathCount;

  VtxType* bfsorder = new VtxType[nVtx];
  VtxType* Pred = new VtxType[xadj[nVtx]];
  EdgeIndex* endpred = new EdgeIndex[nVtx];
  VtxType* level = new VtxType[nVtx];
  PathCount* sigma = new PathCount[nVtx];
  float* delta = new float[nVtx];

  for (VtxType source = 0; source < std::min (nb, nVtx); source++) {
    bc_cpu_deg1_one_source (xadj, adj, nVtx, 0, bc, weight, source, bfsorder, Pred, endpred, level, sigma, delta);
  }

  delete[] bfsorder;
  delete[] Pred;
  delete[] level;
  delete[] sigma;
  delete[] delta;
  delete[] endpred;
}


//forcing instanciation of most common typed graph
template void bc_cpu_deg1<int, int> (int* xadj, int* adj, int nVtx, int , int nb, float* bc, int* weight);
template void bc_cpu_deg1<int, long int> (long int* xadj, int* adj, int nVtx, int , int nb, float* bc, int* weight);
template void bc_cpu_deg1<long int, long int> (long int* xadj, long int* adj, long int nVtx, int , long int nb, float* bc, long int* weight);


template void bc_cpu<int, int> (int* xadj, int* adj, int nVtx, int , int nb, float* bc);
template void bc_cpu<int, long int> (long int* xadj, int* adj, int nVtx, int , int nb, float* bc);
template void bc_cpu<long int, long int> (long int* xadj, long int* adj, long int nVtx, int , long int nb, float* bc);


