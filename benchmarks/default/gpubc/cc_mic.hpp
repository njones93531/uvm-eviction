#include <stdio.h>
#include "offload.h"


#pragma offload_attribute(push,target(mic))
#include "cc-cpu.cpp"
#include <omp.h>
#include <algorithm>
#include "immintrin.h"
#include "timestamp.hpp"

#define TIMER

#pragma offload_attribute(pop)


//make sure the calling xadj, adj and cc are memory aligned on 64B
template <typename VtxType, typename EdgeIndex>
void cc_mic (EdgeIndex* xadj, VtxType* adj, VtxType nVtx, VtxType nb, float* cc) {

#pragma offload target(mic) in(n), in(nb), in(xadj:length(n+1)), in(adj:length(xadj[n])), inout(cc:length(n))
  {
    util::timestamp t1;
    cc_cpu(xadj, adj, n, xadj[n], nb, cc);
    util::timestamp t2;
    printf ("cc-only time: %lf s\n",(double)(t2-t1));
  }
}

template 
void cc_mic<int, int> (int* xadj, int* adj, int nVtx, int nb, float* cc);
template 
void cc_mic<int, long int> (long int* xadj, int* adj, int nVtx, int nb, float* cc);


//make sure the calling xadj, adj and cc are memory aligned on 64B
template <typename VtxType, typename EdgeIndex>
void cc_mic_hybrid (EdgeIndex* xadj, VtxType* adj, VtxType nVtx, VtxType nb, float* cc) {
#pragma offload target(mic) in(n), in(nb), in(xadj:length(n+1)), in(adj:length(xadj[n])), inout(cc:length(n))
  {
    util::timestamp t1;
    cc_cpu_hybrid(xadj, adj, n, nb, cc);
    util::timestamp t2;
    printf ("cc-only time: %lf s\n",(double)(t2-t1));
  }
}

template
void cc_mic_hybrid <int, int> (int* xadj, int* adj, int nVtx, int nb, float* cc);
template
void cc_mic_hybrid <int, long int> (long int* xadj, int* adj, int nVtx, int nb, float* cc);


template <typename VtxType, typename EdgeIndex>
void cc_mic_spmv_soft_vec_opt (EdgeIndex* xadj, VtxType* adj, VtxType nVtx, VtxType nb, float* cc) {
#pragma offload target(mic) in(n), in(nb), in(xadj:length(n+1)), in(adj:length(xadj[n])), inout(cc:length(n))
  {
    util::timestamp t1;
    cc_cpu_spmv_soft_vec_opt(xadj, adj, n, nb, cc);
    util::timestamp t2;
    printf ("cc-only time: %lf s\n",(double)(t2-t1));
  }
}

template 
void cc_mic_spmv_soft_vec_opt<int, int> (int* xadj, int* adj, int nVtx, int nb, float* cc);
template
void cc_mic_spmv_soft_vec_opt <int, long int> (long int* xadj, int* adj, int nVtx, int nb, float* cc);
