#include <iostream>
#include <stdlib.h>


void undefined(){std::cerr<<"UNDEFINED"<<std::endl; exit (-1);}

template <typename VtxType, typename EdgeIndex>
void cc_mic (EdgeIndex* xadj, VtxType* adj, VtxType nVtx, VtxType nb, float* cc)
{undefined();}

template 
void cc_mic<int, int> (int* xadj, int* adj, int nVtx, int nb, float* cc);

template
void cc_mic<int, long int> (long int* xadj, int* adj, int nVtx, int nb, float* cc);

template
void cc_mic<long int, long int> (long int* xadj, long int* adj, long int nVtx, long int nb, float* cc);


template <typename VtxType, typename EdgeIndex>
void cc_mic_hybrid (EdgeIndex* xadj, VtxType* adj, VtxType nVtx, VtxType nb, float* cc)
{undefined();}

template
void cc_mic_hybrid<int, int> (int* xadj, int* adj, int nVtx, int nb, float* cc);
template
void cc_mic_hybrid<int, long int> (long int* xadj, int* adj, int nVtx, int nb, float* cc);
template
void cc_mic_hybrid<long int, long int> (long int* xadj, long int* adj, long int nVtx, long int nb, float* cc);


template <typename VtxType, typename EdgeIndex>
void cc_mic_spmv_soft_vec_opt (EdgeIndex* xadj, VtxType* adj, VtxType nVtx, VtxType nb, float* cc)
{undefined();}

template
void cc_mic_spmv_soft_vec_opt <int, int> (int* xadj, int* adj, int nVtx, int nb, float* cc);
template
void cc_mic_spmv_soft_vec_opt <int, long int> (long int* xadj, int* adj, int nVtx, int nb, float* cc);
template
void cc_mic_spmv_soft_vec_opt <long int, long int> (long int* xadj, long int* adj, long int nVtx, long int nb, float* cc);
