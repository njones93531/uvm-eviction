#include <iostream>
#include "config.h"
#include <omp.h>
#include "util.h"
#include <string.h>
#include <stdio.h>
#include "timestamp.hpp"

//#define FINETIMER
//#define FINETIMER_VERBOSE

#if defined FINETIMER_VERBOSE and not defined FINETIMER
#error defined FINETIMER_VERBOSE and not defined FINETIMER
#endif

template <typename VtxType, typename EdgeIndex>
void cc_cpu  (EdgeIndex* xadj, VtxType* adj, VtxType nVtx, VtxType nb, float* cc)
{
    int nthreads = omp_get_max_threads();
    nb = std::min (nb, nVtx);
    printf("no threads is %d\n", nthreads);
#pragma omp parallel shared(cc) 
    {
        int tid = omp_get_thread_num();
        VtxType* bfsorder = new VtxType[nVtx];
        VtxType* level = new VtxType[nVtx];

        if (bfsorder == NULL || level == NULL) {
            printf ("oom in cc_cpu\n");
            exit(1);
        }

#pragma omp for schedule(dynamic, 4)
        for (VtxType source = 0; source < nb; source++) {
            //if(source % 10 == 0) printf("%d out of %d thead num %d\n", source, nVtx, tid);

            float farness = 0.0f;
            VtxType endofbfsorder = 1;
            bfsorder[0] = source;

            memset(level, 0xff, sizeof(int)*nVtx);
            level[source] = 0;

            //step 1: build shortest path graph
            VtxType cur = 0;
            while (cur != endofbfsorder) {
                VtxType v = bfsorder[cur];
                for (EdgeIndex j = xadj[v]; j < xadj[v + 1]; ++j) {
                    VtxType w = adj[j];
                    if (level[w] < 0) {
                        level[w] = level[v]+1;
                        bfsorder[endofbfsorder++] = w;
                        farness += level[w];
                    }
                }
                cur++;
            }

            cc[source] = 1.0f / farness;
        }

        delete[] bfsorder;
        delete[] level;
    }

    return;
}

template
void cc_cpu <int, int>  (int* xadj, int* adj, int nVtx, int nb, float* cc);
template
void cc_cpu <int, long int>  (long int* xadj, int* adj, int nVtx, int nb, float* cc);
template
void cc_cpu <long int, long int>  (long int* xadj, long int* adj, long int nVtx, long int nb, float* cc);

//this version tries to take advantage of the value of visited[] to skip some computations
template<typename VtxType, typename EdgeIndex, int vector_size>
void cc_cpu_spmv_soft_vec_opt_t (EdgeIndex* __restrict__ xadj, VtxType* __restrict__ adj, VtxType n, VtxType nb, float* __restrict__ cc)
{
    int NB_BFS = vector_size * 32;
    size_t size_alloc = n;
    size_alloc *= NB_BFS / 8;
    int n_align = NB_BFS / 8;

#ifdef FINETIMER
    util::timestamp ft1;
#endif
    int* __restrict__ neighbor = (int*)_mm_malloc(size_alloc, n_align);
    int* __restrict__ current = (int*)_mm_malloc(size_alloc, n_align);
    int* __restrict__ visited = (int*)_mm_malloc(size_alloc, n_align);

    if (neighbor == NULL) {printf("memory allocation failed. Not enough memory?\n");}
    if (current == NULL) {printf("memory allocation failed. Not enough memory?\n");}
    if (visited == NULL) {printf("memory allocation failed. Not enough memory?\n");}

    nb = std::min(n, nb);

#ifdef FINETIMER
    util::timestamp init(0,0);
    util::timestamp spmm(0,0);
    util::timestamp update(0,0);

    long spmm_traversed = 0;
    long update_traversed = 0;

#pragma omp parallel for schedule (dynamic, CC_CHUNK)
    for (int i = 0; i < n; ++i) {
#pragma unroll
      for (int k=0; k< vector_size; ++k) {                                                     
        current[i*vector_size+k] = 0;  
        neighbor[i*vector_size+k] = 0;
        visited[i*vector_size+k] = 0; 
      }
    }
    
    util::timestamp ft2;
    std::cerr<<"memalloc + first touch: "<<ft2-ft1<<" seconds"<<std::endl; 
#endif

    for (VtxType s = 0; s < nb; s += NB_BFS) {
#ifdef FINETIMER
      util::timestamp t1;
#endif
        //initialize bfs traversals
#pragma omp parallel for schedule (dynamic, CC_CHUNK)
        for (VtxType i = 0; i < n; ++i) {
            int cu[vector_size];
#pragma unroll
            for (int j = 0; j < vector_size; j++)
                cu[j] = 0;

            if (i >= s && i < s + NB_BFS && i < nb) {
                int reli = i - s;
                int which_int = reli / 32;
                int which_bit = reli % 32;
                cu[which_int] = 1 << which_bit;
            }

#pragma unroll
            for (int k=0; k< vector_size; ++k) 
              current[i*vector_size+k] = cu[k];

#pragma unroll
            for (int k=0; k< vector_size; ++k)
              visited[i*vector_size+k] = cu[k];
        }
#ifdef FINETIMER
      util::timestamp t2;
      init += t2-t1;
#ifdef FINETIMER_VERBOSE
      std::cerr<<std::endl<<"init: "<<t2-t1<<" seconds"<<std::endl;
#endif
#endif
        bool cont = true;
        VtxType level = 0;
        while (cont ) {
            cont = false;
            ++level;

            float flevel = 1.0f / (float)level;
#ifdef FINETIMER
            util::timestamp t3;
            long spmm_traversed_local = 0;
#endif
            //PERFORM: neighbor = spmv current
#ifdef FINETIMER
#pragma omp parallel for schedule (dynamic,CC_CHUNK) reduction(+:spmm_traversed_local)
#else
#pragma omp parallel for schedule (dynamic,CC_CHUNK)
#endif
            for (VtxType i=0; i< n; ++i) {
              //check for skip
              bool cont = true;
#pragma unroll
              for (int k = 0; k < vector_size; k++)
                if (visited[i*vector_size+k] != 0xFFFFFFFF) {
                  cont = false;
                }
              if (cont) continue;
              
#ifdef FINETIMER
              spmm_traversed_local ++;
#endif
              
              int vali[vector_size];
              
              //#pragma unroll
#pragma unroll
              for (int k=0; k<vector_size; ++k)
                vali[k] = 0;
              
              //for all neighbor
#pragma unroll
              for (EdgeIndex j = xadj[i]; j<xadj[i+1]; ++j) {
                VtxType v = adj[j];
                
#pragma unroll
                for (int k = 0; k < vector_size; k++)
                  vali[k] = vali[k] | current[v*vector_size+k];
              }
              
#pragma unroll
              for (int k=0; k<vector_size; ++k)
                neighbor[i*vector_size+k] = vali[k];
            }

#ifdef FINETIMER
            util::timestamp t4;
            spmm+=t4-t3;
            spmm_traversed += spmm_traversed_local;
#ifdef FINETIMER_VERBOSE
            std::cerr<<"spmm: "<<t4-t3<<" seconds"<<std::endl;
            std::cerr<<"spmm_traversed_local: "<<spmm_traversed_local<<std::endl;
#endif

            long update_traversed_local = 0;
#endif

#ifdef FINETIMER
#pragma omp parallel for schedule (dynamic,CC_CHUNK) reduction(+:update_traversed_local)
#else
#pragma omp parallel for schedule (dynamic,CC_CHUNK)
#endif
            for (VtxType i=0; i< n; ++i) {
              //check for skip
              bool conti = true;
              for (int k = 0; k < vector_size; k++)
                if (visited[i*vector_size+k] != 0xFFFFFFFF) {
                  conti = false;
                }

              if (conti) { 
#pragma unroll
                for (int k=0; k<vector_size; ++k)
                  ((int*)current)[i*vector_size+k] = 0;
                
                continue;
              }
              //current = neighbor - visited            
              int cu[vector_size];
#pragma unroll
              for (int k = 0; k < vector_size; k++) 
                cu[k] = neighbor[i*vector_size+k] & ~visited[i*vector_size+k]; //in other word: current[u] = neighbor[u] & (~visited[u]);
              
              
              {
                bool conti = true;
                for (int k = 0; k < vector_size; k++)
                  if (((int*)cu)[k] != 0) {
                    conti = false;
                  }
                if (conti) { 
#pragma unroll
                  for (int k=0; k<vector_size; ++k)
                    current[i*vector_size+k] = 0;
                  
                  continue;
                }
                
              }
#ifdef FINETIMER
              update_traversed_local ++;
#endif

              //visited = visited + current

#pragma unroll
              for (int k = 0; k < vector_size; k++)
                visited[i*vector_size+k] = cu[k] | visited[i*vector_size+k];
              
              int bcount = 0;
#pragma unroll
              for (int k = 0; k < vector_size; k++) {
                bcount += BitCount32(cu[k]);
              }
              
#pragma unroll
              for (int k=0; k<vector_size; ++k)
                current[i*vector_size+k] = cu[k];
              
              //accumulate to cc
              if (bcount > 0) {
                cc[i] += bcount * flevel;//symetric assumption
                cont = 1;
              }
            }
#ifdef FINETIMER
            util::timestamp t5;
            update+=t5-t4;
            update_traversed += update_traversed_local;
#ifdef FINETIMER_VERBOSE
            std::cerr<<"update: "<<t5-t4<<" seconds"<<std::endl;
            std::cerr<<"update_traversed_local: "<<update_traversed_local<<std::endl;
#endif
#endif
        }
    }

#ifdef FINETIMER
    std::cerr<<std::endl;
    std::cerr<<"init: "<<init<<" seconds"<<std::endl;
    std::cerr<<"spmm: "<<spmm<<" seconds"<<std::endl;
    std::cerr<<"update: "<<update<<" seconds"<<std::endl;

    std::cerr<<"update_traversed:"<<update_traversed<<std::endl;
    std::cerr<<"spmm_traversed:"<<spmm_traversed<<std::endl;
#endif
    _mm_free(neighbor);
    _mm_free(current);
    _mm_free(visited);
}


// spmv-based cc calculation with software vectorization (VEC_SIZE is in 32k bits)
//this version tries to take advantage of the value of visited to skip some operations.
template <typename VtxType, typename EdgeIndex>
void  cc_cpu_spmv_soft_vec_opt (EdgeIndex* xadj, VtxType* adj, VtxType nVtx, VtxType nb, float* cc) 
{
    int vector_size;
    {
        char* str = getenv("VEC_SIZE");
        if (str != NULL)
            vector_size = atoi(str) / 32;
        else {
          vector_size = 1;
          printf("defaulting to 32 BFS at a time\n");
        }
        printf ("VEC_SIZE=%d\n",32*vector_size);
    }
    

    switch (vector_size)
      {
#define CASE(X) case X: cc_cpu_spmv_soft_vec_opt_t<VtxType,EdgeIndex,X>(xadj, adj, nVtx, nb, cc); break;
        CASE(1);CASE(2);CASE(3);CASE(4);CASE(5);CASE(6);CASE(7);CASE(8);
        CASE(16);CASE(32);CASE(64);CASE(128);CASE(256);CASE(512);CASE(1024);CASE(2048);CASE(4096);
      default: 
        printf ("unsupported value of VEC_SIZE\n");
        exit (1);
#undef CASE
      }
}

template 
void cc_cpu_spmv_soft_vec_opt <int, int> (int* xadj, int* adj, int nVtx, int nb, float* cc);
template 
void cc_cpu_spmv_soft_vec_opt <int, long int> (long int* xadj, int* adj, int nVtx, int nb, float* cc);
template 
void cc_cpu_spmv_soft_vec_opt <long int, long int> (long int* xadj, long int* adj, long int nVtx, long int nb, float* cc);

template <typename VtxType, typename EdgeIndex>
void cc_cpu_hybrid (EdgeIndex* xadj, VtxType* adj, VtxType nVtx, VtxType nb, float* cc) {

    int nthreads = omp_get_max_threads();
#pragma omp parallel 
    {
      EdgeIndex no_edges_remaining;
      EdgeIndex no_edges_frontier;
      VtxType norem;

      EdgeIndex eup;
      VtxType cl1;

        int tid = omp_get_thread_num();
        VtxType* que = new VtxType[nVtx];
        VtxType* level = new VtxType[nVtx];
        VtxType* remaining = new VtxType[nVtx];

#pragma omp for schedule(dynamic, 4)
        for (VtxType source = 0; source < std::min (nb, nVtx); source++) {
            for (VtxType i = 0; i < nVtx; i++) {
                level[i] = -1;
                remaining[i] = i;
            }
            level[source] = 0;
            eup = 0;
            que[0] = source;
            VtxType quep = 1;
            EdgeIndex farness = 0; //This can be as large as V^2 so EdgeIndex is probably more appropriate


            norem = nVtx; /* no of unvisited vertices */
            no_edges_remaining = xadj[nVtx] - (xadj[source+1] - xadj[source]); /* number of unvisited edges */
            no_edges_frontier = 0; /* #edges in the frontier */
            for (EdgeIndex j = xadj[source]; j < xadj[source + 1]; j++) { /* this loop visits the neigbors of the source and process its edges*/
                VtxType v = adj[j];
                level[v] = 1;
                que[quep++] = v;
                eup += xadj[v+1] - xadj[v]; /* this amount will be processed in the next iteration */
            }
            farness += xadj[source+1] - xadj[source];

            VtxType cur = 1;
            while (cur != quep) { /* que loop */
                bool flag = false; /* still processing the previous level */
                VtxType clevel = level[que[cur]]; /* current level */
                cl1 = clevel + 1;  /* next level */
                if (clevel == level[que[cur-1]] + 1) { /* if levels are changing */
                    no_edges_frontier = eup; /* the number of edges that will be processed next */
                    no_edges_remaining -= eup; /*we reduce the amount from the total remaining edges */
                    eup = 0;
                    flag = true;  /* next level is reached */
                }

                /* when the number of remainin edges is small we do bottom up */
                /* THIS 0.7 IS NOT CERTAIN, ONE CAN PLAY WITH IT */
                if ((no_edges_frontier > 0.70 * no_edges_remaining ) && flag) {
                    cur = quep;
                    VtxType remp = 0;
                    for (VtxType i = 0; i < norem; i++) {
                        VtxType v = remaining[i];
                        if (level[v] == -1) {
                            for (EdgeIndex j = xadj[v]; j < xadj[v + 1]; j++) {
                                VtxType w = adj[j];
                                if (level[w] == clevel) {
                                    level[v] = cl1;
                                    farness += cl1;
                                    que[quep++] = v;
                                    eup += xadj[v+1] - xadj[v];
                                    break;
                                }
                            }
                            if (level[v] == -1) {
                                remaining[remp++] = v;
                            }
                        }
                    }
                    norem = remp;
                } else { /* when a small number of edges in the frontier we do top down */
                    VtxType v = que[cur++];
                    for (EdgeIndex j = xadj[v]; j < xadj[v + 1]; j++) {
                        VtxType w = adj[j];
                        if (level[w] == -1) {
                            level[w] = cl1;
                            farness += cl1;
                            que[quep++] = w;
                            eup += xadj[w+1] - xadj[w];
                        }
                    }
                }
            }
            cc[source] = 1.0f / farness;
        }
        delete[] que;
        delete[] level;
        delete[] remaining;
    }

}

template 
void cc_cpu_hybrid<int, int> (int* xadj, int* adj, int nVtx, int nb, float* cc);

template
void cc_cpu_hybrid<int, long int> (long int* xadj, int* adj, int nVtx, int nb, float* cc);

template
void cc_cpu_hybrid<long int, long int> (long int* xadj, long int* adj, long int nVtx, long int nb, float* cc);
