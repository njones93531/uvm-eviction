/*
  ---------------------------------------------------------------------
  This file is a part of the source code for the paper "Betweenness
  Centrality on GPUs and Heterogeneous Architectures", published in
  GPGPU'13 workshop. If you use the code, please cite the paper.

  Copyright (c) 2013,
  By:    Ahmet Erdem Sariyuce,
  Kamer Kaya,
  Erik Saule,
  Umit V. Catalyurek
  ---------------------------------------------------------------------
  This file is licensed under the Apache License. For more licensing
  information, please see the README.txt and LICENSE.txt files in the
  main directory.
  ---------------------------------------------------------------------
*/

#include <vector> 
#include <map>
#include <list>
#include <stdio.h>
#include <string.h>
#include <stdio.h>
#include <unistd.h>
#include <cstdlib>
#include <string>
#include <cmath>
#include <iostream>
#include <fstream>
#include <sstream>
#include <sys/time.h>
#include "ulib.h"
#include "timestamp.hpp"
#include "format.h"
#include <assert.h>
#include <omp.h>
#include "config.h"


#include "bc_cpu.hpp"
#include "cc_cpu.hpp"

#include "bc_gpu.hpp"
#include "cc_gpu.hpp"

#include "graphIO.hpp"

#include "format.hpp"

using namespace std;

void init();


template<typename VtxType, typename EdgeIndex>
int preprocess(EdgeIndex *xadj, VtxType* adj, VtxType* tadj, VtxType *np, float* bc, VtxType* weight, VtxType* map_for_order, VtxType* reverse_map_for_order, FILE* ofp);




int main(int argc, char** argv) {
  //typedef int VtxType;
  //typedef int EdgeIndex;

  typedef long int VtxType;
  typedef long int EdgeIndex;


  char c;
  char* infilename, outfilename;
  int threads_per_block = -1, paropt = -1, times = 1, sortopt = 0;

  VtxType nb = 1;
  VtxType xpar = 0;
  

  std::string outputfilename = "bc_out.txt";

  if (!(argc == 6 || argc == 7)) {
    std::cout<<"usage: "<<argv[0]<<" <filename> <will_order:0|1> <kernel> <nbsource> <xpar> [outputfile]"<<std::endl;
    std::cout<<"possible kernels:"<<std::endl
	     <<"BC-CPU-NAIVE BC-CPU-NAIVE-DEG1 (*)"<<std::endl
	     <<"BC-GPU-VERTEX BC-GPU-VERTEX-DEG1 BC-GPU-EDGE BC-GPU-VIRTUAL BC-GPU-VIRTUAL-DEG1 BC-GPU-VIRTUAL-COAL"<<std::endl
	     <<"BC-GPU-VERTEX-MULTI BC-GPU-EDGE-MULTI BC-GPU-VIRTUAL-MULTI (*)"<<std::endl
	     <<std::endl
	     <<"CC-CPU-NAIVE CC-CPU-DO CC-CPU-SPMV (*)"<<std::endl
	     <<"CC-GPU-VIRTUAL CC-GPU-VERTEX-SPMV (*)"<<std::endl
	     <<"CC-MIC-NAIVE CC-MIC-DO CC-MIC-SPMV (*)"<<std::endl
	     <<std::endl<<"(*) denotes the recommended kernel"<<std::endl;
    return -1;
  }

  init();

  int will_order;
  { 
    std::stringstream ss ;
    ss<< argv[2];
    ss >> will_order;
    if (! ss) {
      std::cerr<<"incorect format"<<std::endl;
      return -1;
    }
  }
  int deg1 = 0;


  std::string algo (argv[3]);

  { 
    std::stringstream ss ;
    ss<< argv[4];
    ss >> nb;
    if (! ss) {
      std::cerr<<"incorect format"<<std::endl;
      return -1;
    }
  }

  { 
    std::stringstream ss ;
    ss<< argv[5];
    ss >> xpar; // max-deg # of a virtual vertex
    if (! ss) {
      std::cerr<<"incorect format"<<std::endl;
      return -1;
    }
  }

  if (argc == 7) {
    outputfilename = std::string(argv[6]);
  }

  if (algo.compare("BC-CPU-NAIVE-DEG1") == 0 ||
      algo.compare("BC-GPU-VERTEX-DEG1") == 0 ||
      algo.compare("BC-GPU-VIRTUAL-DEG1") == 0)
    deg1 = 1;


  VtxType n, i, j, nVtx;
  EdgeIndex *xadj, *new_xadj;
  VtxType *adj, *tadj, *mark, *queue, *new_adj;

  VtxType maxcompid, u, v;
  EdgeIndex p;
  VtxType nocomp;

  long* reverse_map = NULL;
  bool do_mapping = ReadGraph<VtxType, EdgeIndex, int>(argv[1], &n, &xadj, &adj, &tadj, nullptr, nullptr, &reverse_map);
  nVtx = n;
  VtxType* compid = (VtxType*) malloc(sizeof(VtxType) * n);
  for(VtxType i = 0; i < n; i++)
    compid[i] = -1;
  VtxType* que = (VtxType*) malloc(sizeof(VtxType) * n);

  //printf("there are %d vertices %d edges\n", n, xadj[n]);
  std::cout<<"there are "<<n<<" vertices "<<xadj[n]<<" edges"<<std::endl;

  VtxType lcompid;
  {
    VtxType qptr, qeptr, largestSize, compsize;
    nocomp = qptr = qeptr = largestSize = 0;
    for (VtxType i = 0; i < n; i++) {
      
      if(compid[i] == -1) {
	compsize = 1;
	compid[i] = nocomp;
	que[qptr++] = i;
	
	while(qeptr < qptr) {
	  u = que[qeptr++];
	  for(p = xadj[u]; p < xadj[u+1]; p++) {
	    v = adj[p];
	    if(compid[v] == -1) {
	      compid[v] = nocomp;
	      que[qptr++] = v;
	      compsize++;
	    }
	  }
	}
	if(largestSize < compsize) {
	  lcompid = nocomp;
	  largestSize = compsize;
	}
	nocomp++;
      }
    }
  }

  EdgeIndex nz = xadj[n];
  EdgeIndex ecount = 0;
  VtxType vcount = 0;

  for(VtxType i = 0; i < n; i++) {
    if(compid[i] == lcompid) {
      que[i] = vcount++;
      for(EdgeIndex p = xadj[i]; p < xadj[i+1]; p++) {
	if(compid[adj[p]] == lcompid)
	  ecount++;
      }
    }
  }

  /*
  EdgeIndex* lxadj = (EdgeIndex*) malloc(sizeof(EdgeIndex) * (vcount+1));
  VtxType* ladj = (VtxType*) malloc(sizeof(VtxType) * (ecount));
  VtxType* ltadj = (VtxType*) malloc(sizeof(VtxType) * (ecount));
  */

  EdgeIndex* lxadj;
  VtxType*   ladj;
  VtxType*   ltadj;

  cudaMallocManaged(&lxadj, sizeof(EdgeIndex) * (n + 1));
  cudaMallocManaged(&ladj, sizeof(VtxType) * nz);
  cudaMallocManaged(&ltadj, sizeof(VtxType) * nz);

  vcount = 0;
  ecount = 0;
  lxadj[0] = 0;
  for(VtxType i = 0; i < n; i++) {
    if(compid[i] == lcompid)  {
      vcount++;

      for(EdgeIndex p = xadj[i]; p < xadj[i+1]; p++) {
	if(compid[adj[p]] == lcompid) {
	  ladj[ecount++] = que[adj[p]];
	}
      }
      lxadj[vcount] = ecount;
    }
  }
  free(compid);	compid = NULL;
  free (que);	que = NULL;
  std::cout<<"largest component graph obtained with "<<vcount<<" vertices "<<ecount<<" edges -- "<<lxadj[vcount]<<std::endl;


  n = vcount;
  nz = ecount;
  cudaFree(xadj); xadj = lxadj;
  cudaFree(adj); adj = ladj;
  free(tadj); tadj = ltadj;

  //printf("before malloc\n");
  //fflush(0);

  EdgeIndex* degs = (EdgeIndex*)malloc(sizeof(EdgeIndex) * n);
  VtxType* myedges = (VtxType*)malloc(sizeof(VtxType) * nz);
  std::copy (xadj, xadj+n, degs); //memcpy(degs, xadj, sizeof(Edges) * n);

  //int ptr;
  for(VtxType i = 0; i < n; i++) {
    for(EdgeIndex ptr = xadj[i]; ptr < xadj[i+1]; ptr++) {
      VtxType j = adj[ptr];
      myedges[degs[j]++] = i;
    }
  }

  //printf("after malloc\n");
  //fflush(0);


  for(VtxType i = 0; i < n; i++) {
    if(xadj[i+1] != degs[i]) {
      std::cout<<"something is wrong i "<<i<<" xadj[i+1] "<<xadj[i+1]<<" degs[i] "<<degs[i]<<std::endl;
      exit(1);
    }
  }

  memcpy(adj, myedges, sizeof(VtxType) * xadj[n]);
  for(VtxType i = 0; i < n; i++) {
    for(EdgeIndex ptr = xadj[i]+1; ptr < xadj[i+1]; ptr++) {
      if(adj[ptr] <= adj[ptr-1]) {
	printf("is not sorted\n");
	exit(1);
      }
    }
  }

  //printf("more after malloc\n");
  //fflush(0);

  for (VtxType i = 0; i<n; ++i)
    degs[i] = xadj[i];

  for(VtxType i = 0; i < n; i++) {
    for(EdgeIndex ptr = xadj[i]; ptr < xadj[i+1]; ptr++) {
      VtxType j = adj[ptr];
      if(i < j) {
	tadj[ptr] = degs[j];
	tadj[degs[j]++] = ptr;
      }
    }
  }

  free(degs);
  free(myedges);

  //printf("more more after malloc\n");
  //fflush(0);

  for(VtxType i = 0; i < n; i++) {
    for(EdgeIndex ptr = xadj[i]; ptr < xadj[i+1]; ptr++) {
      VtxType j = adj[ptr];
      if((adj[tadj[ptr]] != i) || (tadj[ptr] < xadj[j]) || (tadj[ptr] >= xadj[j+1])) {
	std::cout<<"error i "<<i<<" j "<<j<<" ptr "<<ptr<<std::endl;
	std::cout<<"error  xadj[j] "<<xadj[j]<<" xadj[j+1] "<< xadj[j+1]<<std::endl;
	std::cout<<"error tadj[ptr] "<< tadj[ptr]<<std::endl;
	std::cout<<"error adj[tadj[ptr]] "<< adj[tadj[ptr]]<<std::endl;
	exit(1);
      }
    }
  }

  //printf("more more more after malloc\n");
  //fflush(0);

  VtxType* map_for_order = (VtxType *) malloc(n * sizeof(VtxType));
  VtxType* reverse_map_for_order = (VtxType *) malloc(n * sizeof(VtxType));
  VtxType* weight = (VtxType *) malloc(sizeof(VtxType) * n);
  float* bc  = (float *) malloc(sizeof(float) * n);

  for(VtxType i = 0; i < n; i++) {
    weight[i] = 1;
    map_for_order[i] = -1;
    reverse_map_for_order[i] = -1;
    bc[i] = 0.;
  }


  struct timeval t1, t2, t3, t4, t5, t6, t7, gt1, gt2;
  t1.tv_sec = t1.tv_usec = t2.tv_sec = t2.tv_usec = t3.tv_sec = t3.tv_usec = t4.tv_sec = t4.tv_usec = t5.tv_sec = t5.tv_usec = t6.tv_sec = t6.tv_usec = t7.tv_sec = t7.tv_usec = 0;
  double time_preproc, time_order, time_kernel, time_total, time_virt;


  FILE* ofp;
  ofp = fopen("bc_out.txt", "w");

  gettimeofday (&t1, 0);
  if (deg1 == 1) {
    preprocess (xadj, adj, tadj, &n, bc, weight, map_for_order, reverse_map_for_order, ofp);
    nz = xadj[n];
  }

  gettimeofday (&t2, 0);

  if (will_order == 1) {
    order_graph (xadj, adj, weight, bc, n, vcount, deg1, map_for_order, reverse_map_for_order);
  }
  
  free(map_for_order); map_for_order = NULL;

  gettimeofday (&t3, 0);

  if (nb < 0 )
    nb = n;

  nb = min(n, nb);

  

  std::cout<<"will be executed on "<<n<<" vertices "<<xadj[n]<<" "<< nz<<" edges"<<std::endl;

  bool processed = false;


  // kernels..
  if (algo.compare("BC-CPU-NAIVE-DEG1") == 0) {
    bc_cpu_deg1 (xadj, adj, n, nz, nb, bc, weight);
    processed = true;
  }
  else if (algo.compare("BC-GPU-VERTEX-DEG1") == 0) {
    bc_gpu_vertex_deg1 (xadj, adj, n, nz, nb, bc, weight);
    processed = true;
  }
  else if(algo.compare("BC-GPU-VIRTUAL-DEG1") == 0) {
    VtxType* extv = (VtxType*)malloc(sizeof(VtxType) * (nz + 1 + (xpar * WARP)));
    EdgeIndex* start = (EdgeIndex*)malloc(sizeof(EdgeIndex) * (nz + 1 + (xpar * WARP)));
    
    gettimeofday (&t5, 0);
    
    EdgeIndex nov_ext = createVirtualCSR(xadj, adj, n, extv, start, xpar, 0); //number of virtual vertex is about |E|
    
    gettimeofday (&t6, 0);
    bc_gpu_virtual_deg1 (extv, start, adj, n, nz, nov_ext, nb, bc, weight);
    
    free (extv);
    free (start);
    processed = true;
  }
  else if (algo.compare("BC-CPU-NAIVE") == 0) {
    bc_cpu (xadj, adj, n, nz, nb, bc);
    processed = true;
  }
    else if (algo.compare("BC-GPU-VERTEX") == 0) {
      bc_gpu_vertex (xadj, adj, n, nz, nb, bc);
      processed = true;
    }
    else if(algo.compare("BC-GPU-EDGE") == 0) {
      VtxType* is = (VtxType*) malloc(sizeof(VtxType) * nz);
      for(VtxType i = 0; i < n; i++) {
	for(EdgeIndex ptr = xadj[i]; ptr < xadj[i+1]; ptr++) {
	  is[ptr] = i;
	}
      }

      bc_gpu_edge (is, adj, n, nz, nb, bc);

      free (is);
      processed = true;
    }
    else if(algo.compare("BC-GPU-VIRTUAL") == 0) {
      VtxType* extv = (VtxType*)malloc(sizeof(VtxType) * (nz + 1 + (xpar * WARP)));
      EdgeIndex* start = (EdgeIndex*)malloc(sizeof(EdgeIndex) * (nz + 1 + (xpar * WARP)));

      gettimeofday (&t5, 0);

      EdgeIndex nov_ext = createVirtualCSR<VtxType, EdgeIndex> (xadj, adj, n, extv, start, xpar, false);

      gettimeofday (&t6, 0);

      bc_gpu_virtual (extv, start, adj, n, nz, nov_ext, nb, bc);
      free (extv);
      free (start);
      processed = true;
    }
    else if(algo.compare("BC-GPU-VIRTUAL-COAL") == 0) {
      VtxType* extv = (VtxType*)malloc(sizeof(VtxType) * (nz + 1 + (xpar * WARP)));
      EdgeIndex* start = (EdgeIndex*)malloc(sizeof(EdgeIndex) * (nz + 1 + (xpar * WARP)));
      EdgeIndex* startoffset = (EdgeIndex*)malloc(sizeof(int)*(nz + 1 + (xpar * WARP))); //basically xadj[vmap[thread]]+thread_in_vvertex
      VtxType* stride = (VtxType*)malloc(sizeof(VtxType)*(n)); //stride is actually number of virtual vertex per actual vertex

      gettimeofday (&t5, 0);

      EdgeIndex nov_ext = createVirtualCoalescedCSR(xadj, adj, n, extv, start, startoffset, stride, xpar, 0);

      gettimeofday (&t6, 0);

      bc_gpu_virtual_coalesced (extv, xadj, adj, n, startoffset, stride, nz, nov_ext, nb, bc);
      free (extv);
      free (start);
      free (startoffset);
      free (stride);
      processed = true;
    }
    else if (algo.compare("BC-GPU-VERTEX-MULTI") == 0) {
      bc_gpu_vertex_multi (xadj, adj, n, nz, nb, bc);
      processed = true;
    }
    else if(algo.compare("BC-GPU-EDGE-MULTI") == 0) {
      VtxType* is = (VtxType*) malloc(sizeof(VtxType) * nz);
      for(VtxType i = 0; i < n; i++) {
	for(EdgeIndex ptr = xadj[i]; ptr < xadj[i+1]; ptr++) {
	  is[ptr] = i;
	}
      }

      bc_gpu_edge_multi (is, adj, n, nz, nb, bc);

      free (is);
      processed = true;
    }
    else if(algo.compare("BC-GPU-VIRTUAL-MULTI") == 0) {
      VtxType* extv = (VtxType*)malloc(sizeof(VtxType) * (nz + 1 + (xpar * WARP)));
      EdgeIndex* start = (EdgeIndex*)malloc(sizeof(EdgeIndex) * (nz + 1 + (xpar * WARP)));

      gettimeofday (&t5, 0);

      EdgeIndex nov_ext = createVirtualCSR(xadj, adj, n, extv, start, xpar, 0);

      gettimeofday (&t6, 0);

      bc_gpu_virtual_multi (extv, start, adj, n, nz, nov_ext, nb, bc);
      free (extv);
      free (start);
      processed = true;
    }
    else if(algo.compare("CC-GPU-VIRTUAL") == 0) {
      VtxType* extv = (VtxType*)malloc(sizeof(VtxType) * (nz + 1 + (xpar * WARP)));
      EdgeIndex* start = (EdgeIndex*)malloc(sizeof(EdgeIndex) * (nz + 1 + (xpar * WARP)));

      gettimeofday (&t5, 0);

      EdgeIndex nov_ext = createVirtualCSR(xadj, adj, n, extv, start, xpar, 0);

      gettimeofday (&t6, 0);

      cc_gpu_virtual (extv, start, adj, n, nz, nov_ext, nb, bc);
      free (extv);
      free (start);
      processed = true;
    }
    else if(algo.compare("CC-CPU-NAIVE") == 0) {
      cc_cpu (xadj, adj, n, nb, bc);
      processed = true;
    }
    else if(algo.compare("CC-CPU-DO") == 0) {
      cc_cpu_hybrid (xadj, adj, n, nb, bc);
      processed = true;
    } else if(algo.compare("CC-MIC-NAIVE") == 0) {
      cc_mic(xadj, adj, n, nb, bc);
      processed = true;
    } else if(algo.compare("CC-MIC-DO") == 0) {
      cc_mic_hybrid(xadj, adj, n, nb, bc);
      processed = true;
    } else if (algo.compare("CC-GPU-VERTEX-SPMV") == 0) {
      cc_gpu_vertex_multi_spmv (xadj, adj, n, nb, bc);
      processed = true;
    } else if (algo.compare("CC-MIC-SPMV") == 0) {
      cc_mic_spmv_soft_vec_opt (xadj, adj, n, nb, bc);
      processed = true;
    } else if (algo.compare("CC-CPU-SPMV") == 0) {
      cc_cpu_spmv_soft_vec_opt (xadj, adj, n, nb, bc);
      processed = true;
    }

  if (!processed) {
    std::cerr<<"unknown algorithm"<<std::endl;
    return -1;
  }
  if (n > 5000)
    std::cout<<"bc[0]="<<bc[0]<<" bc[100]="<<bc[100]<<" bc[200]="<<bc[200]<<" bc[1000]="<<bc[1000]<<" bc[5000]="<<bc[5000]<<std::endl;

  gettimeofday (&t4, 0);


  if (will_order+deg1 > 0)
    for (VtxType i = 0; i < n; i++) {
      if (reverse_map_for_order[i] != -1) {
	//fprintf(ofp, "bc[%d]: %lf\n", reverse_map_for_order[i], bc[i]);
	std::stringstream ss;
	ss<<"bc["<<reverse_map_for_order[i]<<"]: "<<bc[i];
	fprintf(ofp, "%s\n", ss.str().c_str());
      }
    }
  else
    for (VtxType i = 0; i < n; i++) {
	std::stringstream ss;
	ss<<"bc["<<i<<"]: "<<bc[i];
	fprintf(ofp, "%s\n", ss.str().c_str());
	//fprintf(ofp, "bc[%d]: %lf\n", i, bc[i]);
    }
  free(reverse_map_for_order); reverse_map_for_order = NULL;

  fclose(ofp);
  FILE* lfp;
  lfp = fopen("bc_out.txt", "r");

  double* result_bc = new double[nVtx];
  {
    double val;
    VtxType id;
    char a,b,d,e,f,g, h;
    bool cont = true;
    int maxsize = 1000000;
    char line [maxsize];
    while (cont) {
      char* ret = fgets(line, maxsize, lfp);
      if (ret == NULL)
	cont = false;
      else {
	std::string s (line);
	std::stringstream ss (s);
	ss>>a>>b>>c>>id>>d>>e>>g>>val>>h;
	if (ss)
	  result_bc[id] = val;	  
	else
	  cont = false;
      }
      
    }
    //while (fscanf(lfp, "%c%c%c%d%c%c%c%lf%c", &a, &b, &f, &id, &d, &e, &g, &val, &h) != EOF) {
    //result_bc[id] = val;
    //}
  }
  fclose(lfp);
  ofp = fopen(outputfilename.c_str(), "w");
  if (do_mapping == false) {
    for(VtxType i = 0; i < nVtx; i++) {
      std::stringstream ss;
      ss<<"bc["<<i<<"]: "<<result_bc[i]/2;
      fprintf(ofp, "%s\n", ss.str().c_str());
      //fprintf(ofp, "bc[%d]: %lf\n", i, result_bc[i]/2);
    }
  }
  else {
    double* last_bc = new double[nVtx];
    for(VtxType i = 0; i < nVtx; i++)
      last_bc[reverse_map[i]] = result_bc[i]/2;
    for(VtxType i = 0; i < nVtx; i++) {
      std::stringstream ss;
      ss<<"bc["<<i<<"]: "<<last_bc[i];
      fprintf(ofp, "%s\n", ss.str().c_str());
      //fprintf(ofp, "bc[%d]: %lf\n", i, last_bc[i]);
    }
    delete[] last_bc;
  }
  fclose(ofp);
  delete[] result_bc;

  free (bc); bc = NULL;
  free (weight); weight = NULL;
 


  time_preproc = (1000000.0 * (t2.tv_sec-t1.tv_sec) + t2.tv_usec-t1.tv_usec) / 1000000.0;
  cout << "preproc time: " <<time_preproc<<" secs\n";
  time_order = (1000000.0 * (t3.tv_sec-t2.tv_sec) + t3.tv_usec-t2.tv_usec) / 1000000.0;
  cout << "ordering time: " <<time_order<<" secs\n";
  time_kernel = (1000000.0 * (t4.tv_sec-t3.tv_sec) + t4.tv_usec-t3.tv_usec) / 1000000.0;
  cout << "kernel time: " <<time_kernel<<" secs\n";
  time_total = (1000000.0 * (t4.tv_sec-t1.tv_sec) + t4.tv_usec-t1.tv_usec) / 1000000.0;
  cout << "total time: " <<time_total<<" secs "<<"for "<<nb<<" bfs calls\n";
  cout << "performance: " <<(double)nb*xadj[n]/(double)time_total/1024/1024<<" MTEPS\n";
  time_virt = (1000000.0 * (t6.tv_sec-t5.tv_sec) + t6.tv_usec-t5.tv_usec) / 1000000.0;
  cout << "virtualization time: " <<time_virt<<" secs "<<"for "<<nb<<" bfs calls\n";

  cudaFree(xadj);  xadj= NULL; 
  cudaFree(adj);   adj = NULL;
  cudaFree(tadj);  tadj= NULL;

  return 0;
}

