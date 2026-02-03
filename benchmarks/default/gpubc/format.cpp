#include <stdlib.h>
#include <string.h>
#include <stdio.h>
#include <assert.h>
#include <iostream>

//memory layout function
template <typename VtxType, typename EdgeIndex>
EdgeIndex createVirtualCSR(EdgeIndex* ptrs, VtxType* js, VtxType nov, VtxType* vmap, EdgeIndex* virptrs, VtxType maxload, bool permuteAdj) {
  EdgeIndex vcount = 0;
	VtxType* temp = (VtxType*)malloc(sizeof(VtxType) * ptrs[nov]);
	EdgeIndex* temp2 = (EdgeIndex*)malloc(sizeof(EdgeIndex) * ptrs[nov]);

	std::cout<<"virtualizing vertices! "<< nov<<std::endl;

	virptrs[0] = 0;
	for(VtxType i = 0; i < nov; i++) {
	  VtxType deg = ptrs[i+1] - ptrs[i];
		//		cout<<deg<<endl;
		VtxType nvirtual = deg / maxload;
		VtxType remaining = deg % maxload;

		for(VtxType j = 0; j < nvirtual; j++) { /* these are for full virtual vertices */
			vmap[vcount] = i;
			virptrs[vcount + 1] = virptrs[vcount] + maxload;
			vcount++;
		}

		if(remaining > 0) {
			if(nvirtual > 0) {
				VtxType dif = (maxload - remaining) / 2;
				virptrs[vcount] -= dif;
				remaining += dif;
			}
			vmap[vcount] = i;
			virptrs[vcount + 1] = virptrs[vcount] + remaining;
			vcount++;
		}
	}

	if(permuteAdj) { /* scatters the nonzeros to virtual vertices */
		EdgeIndex start, to;
		memcpy(temp2, virptrs, sizeof(EdgeIndex) * (vcount+1));
		start = 0;
		for(VtxType i = 0; i < nov; i++) {
			while(vmap[start] < i && start < vcount) {
				start++;
			}
			if(start < vcount && vmap[start] == i) {
				to = start;
				for(EdgeIndex p = ptrs[i]; p < ptrs[i+1];) {
					temp[temp2[to]++] = js[p++];

					while(p < ptrs[i+1]) {
						to++;
						if(to == vcount || vmap[to] != i) to = start;
						if(temp2[to] < virptrs[to+1]) break;
					}
				}
			}
		}
	}

	std::cout<<"number of virtual vertices "<<vcount<<std::endl;

	if(ptrs[nov] != virptrs[vcount]) {
	  std::cout<<"VIRTUAL: "<<ptrs[nov]<<" != "<<virptrs[vcount]<<std::endl;
		exit(1);
	}

	free(temp);
	free(temp2);
	return vcount;
}


//memory layout functions
template <typename VtxType, typename EdgeIndex>
EdgeIndex createVirtualCoalescedCSR(EdgeIndex* ptrs, VtxType* js, VtxType nov, VtxType* vmap, EdgeIndex* virptrs, EdgeIndex* startoffset, VtxType* stride, VtxType maxload, bool permuteAdj) {
	EdgeIndex vcount = 0;
	VtxType* temp = (VtxType*)malloc(sizeof(VtxType) * ptrs[nov]);
	EdgeIndex* temp2 = (EdgeIndex*)malloc(sizeof(EdgeIndex) * ptrs[nov]);

	std::cout<<"virtualizing vertices! "<< nov<<std::endl;

	virptrs[0] = 0;
	for(VtxType i = 0; i < nov; i++) {
		VtxType deg = ptrs[i+1] - ptrs[i];
		VtxType nvirtual = deg / maxload;
		VtxType remaining = deg % maxload;

		stride[i] = nvirtual+(remaining>0); //total number of virtual vertex for vertex i

		for(VtxType j = 0; j < nvirtual; j++) { /* these are for full virtual vertices */
			vmap[vcount] = i;
			virptrs[vcount + 1] = virptrs[vcount] + maxload;


			startoffset[vcount] = ptrs[i]+j;

			vcount++;
		}

		if(remaining > 0) {
			startoffset[vcount] = ptrs[i]+nvirtual;
			if(nvirtual > 0) {
				VtxType dif = (maxload - remaining) / 2;
				virptrs[vcount] -= dif;
				remaining += dif;
			}
			vmap[vcount] = i;
			virptrs[vcount + 1] = virptrs[vcount] + remaining;
			vcount++;
		}
	}

	if(permuteAdj) { /* scatters the nonzeros to virtual vertices */
		EdgeIndex start, to;
		memcpy(temp2, virptrs, sizeof(int) * (vcount+1));
		start = 0;
		for(VtxType i = 0; i < nov; i++) {
			while(vmap[start] < i && start < vcount) {
				start++;
			}
			if(start < vcount && vmap[start] == i) {
				to = start;
				for(EdgeIndex p = ptrs[i]; p < ptrs[i+1];) {
					temp[temp2[to]++] = js[p++];

					while(p < ptrs[i+1]) {
						to++;
						if(to == vcount || vmap[to] != i) to = start;
						if(temp2[to] < virptrs[to+1]) break;
					}
				}
			}
		}
	}

	std::cout<<"number of virtual vertices "<<vcount<<std::endl;

	if(ptrs[nov] != virptrs[vcount]) {
	  std::cout<<"VIRTUAL: "<<ptrs[nov]<<" != "<<virptrs[vcount]<<std::endl;
		exit(1);
	}

	free(temp);
	free(temp2);
	return vcount;
}


template <typename VtxType, typename EdgeIndex>
void order_graph (EdgeIndex* xadj, VtxType* adj, VtxType* weight, float* bc, VtxType n, VtxType vcount, int deg1, VtxType* map_for_order, VtxType* reverse_map_for_order) {

  EdgeIndex *new_xadj;
  VtxType *new_adj;

	new_xadj = (EdgeIndex*) calloc((n + 1), sizeof(EdgeIndex));
	new_adj = (VtxType*) malloc(sizeof(VtxType) * xadj[n]);

	VtxType* my_map_for_order = (VtxType *) malloc(n * sizeof(VtxType));
	VtxType* my_reverse_map_for_order = (VtxType *) malloc(n * sizeof(VtxType));
	for (VtxType i = 0; i < n; i++) {
		my_map_for_order[i] = my_reverse_map_for_order[i] = -1;
	}

	int* mark = (int*) calloc((n + 1), sizeof(int));
	VtxType* bfsorder = (VtxType*) malloc((n + 1) * sizeof(VtxType));
	VtxType endofbfsorder = 0;

	for (VtxType i = 0; i < n; i++) {
		if (xadj[i+1] > xadj[i]) {
			bfsorder[endofbfsorder++] = i;
			mark[i] = 1;
			break;
		}
	}

	{
	  VtxType cur = 0;
	  while (cur != endofbfsorder) {
		VtxType v = bfsorder[cur];
		my_reverse_map_for_order[cur] = v;
		my_map_for_order[v] = cur;
		for (EdgeIndex j = xadj[v]; j < xadj[v+1]; j++) {
			VtxType w = adj[j];
			if (mark[w] == 0) {
				mark[w] = 1;
				bfsorder[endofbfsorder++] = w;
			}
		}
		cur++;
	  }
	  for (VtxType i = 0; i < n; i++) {
		if (mark[i] == 0) {
			my_reverse_map_for_order[cur] = i;
			my_map_for_order[i] = cur;
			cur++;
		}
	  }
	}

	EdgeIndex ptr = 0;
	for (VtxType i = 0; i < n; i++) {
		new_xadj[i+1] = new_xadj[i];
		VtxType u = my_reverse_map_for_order[i];
		for (EdgeIndex j = xadj[u]; j < xadj[u+1]; j++) {
			VtxType val = adj[j];
			if (!(ptr < xadj[n])) {
				printf("ptr is not less than xadj[n]\n");
				exit(1);
			}		
			if (!(val < n)) {
				printf("val is not less than n\n");
				exit(1);
			}
			new_adj[ptr++] = my_map_for_order[val];
			new_xadj[i+1]++;
		}
	}

	free(mark);
	free(bfsorder);

	VtxType* new_weight = (VtxType*) malloc (sizeof(VtxType) * n);
	float* new_bc = (float*) malloc (sizeof(float) * n);
	for (VtxType i = 0; i < n; i++) {
		new_bc[my_map_for_order[i]] = bc[i];
		new_weight[my_map_for_order[i]] = weight[i];
	}


	VtxType* temp_map_for_order = (VtxType *) malloc(vcount * sizeof(VtxType));
	VtxType* temp_reverse_map_for_order = (VtxType *) malloc(vcount * sizeof(VtxType));

	if (deg1) {
		for (VtxType i = 0; i < vcount; i++) {
			if (map_for_order[i] != -1) {
				VtxType u = my_map_for_order[map_for_order[i]];
				temp_map_for_order[i] = u;
				temp_reverse_map_for_order[u] = i;
			}
		}
	}
	else
		for (VtxType i = 0; i < vcount; i++) {
			VtxType u = my_map_for_order[i];
			temp_map_for_order[i] = u;
			temp_reverse_map_for_order[u] = i;
		}

	memcpy(map_for_order, temp_map_for_order, sizeof(VtxType) * vcount);
	memcpy(reverse_map_for_order, temp_reverse_map_for_order, sizeof(VtxType) * vcount);

	free (my_map_for_order);
	free (my_reverse_map_for_order);
	free (temp_map_for_order);
	free (temp_reverse_map_for_order);

	memcpy(xadj, new_xadj, sizeof(EdgeIndex) * (n+1));
	memcpy(adj, new_adj, sizeof(VtxType) * xadj[n]);
	free (new_adj);
	free (new_xadj);

	memcpy(bc, new_bc, sizeof(float)*n);
	memcpy(weight, new_weight, sizeof(VtxType)*n);
	free(new_bc);
	free(new_weight);     

}

//explicit instanciation

template int createVirtualCSR<int, int> (int* ptrs, int* js, int nov, int* vmap, int* virptrs, int maxload, bool permuteAdj);
template long int createVirtualCSR<int, long int> (long int* ptrs, int* js, int nov, int* vmap, long int* virptrs, int maxload, bool permuteAdj);
template long int createVirtualCSR<long int, long int> (long int* ptrs, long int* js, long int nov, long int* vmap, long int* virptrs, long int maxload, bool permuteAdj);

template int createVirtualCoalescedCSR<int, int> (int* ptrs, int* js, int nov, int* vmap, int* virptrs, int* startoffset, int* stride, int maxload, bool permuteAdj);
template long int createVirtualCoalescedCSR<int, long int> (long int* ptrs, int* js, int nov, int* vmap, long int* virptrs, long int* startoffset, int* stride, int maxload, bool permuteAdj);
template long int createVirtualCoalescedCSR<long int, long int> (long int* ptrs, long int* js, long int nov, long int* vmap, long int* virptrs, long int* startoffset, long int* stride, long int maxload, bool permuteAdj);

template void order_graph<int, int> (int* xadj, int* adj, int* weight, float* bc, int n, int vcount, int deg1, int* map_for_order, int* reverse_map_for_order);
template void order_graph<int, long int> (long int* xadj, int* adj, int* weight, float* bc, int n, int vcount, int deg1, int* map_for_order, int* reverse_map_for_order);
template void order_graph<long int, long int> (long int* xadj, long int* adj, long int* weight, float* bc, long int n, long int vcount, int deg1, long int* map_for_order, long int* reverse_map_for_order);
