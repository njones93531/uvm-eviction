#include <hip/hip_runtime.h>
#include <stdio.h>
#include <unistd.h>
#include <stdlib.h>
#include <time.h>
#include <vector>
#include <string>
#include <cstring>
#include <climits>
#include <queue>
#include <stdexcept>

#define BLOCK_SIZE 32
#define TPB BLOCK_SIZE*BLOCK_SIZE
#define log2TPB 10

#define die(x) do{ perror(x); exit(1); }while(0);

#define frand()	(float)rand()/(float)rand()

#define hipErrchk(ans)                      \
  {                                         \
    hipAssert((ans), __FILE__, __LINE__);	\
  }

struct Graph {
	uint64_t* adjList;
	uint64_t* edgeOffset;
	uint64_t* edgeSize;
	uint64_t numVertices = 0;
	uint64_t numEdges = 0;
};

inline uint64_t rand64(){
	return (uint64_t)rand() << 32 | (uint64_t)rand();
}

void populateGraph(Graph* G, uint64_t vertices, uint64_t edges){
	if(edges > (vertices*(vertices-1))>>1) die("Invalid graph configuration.\n");
	G->numVertices = vertices;
	G->numEdges = edges;

	std::vector<std::vector<uint64_t>> adjacencyLists(vertices);
	if(edges == (vertices*(vertices-1))>>1){
		for(uint64_t i = 0; i < vertices; i++){
			for(uint64_t j = i+1; j < vertices; j++){
				adjacencyLists[i].push_back(j);
				adjacencyLists[j].push_back(i);
			}
		}
	}else{
		for(uint64_t i = 0; i < edges; i++){
			uint64_t u = rand64() % vertices;
			uint64_t v = rand64() % vertices;
			adjacencyLists[u].push_back(v);
			adjacencyLists[v].push_back(u);
		}
	}

	uint64_t size = 0;
	for(uint64_t i = 0; i < vertices; i++){
		G->edgeOffset[i] = size;
		G->edgeSize[i] = adjacencyLists[i].size();
		for(auto &edge: adjacencyLists[i]){
			G->adjList[size] = edge;
			size += 1;
		}
	}
}

inline void hipAssert(hipError_t code, const char *file, int line, bool abort = true){
  if (code != hipSuccess) {
    if (abort) {
      std::string msg;
      msg += "HIPassert: ";
      msg += hipGetErrorString(code);
      msg += " ";
      msg += file;
      msg += ":";
      msg += std::to_string(line);
      throw std::runtime_error(msg);
    } else {
      fprintf(stderr, "HIPassert: %s %s %d\n", hipGetErrorString(code), file, line);
    }
  }
}

__global__ void simpleBfs(uint64_t N, uint64_t level, uint64_t* adjList, uint64_t* edgeOffset, uint64_t* edgeSize, uint64_t* distance, uint64_t* parent, uint64_t* changed){
	uint64_t idx = (uint64_t)blockIdx.x*(uint64_t)blockDim.x*(uint64_t)blockDim.y + (uint64_t)threadIdx.x + (uint64_t)threadIdx.y*(uint64_t)blockDim.x;
	bool isChanged = false;
	if(idx < N && distance[idx] == level){
		for(uint64_t i = edgeOffset[idx]; i < edgeOffset[idx] + edgeSize[idx]; i++){
			uint64_t v = adjList[i];
			if(level+1 < distance[v]){
				distance[v] = level + 1;
				parent[v] = idx;
				isChanged = true;
			}
		}
	}
	if(isChanged) *changed = 1;
}

__global__ void queueBfs(uint64_t level, uint64_t* adjList, uint64_t* edgeOffset, uint64_t* edgeSize, uint64_t* distance, uint64_t* parent, uint64_t queueSize, uint64_t* nextQueueSize, uint64_t* currentQueue, uint64_t* nextQueue){
	uint64_t idx = (uint64_t)blockIdx.x*(uint64_t)blockDim.x*(uint64_t)blockDim.y + (uint64_t)threadIdx.x + (uint64_t)threadIdx.y*(uint64_t)blockDim.x;
	if(idx < queueSize){
		uint64_t u = currentQueue[idx];
		for(uint64_t i = edgeOffset[u]; i < edgeOffset[u] + edgeSize[u]; i++){
			uint64_t v = adjList[i];
			if(distance[v] == ULLONG_MAX && atomicMin(&distance[v], level+1) == ULLONG_MAX){
				parent[v] = i;
				uint64_t position = atomicAdd(nextQueueSize, 1);
				nextQueue[position] = v;
			}
		}
	}
}

__global__ void nextLayer(uint64_t level, uint64_t* adjList, uint64_t* edgeOffset, uint64_t* edgeSize, uint64_t* distance, uint64_t* parent, uint64_t queueSize, uint64_t* currentQueue){
	uint64_t idx = (uint64_t)blockIdx.x*(uint64_t)blockDim.x*(uint64_t)blockDim.y + (uint64_t)threadIdx.x + (uint64_t)threadIdx.y*(uint64_t)blockDim.x;
	if(idx < queueSize){
		uint64_t u = currentQueue[idx];
		for(uint64_t i = edgeOffset[u]; i < edgeOffset[u] + edgeSize[u]; i++){
			uint64_t v = adjList[i];
			if(level+1 < distance[v]){
				distance[v] = level + 1;
				parent[v] = i;
			}
		}
	}
}

__global__ void countDegrees(uint64_t* adjList, uint64_t* edgeOffset, uint64_t* edgeSize, uint64_t* parent, uint64_t queueSize, uint64_t* currentQueue, uint64_t* degrees){
	uint64_t idx = (uint64_t)blockIdx.x*(uint64_t)blockDim.x*(uint64_t)blockDim.y + (uint64_t)threadIdx.x + (uint64_t)threadIdx.y*(uint64_t)blockDim.x;
	if(idx < queueSize){
		uint64_t u = currentQueue[idx];
		uint64_t degree = 0;
		for(uint64_t i = edgeOffset[u]; i < edgeOffset[u] + edgeSize[u]; i++){
			uint64_t v = adjList[i];
			if(parent[v] == i && v != u){
				degree += 1;
			}
		}
		degrees[idx] = degree;
	}
}

__global__ void scanDegrees(uint64_t size, uint64_t* degrees, uint64_t* incrDegrees){
	uint64_t idx = (uint64_t)blockIdx.x*(uint64_t)blockDim.x*(uint64_t)blockDim.y + (uint64_t)threadIdx.x + (uint64_t)threadIdx.y*(uint64_t)blockDim.x;
	if(idx < size){
		__shared__ uint64_t prefixSum[TPB];
		uint64_t modulo = (uint64_t)threadIdx.x + (uint64_t)blockDim.x*(uint64_t)threadIdx.y;
		prefixSum[modulo] = degrees[idx];
		__syncthreads();

		for(uint64_t nodeSize = 2; nodeSize <= TPB; nodeSize <<= 1){
			if((modulo & (nodeSize-1)) == 0){
				if(idx + (nodeSize >> 1) < size){
					uint64_t nextPosition = modulo + (nodeSize >> 1);
					prefixSum[modulo] += prefixSum[nextPosition];
				}
			}
			__syncthreads();
		}

		if(modulo == 0){
			uint64_t block = idx >> log2TPB;
			incrDegrees[block+1] = prefixSum[modulo];
		}

		for(uint64_t nodeSize = TPB; nodeSize > 1; nodeSize >>= 1){
			if((modulo & (nodeSize-1)) == 0){
				if(idx + (nodeSize >> 1) < size){
					uint64_t nextPosition = modulo + (nodeSize >> 1);
					uint64_t tmp = prefixSum[modulo];
					prefixSum[modulo] -= prefixSum[nextPosition];
					prefixSum[nextPosition] = tmp;
				}
			}
			__syncthreads();
		}
		degrees[idx] = prefixSum[modulo];
	}
}

__global__ void assignVerticesNextQueue(uint64_t* adjList, uint64_t* edgeOffset, uint64_t* edgeSize, uint64_t* parent, uint64_t queueSize, uint64_t* currentQueue, uint64_t* nextQueue, uint64_t* degrees, uint64_t* incrDegrees, uint64_t nextQueueSize){
	uint64_t idx = (uint64_t)blockIdx.x*(uint64_t)blockDim.x*(uint64_t)blockDim.y + (uint64_t)threadIdx.x + (uint64_t)threadIdx.y*(uint64_t)blockDim.x;
	if(idx < queueSize){
		__shared__ uint64_t sharedIncrement;
		if(!(threadIdx.x + blockDim.x*threadIdx.y)){
			sharedIncrement = incrDegrees[idx >> log2TPB];
		}
		__syncthreads();
		uint64_t sum = 0;
		if(threadIdx.x + blockDim.x*threadIdx.y){
			sum = degrees[idx-1];
		}

		uint64_t u = currentQueue[idx];
		uint64_t counter = 0;
		for(uint64_t i = edgeOffset[u]; i < edgeOffset[u] + edgeSize[u]; i++){
			uint64_t v = adjList[i];
			if(parent[v] == i && v != u){
				uint64_t nextQueuePlace = sharedIncrement + sum + counter;
				nextQueue[nextQueuePlace] = v;
				counter++;
			}
		}
	}
}

void initConstData(uint64_t* a, uint64_t len, uint64_t val){
	for(uint64_t i = 0; i < len; i++){
		a[i] = val;
	}
}

void refBfs(uint64_t startVertex, Graph* G, uint64_t* distance, uint64_t* parent, uint64_t* visited){
	for(uint64_t i = 0; i < G->numVertices; i++){
		distance[i] = ULLONG_MAX;
	}
	distance[startVertex] = 0;
	parent[startVertex] = startVertex;
	visited[startVertex] = 1;
	std::queue<uint64_t> Q;
	Q.push(startVertex);
	while(!Q.empty()){
		uint64_t u = Q.front();
		Q.pop();
		for(uint64_t i = G->edgeOffset[u]; i < G->edgeOffset[u] + G->edgeSize[u]; i++){
			uint64_t v = G->adjList[i];
			if(!visited[v]){
				visited[v] = 1;
				distance[v] = distance[u] + 1;
				parent[v] = i;
				Q.push(v);
			}
		}
	}
}

int main(int argc, char* argv[]){
	uint64_t edges, vertices, startVertex;
	uint64_t *adjList, *edgeOffset, *edgeSize, *distance, *parent, *currentQueue, *nextQueue, *degrees, *incrDegrees;
#ifdef CHECK
	uint64_t *goldDistance, *goldParent, *goldVisited;
#endif
	struct timespec start, end;
	long elapsed;

	if(argc != 3){
		printf("Usage: %s <# edges> <# vertices>\n", argv[0]);
		return 1;
	}

	edges = strtol(argv[1], NULL, 10);
	vertices = strtol(argv[2], NULL, 10);
	uint64_t mem = sizeof(uint64_t)*(8*vertices+2*edges);
	printf("Memory Used (B): %lu\n", mem);

	int count;
	hipGetDeviceCount(&count);
	printf("\nHip Devices Detected: %d\n", count);

	printf("\nAllocating Memory...\n");
	clock_gettime(CLOCK_PROCESS_CPUTIME_ID, &start);
	hipErrchk(hipMallocManaged((void**)&adjList, sizeof(uint64_t)*2*edges));
	hipErrchk(hipMallocManaged((void**)&edgeOffset, sizeof(uint64_t)*vertices));
	hipErrchk(hipMallocManaged((void**)&edgeSize, sizeof(uint64_t)*vertices));
	hipErrchk(hipMallocManaged((void**)&distance, sizeof(uint64_t)*vertices));
	hipErrchk(hipMallocManaged((void**)&parent, sizeof(uint64_t)*vertices));
	hipErrchk(hipMallocManaged((void**)&currentQueue, sizeof(uint64_t)*vertices));
	hipErrchk(hipMallocManaged((void**)&nextQueue, sizeof(uint64_t)*vertices));
	hipErrchk(hipMallocManaged((void**)&degrees, sizeof(uint64_t)*vertices));
	hipErrchk(hipMallocManaged((void**)&incrDegrees, sizeof(uint64_t)*vertices));
#ifdef CHECK
	goldDistance = (uint64_t*)malloc(sizeof(uint64_t)*vertices);
	goldParent = (uint64_t*)malloc(sizeof(uint64_t)*vertices);
	goldVisited = (uint64_t*)malloc(sizeof(uint64_t)*vertices);
#endif
	clock_gettime(CLOCK_PROCESS_CPUTIME_ID, &end);
	elapsed = (end.tv_sec-start.tv_sec)*1e9+(end.tv_nsec-start.tv_nsec);
	printf("\tElapsed time (ns): %ld\n", elapsed);

	printf("\nInitializing Memory...\n");
	clock_gettime(CLOCK_PROCESS_CPUTIME_ID, &start);
	Graph G;
	G.adjList = adjList;
	G.edgeOffset = edgeOffset;
	G.edgeSize = edgeSize;
	populateGraph(&G, vertices, edges);
	initConstData(distance, vertices, ULLONG_MAX);
	initConstData(parent, vertices, ULLONG_MAX);
	startVertex = rand64() % vertices;
	distance[startVertex] = 0;
	parent[startVertex] = 0;
	clock_gettime(CLOCK_PROCESS_CPUTIME_ID, &end);
	elapsed = (end.tv_sec-start.tv_sec)*1e9+(end.tv_nsec-start.tv_nsec);
	printf("\tElapsed time (ns): %ld\n", elapsed);

	printf("\nRunning BFS...\n");
	clock_gettime(CLOCK_PROCESS_CPUTIME_ID, &start);
	uint64_t queueSize = 1;
	uint64_t nextQueueSize = 0;
	uint64_t level = 0;
	currentQueue[0] = startVertex;
	while(queueSize){
		nextLayer<<<queueSize/TPB+1, TPB>>>(level, adjList, edgeOffset, edgeSize, distance, parent, queueSize, currentQueue);
		hipErrchk(hipGetLastError());
		hipDeviceSynchronize();
		countDegrees<<<queueSize/TPB+1, TPB>>>(adjList, edgeOffset, edgeSize, parent, queueSize, currentQueue, degrees);
		hipErrchk(hipGetLastError());
		hipDeviceSynchronize();
		scanDegrees<<<queueSize/TPB+1, TPB>>>(queueSize, degrees, incrDegrees);
		hipErrchk(hipGetLastError());
		hipDeviceSynchronize();
		nextQueueSize = incrDegrees[(queueSize-1)/TPB+1];
		assignVerticesNextQueue<<<queueSize/TPB+1, TPB>>>(adjList, edgeOffset, edgeSize, parent, queueSize, currentQueue, nextQueue, degrees, incrDegrees, nextQueueSize);
		hipErrchk(hipGetLastError());
		hipDeviceSynchronize();
		level += 1;
		queueSize = nextQueueSize;
		std::swap(currentQueue, nextQueue);
	}
	clock_gettime(CLOCK_PROCESS_CPUTIME_ID, &end);
	elapsed = (end.tv_sec-start.tv_sec)*1e9+(end.tv_nsec-start.tv_nsec);
	printf("\tElapsed time (ns): %ld\n", elapsed);

#ifdef CHECK
	printf("Checking Correctness...\n");
	refBfs(startVertex, &G, goldDistance, goldParent, goldVisited);
	bool fail = false;
	uint64_t fc = 0;
	for(uint64_t i = 0; i < vertices; i++){
		if(distance[i] != goldDistance[i]){
			if(!fail) printf("[%lu]: %lu != %lu", i, goldDistance[i], distance[i]);
			fail = true;
			fc += 1;
		}
	}
	if(fail) printf("\tResult: Failed, %lu incorrect entries\n", fc);
	else printf("\tResult: Passed\n");
#else
	printf("Results Unverified\n");
#endif

	hipErrchk(hipFree(adjList));
	hipErrchk(hipFree(edgeOffset));
	hipErrchk(hipFree(edgeSize));
	hipErrchk(hipFree(distance));
	hipErrchk(hipFree(parent));
	hipErrchk(hipFree(currentQueue));
	hipErrchk(hipFree(nextQueue));
	hipErrchk(hipFree(degrees));
	hipErrchk(hipFree(incrDegrees));
#ifdef CHECK
	free(goldDistance);
	free(goldParent);
	free(goldVisited);
#endif
	return 0;
}
