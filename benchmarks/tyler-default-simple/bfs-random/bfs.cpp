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
#define BLOCK_SIZE 1024
#define INF ULLONG_MAX

// CUDA warp == AMD wavefront (MI250x wavefront = 64, `rocminfo` for full hw info)
#define WARP_SIZE 64
#define WARP_SHIFT 6

#define MEM_ALIGN (!(0xfULL))

#define die(x) do{ perror(x); exit(1); }while(0);

#define frand()	(float)rand()/(float)rand()

#define hipErrchk(ans)                      \
  {                                         \
    hipAssert((ans), __FILE__, __LINE__);	\
  }

inline uint64_t rand64(){
	return (uint64_t)rand() << 32 | (uint64_t)rand();
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

__global__ void bfsBase(uint64_t* label, const uint64_t level, const uint64_t vertexCount, const uint64_t* vertexList, const uint64_t* edgeList, bool* changed){
	const uint64_t tid = (uint64_t)blockDim.x * BLOCK_SIZE * (uint64_t)blockIdx.y + blockDim.x * blockIdx.x + threadIdx.x;
	if(tid < vertexCount && label[tid] == level){
		const uint64_t start = vertexList[tid];
		const uint64_t end = vertexList[tid+1];
		for(uint64_t i = start; i < end; i++){
			const uint64_t next = edgeList[i];
			if(label[next] == INF){
				label[next] = level+1;
				*changed = true;
			}
		}
	}
}

__global__ void bfsCoalesce(uint64_t* label, const uint64_t level, const uint64_t vertexCount, const uint64_t* vertexList, const uint64_t* edgeList, bool* changed){
	const uint64_t tid = (uint64_t)blockDim.x * BLOCK_SIZE * (uint64_t)blockIdx.y + blockDim.x * blockIdx.x + threadIdx.x;
	const uint64_t warpIdx = tid >> WARP_SHIFT;
	uint64_t laneIdx = tid & ((1 << WARP_SHIFT)-1);
	if(warpIdx < vertexCount && label[warpIdx] == level){
		const uint64_t start = vertexList[warpIdx];
		const uint64_t shiftStart = start & MEM_ALIGN;
		const uint64_t end = vertexList[warpIdx+1];
		for(uint64_t i = shiftStart + laneIdx; i < end; i+=WARP_SIZE){
			if(i >= start){
				const uint64_t next = edgeList[i];
				if(label[next] == INF){
					label[next] = level+1;
					*changed = true;
				}
			}
		}
	}
}

__global__ void bfsCoalesceChunk(uint64_t* label, const uint64_t level, const uint64_t vertexCount, const uint64_t* vertexList, const uint64_t* edgeList, bool* changed){
	return;
}

// generates a graph in CSR with specified number of vertices and edges, handles single node edges but not uniqueness
void genGraph(uint64_t* vertexList, uint64_t* edgeList, const uint64_t vertexCount, const uint64_t edgeCount){
	std::vector<std::vector<uint64_t>> adjacencyLists(vertexCount);
	for(uint64_t i = 0; i < edgeCount; i++){
again:
		uint64_t u = rand64() % vertexCount;
		uint64_t v = rand64() % vertexCount;
		if(u == v) goto again;
		adjacencyLists[u].push_back(v);
		adjacencyLists[v].push_back(u);
	}
	uint64_t offset = 0;
	for(uint64_t i = 0; i < vertexCount; i++){
		vertexList[i] = offset;
		for(auto &edge: adjacencyLists[i]){
			edgeList[offset] = edge;
			offset += 1;
		}
	}
}

void refBfs(const uint64_t startVertex, const uint64_t vertexCount, const uint64_t* vertexList, const uint64_t* edgeList, uint64_t* label, bool* visited){
	for(uint64_t i = 0; i < vertexCount; i++){
		label[i] = INF;
		visited[i] = false;
	}
	label[startVertex] = 0;
	visited[startVertex] = true;
	std::queue<uint64_t> Q;
	Q.push(startVertex);
	while(!Q.empty()){
		uint64_t u = Q.front();
		uint64_t start = vertexList[u];
		uint64_t end = vertexList[u+1];
		Q.pop();
		for(uint64_t i = start; i < end; i++){
			uint64_t v = edgeList[i];
			if(!visited[v]){
				visited[v] = true;
				label[v] = label[u] + 1;
				Q.push(v);
			}
		}
	}
}

int main(int argc, char* argv[]){
	uint64_t edges, vertices, startVertex, level;
	uint64_t *vertexList, *edgeList, *label;
	bool *changed;
#ifdef CHECK
	uint64_t *goldLabel;
	bool *visited;
#endif
	struct timespec start, end;
	long elapsed;

	if(argc != 3){
		printf("Usage: %s <# edges> <# vertices>\n", argv[0]);
		return 1;
	}

	srand(time(NULL));
	edges = strtol(argv[1], NULL, 10);
	vertices = strtol(argv[2], NULL, 10);
	uint64_t mem = sizeof(uint64_t)*(2*vertices+2*edges);
	printf("Memory Used (B): %lu\n", mem);

	int count;
	hipGetDeviceCount(&count);
	printf("\nHip Devices Detected: %d\n", count);
	
	printf("\nAllocating Memory...\n");
	clock_gettime(CLOCK_PROCESS_CPUTIME_ID, &start);
	hipErrchk(hipMallocManaged((void**)&edgeList, sizeof(uint64_t)*2*edges));
	hipErrchk(hipMallocManaged((void**)&vertexList, sizeof(uint64_t)*vertices));
	hipErrchk(hipMallocManaged((void**)&label, sizeof(uint64_t)*vertices));
	hipErrchk(hipMallocManaged((void**)&changed, sizeof(bool)));
#ifdef CHECK
	goldLabel = (uint64_t*)malloc(sizeof(uint64_t)*vertices);
	visited = (bool*)malloc(sizeof(bool)*vertices);
#endif
	clock_gettime(CLOCK_PROCESS_CPUTIME_ID, &end);
	elapsed = (end.tv_sec-start.tv_sec)*1e9+(end.tv_nsec-start.tv_nsec);
	printf("\tElapsed time (ns): %ld\n", elapsed);

	printf("\nAllocations:\n");
	printf("\tAllocation \"EdgeList\": %p - %p\n", edgeList, edgeList+2*edges);
	printf("\tAllocation \"VertexList\": %p - %p\n", vertexList, vertexList+vertices);
	printf("\tAllocation \"Labels\": %p - %p\n", label, label+vertices);

	printf("\nInitializing Memory...\n");
	clock_gettime(CLOCK_PROCESS_CPUTIME_ID, &start);
	genGraph(vertexList, edgeList, vertices, edges);
	startVertex = rand64() % vertices;
	for(uint64_t i = 0; i < vertices; i++){
		label[i] = INF;
	}
	label[startVertex] = 0;
	clock_gettime(CLOCK_PROCESS_CPUTIME_ID, &end);
	elapsed = (end.tv_sec-start.tv_sec)*1e9+(end.tv_nsec-start.tv_nsec);
	printf("\tElapsed time (ns): %ld\n", elapsed);

	printf("\nPerforming BFS...\n");
	clock_gettime(CLOCK_PROCESS_CPUTIME_ID, &start);
	uint64_t numblocks = ((vertices * WARP_SIZE + BLOCK_SIZE) / BLOCK_SIZE);
	dim3 blocks(BLOCK_SIZE, (numblocks + BLOCK_SIZE)/BLOCK_SIZE);
	level = 0;
	do{
		*changed = false;
		bfsBase<<<blocks, BLOCK_SIZE>>>(label, level, vertices, vertexList, edgeList, changed);
		hipErrchk(hipGetLastError());
		hipDeviceSynchronize();
		level += 1;
	}while(*changed);
	clock_gettime(CLOCK_PROCESS_CPUTIME_ID, &end);
	elapsed = (end.tv_sec-start.tv_sec)*1e9+(end.tv_nsec-start.tv_nsec);
	printf("\tElapsed time (ns): %ld\n", elapsed);

#ifdef CHECK
	printf("Checking Correctness...\n");
	refBfs(startVertex, vertices, vertexList, edgeList, goldLabel, visited);
	bool fail = false;
	uint64_t fc = 0;
	for(uint64_t i = 0; i < vertices; i++){
		if(label[i] != goldLabel[i]){
			if(!fail) printf("[%lu]: %lu != %lu", i, goldLabel[i], label[i]);
			fail = true;
			fc += 1;
		}
	}
	if(fail) printf("\tResult: Failed, %lu incorrect entries\n", fc);
	else printf("\tResult: Passed\n");
#else
	printf("Results Unverified\n");
#endif

	
	hipErrchk(hipFree(vertexList));
	hipErrchk(hipFree(edgeList));
	hipErrchk(hipFree(label));
	hipErrchk(hipFree(changed));
#ifdef CHECK
	free(goldLabel);
	free(visited);
#endif

	return 0;
}
