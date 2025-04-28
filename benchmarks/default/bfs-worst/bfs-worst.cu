#include <cuda_runtime.h>
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
#include <cstdint>
#define RANGE_SIZE (1<<12)
#define BLOCK_SIZE 1024
#define INF ULLONG_MAX

// CUDA warp == AMD wavefront (MI250x wavefront = 64, `rocminfo` for full hw info)
#define WARP_SIZE 32
#define WARP_SHIFT 5

#define MEM_ALIGN (!(0xfULL))

#define die(x) do{ perror(x); exit(1); }while(0);

#define frand()	(float)rand()/(float)rand()

#define cudaErrchk(ans)                      \
  {                                         \
    cudaAssert((ans), __FILE__, __LINE__);	\
  }

inline uint64_t rand64(){
	return (uint64_t)rand() << 32 | (uint64_t)rand();
}

inline void cudaAssert(cudaError_t code, const char *file, int line, bool abort = true){
  if (code != cudaSuccess) {
    if (abort) {
      std::string msg;
      msg += "CUDAassert: ";
      msg += cudaGetErrorString(code);
      msg += " ";
      msg += file;
      msg += ":";
      msg += std::to_string(line);
      throw std::runtime_error(msg);
    } else {
      fprintf(stderr, "CUDAassert: %s %s %d\n", cudaGetErrorString(code), file, line);
    }
  }
}

__global__ void bfsBase(uint64_t* label, const uint64_t level, const uint64_t vertexCount, const uint64_t* vertexList, const uint64_t* edgeList, bool* changed){
	const uint64_t tid = (uint64_t)blockDim.x * BLOCK_SIZE * (uint64_t)blockIdx.y + blockDim.x * blockIdx.x + threadIdx.x;
	if(tid < vertexCount && label[tid] == level){
		const uint64_t start = vertexList[tid];
		const uint64_t end = tid+1 < vertexCount ? vertexList[tid+1] : edgeList[0];
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

// generates a graph that should have the worst possible performance of any (V,E) graph under SVM
void genGraph(uint64_t* vertexList, uint64_t* edgeList, const uint64_t root, const uint64_t vertexCount, const uint64_t edgeCount){
	uint64_t depth = (uint64_t)RANGE_SIZE/(2UL*sizeof(uint64_t));
	uint64_t breadth = (vertexCount) / (depth);
	if( (vertexCount) % (depth) != 0 ) breadth += 1;
	uint64_t offset = 0;
	for(uint64_t i = 0; i < vertexCount; i++){
		vertexList[i] = offset;
		if(i == root){ // where BFS starts
			for(uint64_t j = 0; j < breadth; j++){
				uint64_t node = j*((uint64_t)RANGE_SIZE/(2UL*sizeof(uint64_t)));
				if(node == root) node += 1;
				edgeList[offset] = node;
				offset += 1;
			}
		}else{
			if((i*2*sizeof(uint64_t)) % RANGE_SIZE == 0){ // nodes certainly connected to the root
				edgeList[offset] = root;
				offset += 1;
				if(i+1 == root){
					if(i+2 != vertexCount){
						edgeList[offset] = i+2;
						offset += 1;
					}
				}else{
					if(i+1 != vertexCount){
						edgeList[offset] = i+1;
						offset += 1;
					}
				}
			}else if( ((i+1)*2*sizeof(uint64_t)) % RANGE_SIZE == 0 || i+1 == vertexCount ){ // nodes that are certainly a leaf
				if(i-1 == root){
					if( ((i-1)*2*sizeof(uint64_t)) % RANGE_SIZE == 0) edgeList[offset] = root;
					else edgeList[offset] = i-2;
				}
				else{
					edgeList[offset] = i-1;
				}
				offset += 1;
			}else{ // all other nodes
				if(i-1 == root){
					if( ((i-1)*2*sizeof(uint64_t)) % RANGE_SIZE == 0) edgeList[offset] = root;
					else edgeList[offset] = i-2;
				}
				else{
					edgeList[offset] = i-1;
				}
				offset += 1;
				if(i+1 == root){
					if(i+2 != vertexCount){
						edgeList[offset] = i+2;
						offset += 1;
					}
				}else{
					if(i+1 != vertexCount){
						edgeList[offset] = i+1;
						offset += 1;
					}
				}
			}
		}
	}
}

void dumpGraph(uint64_t vertices, uint64_t* vertexList, uint64_t* edgeList, FILE* file){
	for(uint64_t i = 0; i < vertices; i++){
		fprintf(file, "%lu: ", i);
		uint64_t start = vertexList[i];
		uint64_t end;
		if(i+1 == vertices){
			end = start+1;
		}else{
			end = vertexList[i+1];
		}
		for(uint64_t j = start; j < end; j++){
			fprintf(file, " %lu,", edgeList[j]);
		}
		fprintf(file, "\n");
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

	if(argc < 3){
		printf("Usage: %s <# edges> <# vertices>\n", argv[0]);
		return 1;
	}

	srand(time(NULL));
	edges = strtol(argv[1], NULL, 10);
	vertices = strtol(argv[2], NULL, 10);
	uint64_t mem = sizeof(uint64_t)*(2*vertices+2*edges);
	printf("Memory Used (B): %lu\n", mem);

	int count;
	cudaErrchk(cudaGetDeviceCount(&count));
	printf("\nHip Devices Detected: %d\n", count);
	
	printf("\nAllocating Memory...\n");
	clock_gettime(CLOCK_PROCESS_CPUTIME_ID, &start);
	cudaErrchk(cudaMallocManaged((void**)&edgeList, sizeof(uint64_t)*2*edges));
	cudaErrchk(cudaMallocManaged((void**)&vertexList, sizeof(uint64_t)*vertices));
	cudaErrchk(cudaMallocManaged((void**)&label, sizeof(uint64_t)*vertices));
	cudaErrchk(cudaMallocManaged((void**)&changed, sizeof(bool)));
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

	
	printf("Size of edgeList:\t%.2f\n", (float)(2 * edges * sizeof(uint64_t)) / (1024. * 1024. * 1024.));
    	printf("Size of vertexList:\t%.2f\n", (float)(vertices * sizeof(uint64_t)) / (1024. * 1024. * 1024.));
    	printf("Size of label:\t%.2f\n", (float)(vertices * sizeof(uint64_t)) / (1024. * 1024. * 1024.));


	
	printf("\nInitializing Memory...\n");
	clock_gettime(CLOCK_PROCESS_CPUTIME_ID, &start);
	startVertex = rand64() % vertices;
	genGraph(vertexList, edgeList, startVertex, vertices, edges);
	for(uint64_t i = 0; i < vertices; i++){
		label[i] = INF;
	}
	label[startVertex] = 0;
	
	clock_gettime(CLOCK_PROCESS_CPUTIME_ID, &end);
	elapsed = (end.tv_sec-start.tv_sec)*1e9+(end.tv_nsec-start.tv_nsec);
	printf("\tElapsed time (ns): %ld\n", elapsed);

	//FILE* file = fopen("./graph.out", "w");
	//dumpGraph(vertices, vertexList, edgeList, file);
	//return 0;

#ifdef CHECK
	struct timespec in, out;
	long iterTime;
#endif
	printf("\nPerforming BFS...\n");
	clock_gettime(CLOCK_PROCESS_CPUTIME_ID, &start);
	uint64_t numblocks = ((vertices * WARP_SIZE + BLOCK_SIZE) / BLOCK_SIZE);
	dim3 blocks(BLOCK_SIZE, (numblocks + BLOCK_SIZE)/BLOCK_SIZE);
	level = 0;
#ifdef CHECK
	int width = 0;
	uint64_t depth = (uint64_t)RANGE_SIZE/(2UL*sizeof(uint64_t));
	if(edges < depth) depth = edges;
	while(depth > 0){
		depth /= 10;
		width += 1;
	}
	depth = (uint64_t)RANGE_SIZE/(2UL*sizeof(uint64_t));
	if(edges < depth) depth = edges;
#endif
	do{
#ifdef CHECK
		clock_gettime(CLOCK_PROCESS_CPUTIME_ID, &in);
#endif
		*changed = false;
		bfsBase<<<blocks, BLOCK_SIZE>>>(label, level, vertices, vertexList, edgeList, changed);
		//bfsCoalesce<<<blocks, BLOCK_SIZE>>>(label, level, vertices, vertexList, edgeList, changed);
		cudaErrchk(cudaGetLastError());
		cudaErrchk(cudaDeviceSynchronize());
#ifdef CHECK
		clock_gettime(CLOCK_PROCESS_CPUTIME_ID, &out);
		iterTime = (out.tv_sec-in.tv_sec)*1e9+(out.tv_nsec-in.tv_nsec);
		printf("\r%*lu / %lu [ %ld ]", width, level, depth, iterTime);
#endif
		level += 1;
		//if(level == 50) break;
	}while(*changed);
#ifdef CHECK
	printf("\n");
#endif
	clock_gettime(CLOCK_PROCESS_CPUTIME_ID, &end);
	elapsed = (end.tv_sec-start.tv_sec)*1e9+(end.tv_nsec-start.tv_nsec);
	printf("GPU Runtime: %0.6lfs\n", elapsed / 1000000000.);

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

	
	cudaErrchk(cudaFree(vertexList));
	cudaErrchk(cudaFree(edgeList));
	cudaErrchk(cudaFree(label));
	cudaErrchk(cudaFree(changed));


#ifdef CHECK
	free(goldLabel);
	free(visited);
#endif

	return 0;
}
