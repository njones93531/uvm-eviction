#include <cuda.h>
#include <cuda_runtime.h>
#include <iostream>
#include <chrono>
#include <random>

// Define hash table entry as a struct
struct HashEntry {
    int valid;
    int values[];  // Flexible array member (size defined dynamically)
};

// Simple hash function
__device__ __host__ inline uint32_t hash(uint32_t key, uint32_t table_size) {
    return key * 2654435761 % table_size;
}

// CUDA kernel to insert keys into the hash table
__global__ void hash_insert(HashEntry* table, uint32_t* keys, int num_keys, int table_size, int value_size) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= num_keys) return;

    uint32_t key = keys[idx];
    uint32_t h = hash(key, table_size);
    
    while (atomicCAS(&table[h].valid, 0, 1) != 0) {
        h = (h + 1) % table_size;  // Linear probing
    }

    // Store large value array
    for (int i = 0; i < value_size; i++) {
        table[h].values[i] = key + i;
    }
}

// CUDA kernel to look up keys in the hash table
__global__ void hash_lookup(HashEntry* table, uint32_t* keys, int* results, int num_keys, int table_size, int value_size) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= num_keys) return;

    uint32_t key = keys[idx];
    uint32_t h = hash(key, table_size);

    while (table[h].valid == 1 && table[h].values[0] != key) {
        h = (h + 1) % table_size;
    }

    results[idx] = (table[h].valid == 1 && table[h].values[0] == key);
}

int main(int argc, char* argv[]) {
    if (argc != 4) {
        std::cerr << "Usage: " << argv[0] << " <table_size> <num_keys> <value_size>\n";
        return 1;
    }

    // Parse command-line arguments
    int TABLE_SIZE = std::stoi(argv[1]);
    int NUM_KEYS = std::stoi(argv[2]);
    int VALUE_SIZE = std::stoi(argv[3]);

    std::cout << "Running with:\n";
    std::cout << "  Table Size: " << TABLE_SIZE << "\n";
    std::cout << "  Number of Keys: " << NUM_KEYS << "\n";
    std::cout << "  Value Size: " << VALUE_SIZE << "\n";
    std::cout << "Problem Size:%.2f\n" << TABLE_SIZE * sizeof(HashEntry) + VALUE_SIZE * sizeof(int) + NUM_KEYS * sizeof(uint32_t) * 2 << "\n";

    // Allocate hash table in UVM (oversubscription possible)
    HashEntry* d_table;
    cudaMallocManaged(&d_table, TABLE_SIZE * sizeof(HashEntry) + VALUE_SIZE * sizeof(int));

    // Initialize table
    for (int i = 0; i < TABLE_SIZE; i++) {
        d_table[i].valid = 0;
    }

    // Allocate keys in UVM
    uint32_t* d_keys;
    cudaMallocManaged(&d_keys, NUM_KEYS * sizeof(uint32_t));

    // Allocate results for lookups
    int* d_results;
    cudaMallocManaged(&d_results, NUM_KEYS * sizeof(int));

    // Generate random keys
    std::random_device rd;
    std::mt19937 gen(rd());
    std::uniform_int_distribution<uint32_t> dist(1, UINT32_MAX);
    for (int i = 0; i < NUM_KEYS; i++) {
        d_keys[i] = dist(gen);
    }

    // Prefetch keys to CPU
    cudaMemPrefetchAsync(d_keys, NUM_KEYS * sizeof(uint32_t), cudaCpuDeviceId, 0);

    // Launch insertion kernel
    int block_size = 256;
    int num_blocks = (NUM_KEYS + block_size - 1) / block_size;

    cudaEvent_t start, stop;
    cudaEventCreate(&start);
    cudaEventCreate(&stop);

    cudaEventRecord(start);
    hash_insert<<<num_blocks, block_size>>>(d_table, d_keys, NUM_KEYS, TABLE_SIZE, VALUE_SIZE);
    cudaEventRecord(stop);
    cudaEventSynchronize(stop);

    float insert_time = 0;
    cudaEventElapsedTime(&insert_time, start, stop);
    std::cout << "Insertion Time: " << insert_time << " ms\n";

    // Launch lookup kernel
    cudaEventRecord(start);
    hash_lookup<<<num_blocks, block_size>>>(d_table, d_keys, d_results, NUM_KEYS, TABLE_SIZE, VALUE_SIZE);
    cudaEventRecord(stop);
    cudaEventSynchronize(stop);

    float lookup_time = 0;
    cudaEventElapsedTime(&lookup_time, start, stop);
    std::cout << "Lookup Time: " << lookup_time << " ms\n";

    // Cleanup
    cudaFree(d_table);
    cudaFree(d_keys);
    cudaFree(d_results);

    return 0;
}

