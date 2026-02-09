extern "C" {
#include "problem_handle.h"  
}

#include <math.h>
#include <sys/time.h>

__global__ void backwards(int* table, int* weights, int* values, int cols, int i) {
	int j = blockIdx.x * blockDim.x + threadIdx.x; 

	if (j >= cols)
		return;

	if (weights[i - 1] > j) {
		table[(long) i * cols + j] = table[(long) (i - 1) * cols + j];
	} else {
		table[(long) i * cols + j] = max(table[(long)(i - 1) * cols + (j - weights[i - 1])] + values[i - 1], table[(long)(i - 1) * cols + j]);
	}
}

struct prob read_prob(char *filename, int num_items) {
	FILE *file = fopen(filename, "r");

	struct prob ret;

	ret.num_items = num_items;

	cudaMallocManaged(&ret.weights, sizeof(int) * ret.num_items);
	cudaMallocManaged(&ret.values,  sizeof(int) * ret.num_items);

	int num;	

	for (int i = 0; i < num_items; i++) {
		num = fscanf(file, "%d,%d\n", &ret.weights[i], &ret.values[i]);
		if (num != 2) {
			printf("ERROR");
			exit(1);
		}
	}

	fclose(file);

	return ret;
}

int main(int argc, char* argv[]) {
	char* file;
	int num_items_in_file;
	int wc;

	if (argc == 1) {
		wc = 10;
		file = "data/small.csv";
		num_items_in_file = 10;
	} else if (argc == 4) {
		wc = atoi(argv[1]);
		file = argv[2];
		num_items_in_file = atoi(argv[3]);
	} else {
		printf("ARG_ERROR:\n\t./main <weight cap> <file> <num_items_in_file>\n");
		exit(1);
	}	

	struct prob p = read_prob(file, num_items_in_file);
	p.weight_cap = wc;

	if (argc == 1) {
		p.weight_cap = wc;
	} else {
		p.weight_cap = atoi(argv[1]);
	}

	int table_rows = p.num_items + 1;
	int table_cols = p.weight_cap + 1;

	int threads = 32;
	int blocks = (table_cols / threads) + 1;

	printf("B: %d, T: %d\n", blocks, threads);

	long table_size = (sizeof(int) * table_rows * table_cols);
	int* d_table;
	cudaMallocManaged(&d_table, table_size);

	memset(d_table, 0, table_size);
	d_table[(long)(table_rows - 1) * table_cols + (table_cols - 1)] = 1;

	cudaDeviceSynchronize();
	cudaEvent_t start, stop;
	cudaEventCreate(&start);
	cudaEventCreate(&stop);

	cudaEventRecord(start);
	for (int i = 1; i < table_rows; i++)
		backwards<<<blocks,threads>>>(d_table, p.weights, p.values, table_cols, i);

	cudaDeviceSynchronize();
	cudaEventRecord(stop);
	cudaEventSynchronize(stop);
	float milliseconds = 0;
	cudaEventElapsedTime(&milliseconds, start, stop);
	printf("time: %lfms\n", milliseconds);
	long cells = (long) table_rows * table_cols;
	printf("cell/ms: %lf\n", cells / milliseconds); 

	/*
	for (int i = 0 ; i < table_rows; i++) {
		for (int j = 0; j < table_cols; j++) {
			printf("%d\t", d_table[i * table_cols + j]);
		}
		printf("\n");
	} 
	*/

	printf("ANSWER: %d\n", d_table[(long)(table_rows - 1) * table_cols + (table_cols - 1)]);
}
