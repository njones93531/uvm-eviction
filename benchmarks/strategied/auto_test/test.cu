
int main(void) {
	CHECK_CUDA( cudaMallocManaged((void **) &h_values, nnz * sizeof(float), cudaMemAttachGlobal) )
}

for (int i = 0; i < 10; i++) {
	//do something
	h_values[i] = 1;
}


CHECK_CUDA( cudaMallocManaged((void **) &h_col_indices, nnz * sizeof(int),cudaMemAttachGlobal) )

CHECK_CUDA( cudaMallocManaged((void **) &h_row_offsets, (N + 1) * sizeof(int), cudaMemAttachGlobal) )
