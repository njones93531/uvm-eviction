

int main(void) {
	return 1;
}

CHECK_CUDA( cudaMallocManaged((void**) &hA_csrOffsets, (A_num_rows + 1) * sizeof(int), cudaMemAttachGlobal))
    CHECK_CUDA( cudaMallocManaged((void**) &hA_columns, A_nnz * sizeof(int), cudaMemAttachGlobal))
    CHECK_CUDA( cudaMallocManaged((void**) &hA_values,  A_nnz * sizeof(float), cudaMemAttachGlobal))
    CHECK_CUDA( cudaMallocManaged((void**) &hB,         B_size * sizeof(float), cudaMemAttachGlobal))
    CHECK_CUDA( cudaMallocManaged((void**) &hC,         C_size * sizeof(float), cudaMemAttachGlobal))
    if (CPU) {
        CHECK_CUDA( cudaMallocManaged((void**) &hC_result,         C_size * sizeof(float), cudaMemAttachGlobal))
    }

    // PLACE ALLOCATION POLICY

blah blah blah
