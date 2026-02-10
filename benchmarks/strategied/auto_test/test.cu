CHECK_CUDA( cudaMallocManaged((void **) &h_values, nnz * sizeof(float), cudaMemAttachGlobal) )

CHECK_CUDA( cudaMallocManaged((void **) &h_col_indices, nnz * sizeof(int),cudaMemAttachGlobal) )

CHECK_CUDA( cudaMallocManaged((void **) &h_row_offsets, (N + 1) * sizeof(int), cudaMemAttachGlobal) )
