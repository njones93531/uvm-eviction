#include <cuda.h>
#include <cusparse.h>
#include <stdlib.h>
#include <stdio.h>
#include <time.h>
#include <sys/stat.h>
#include <string.h>
#include <unistd.h>
#include <fcntl.h>
#include <sys/mman.h>

#define CPU 0
#define ONE_BASE 1
#define ERROR_THRESHOLD 0.001
#define ALGORITHM CUSPARSE_SPMV_COO_ALG2

// Check the status returned from CUDA runtime function calls
#define CHECK_CUDA(func)                                                 \
{                                                                       \
    cudaError_t status = (func);                                        \
    if (status != cudaSuccess) {                                        \
        printf("CUDA API failed at line %d with error: %s (%d)\n",      \
               __LINE__, cudaGetErrorString(status), status);           \
        return EXIT_FAILURE;                                            \
    }                                                                   \
}

#define CHECK_CUSPARSE(func)                                             \
{                                                                       \
    cusparseStatus_t status = (func);                                   \
    if (status != CUSPARSE_STATUS_SUCCESS) {                             \
        printf("CUSPARSE API failed at line %d with error: %s (%d)\n",   \
               __LINE__, cusparseGetErrorString(status), status);        \
        return EXIT_FAILURE;                                            \
    }                                                                   \
}

void usage(char *argv[]) {
    fprintf(stderr, "Usage: %s <infile>\n", argv[0]);
    exit(1);
}


double getProblemSize(size_t N, size_t nnz) {
    double alloc0 = (double)nnz * sizeof(float);
    double alloc1 = (double)nnz * sizeof(int);
    double alloc2 = (double)nnz * sizeof(int);
    double alloc3 = (double)N * sizeof(float);
    double alloc4 = (double)N * sizeof(float);
    return (alloc0 + alloc1 + alloc2 + alloc3 + alloc4) / (1024. * 1024. * 1024.);
}

// CPU-based SPMV
void cpu_spmv(const size_t num_rows, const size_t nnz,
              const float *values, const int *col_indices, const int *row_indices,
              const float *x, float *y) {
    for (size_t i = 0; i < nnz; ++i) {
        size_t row = row_indices[i];
        size_t col = col_indices[i];
	if(ONE_BASE){
	    row--;
	    col--;
	}
        y[row] += values[i] * x[col];
    }
}

// Comparison function for qsort
int compare(const void *a, const void *b) {
    return (*(int*)a - *(int*)b);
}

void adv_cursor(char ** cursor){ 
    char * tmp = *cursor;
    while(*tmp!='\n') tmp++;
    *cursor = tmp+1;
}

int main(int argc, char *argv[]) {
    // User inputs: size and density
    if (argc != 2) {
        usage(argv);
    }
    size_t N;
    size_t nnz;
    size_t fsize;
    struct stat st;
    int infile = open(argv[1], O_RDONLY);
    float alpha = 1.0f;
    float beta = 0.0f;

    // Validate input
    if (infile == -1) {
        usage(argv);
    }

    //Read in entire file
    stat(argv[1], &st);
    fsize = st.st_size;
    printf("Reading input of %.2f GB\n", (double)fsize / (1024. * 1024. * 1024.));
    char * buffer = (char *) mmap (0, fsize, PROT_READ, MAP_PRIVATE, infile, 0);
    close(infile);
    printf("Finished reading input\n");

    // Generate sparse matrix and vector
    float *h_values;
    int *h_col_indices;
    int *h_row_indices;
    float *h_x;
    float *h_y;
    float *h_cpu_y;
    char * cursor = buffer;
    //Skip the .mtx comments and get to the dims line
    int comments = 0; 
    while(cursor[0] == '%'){ 
	    adv_cursor(&cursor);
	    comments++;
    }
    printf("%d comment lines skipped\n", comments);
    sscanf(cursor, "%zu %zu %zu", &N, &N, &nnz);

    printf("Problem size: %.2f GB\n", getProblemSize(N, nnz));

    CHECK_CUDA(cudaMallocManaged((void **)&h_values, nnz * sizeof(float), cudaMemAttachGlobal));
    CHECK_CUDA(cudaMallocManaged((void **)&h_col_indices, nnz * sizeof(int), cudaMemAttachGlobal));
    CHECK_CUDA(cudaMallocManaged((void **)&h_row_indices, nnz * sizeof(int), cudaMemAttachGlobal));
    CHECK_CUDA(cudaMallocManaged((void **)&h_x, N * sizeof(float), cudaMemAttachGlobal));
    CHECK_CUDA(cudaMallocManaged((void **)&h_y, N * sizeof(float), cudaMemAttachGlobal));
    if (CPU) {
        CHECK_CUDA(cudaMallocManaged((void **)&h_cpu_y, N * sizeof(float), cudaMemAttachGlobal));
    }
    printf("Building matrix\n");
    for (size_t i = 0; i < nnz; ++i) {
	adv_cursor(&cursor);
	char * pEnd;
	h_col_indices[i] = (uint) strtol(cursor, &pEnd, 10);
	h_row_indices[i] = (uint) strtol(pEnd, &pEnd, 10);
        h_values[i] = (float)rand() / RAND_MAX;
    }
    for (size_t i = 0; i < N; ++i) {
        h_x[i] = (float)rand() / RAND_MAX;
        h_y[i] = 0.0; // Initialize output vector
    }
    munmap(buffer, fsize);
    printf("Finished building matrix\n");
    //.mtx files should be sorted by column
    //We tranpose, so matrix should be sorted already
    //qsort(h_row_indices, nnz, sizeof(int), compare);

    // CUSPARSE APIs
    cusparseHandle_t handle = NULL;
    cusparseSpMatDescr_t matA;
    cusparseDnVecDescr_t vecX, vecY;
    void *dBuffer = NULL;
    size_t bufferSize = 0;
    CHECK_CUSPARSE(cusparseCreate(&handle))
    // Create sparse matrix A in COO format
    if(ONE_BASE) CHECK_CUSPARSE(cusparseCreateCoo(&matA, N, N, nnz,
                                     h_row_indices, h_col_indices, h_values, CUSPARSE_INDEX_32I, CUSPARSE_INDEX_BASE_ONE, CUDA_R_32F))
    else CHECK_CUSPARSE(cusparseCreateCoo(&matA, N, N, nnz,
                                     h_row_indices, h_col_indices, h_values, CUSPARSE_INDEX_32I, CUSPARSE_INDEX_BASE_ZERO, CUDA_R_32F))

    // Create dense vector X
    CHECK_CUSPARSE(cusparseCreateDnVec(&vecX, N, h_x, CUDA_R_32F))
    // Create dense vector y
    CHECK_CUSPARSE(cusparseCreateDnVec(&vecY, N, h_y, CUDA_R_32F))
    // Allocate an external buffer if needed
    CHECK_CUSPARSE(cusparseSpMV_bufferSize(
                         handle, CUSPARSE_OPERATION_NON_TRANSPOSE,
                         &alpha, matA, vecX, &beta, vecY, CUDA_R_32F,
                         ALGORITHM, &bufferSize))
    CHECK_CUDA(cudaMallocManaged(&dBuffer, bufferSize, cudaMemAttachGlobal))
    clock_t begin = clock();
    // Execute SpMV
    CHECK_CUSPARSE(cusparseSpMV(handle, CUSPARSE_OPERATION_NON_TRANSPOSE,
                                &alpha, matA, vecX, &beta, vecY, CUDA_R_32F,
                                ALGORITHM, dBuffer))

    CHECK_CUDA(cudaDeviceSynchronize())
    clock_t end = clock();
    double kernel_time = (double)(end - begin) / CLOCKS_PER_SEC;
    // Destroy matrix/vector descriptors
    CHECK_CUSPARSE(cusparseDestroySpMat(matA))
    CHECK_CUSPARSE(cusparseDestroyDnVec(vecX))
    CHECK_CUSPARSE(cusparseDestroyDnVec(vecY))
    CHECK_CUSPARSE(cusparseDestroy(handle))

    int correct = 1;
    printf("SPMV computed in %.2f seconds\n", kernel_time);

// Wait for GPU to finish and check for any errors
 
   if(CPU){
         cpu_spmv(N, nnz, h_values, h_col_indices, h_row_indices, h_x, h_cpu_y);
        // device result check
        for (size_t i = 0; i < N; i++) {
		double abs_err = fabs(h_y[i] - h_cpu_y[i]);
		double relative_err = abs_err / (double)h_cpu_y[i];
            if (relative_err > ERROR_THRESHOLD) { // direct floating point comparison is not
                correct = 0;            // reliable
                //printf("%zu differs by %.4f. Device: %.4f CPU: %.4f\n",i,abs_err, h_y[i], h_cpu_y[i]);
                break;

    	    }
    	}
    }
    if (correct)
    	printf("spmv_csr test PASSED\n");
    else
        printf("spmv_csr test FAILED: wrong result\n");

    
    // Free memory
    CHECK_CUDA( cudaFree(h_values) )
    CHECK_CUDA( cudaFree(h_col_indices) )
    CHECK_CUDA( cudaFree(h_row_indices) )
    CHECK_CUDA( cudaFree(h_x) )
    CHECK_CUDA( cudaFree(h_y) )

    return 0;
}
