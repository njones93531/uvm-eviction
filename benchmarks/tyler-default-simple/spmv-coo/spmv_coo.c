#include <cuda.h>
#include <cusparse.h>
#include <stdlib.h>
#include <stdio.h>
#include <time.h>

#define CPU 1
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
    fprintf(stderr, "Usage: %s <psize (float, GB)> <density (float)>\n", argv[0]);
    fprintf(stderr, "Density must be large enough that nnz > N\n");
    exit(1);
}

// Formula comes from completing the square of memory formula
// Assumes all memory allocations have the same size type (4 bytes)
size_t getN(double psize, double density) {
    double a = 2. / (4 * density);
    double b = ((psize / sizeof(float)) - 1.) / (3 * density);
    double result = sqrt((a * a) + b) - a;
    return (size_t)result;
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
        y[row] += values[i] * x[col];
    }
}

// Comparison function for qsort
int compare(const void *a, const void *b) {
    return (*(int*)a - *(int*)b);
}

int main(int argc, char *argv[]) {
    // User inputs: size and density
    if (argc != 3) {
        usage(argv);
    }
    double psize = atof(argv[1]);
    double density = atof(argv[2]);
    psize *= 1024. * 1024. * 1024.;
    size_t N = getN(psize, density);
    size_t nnz = (size_t)(density * (double)N * (double)N);
    printf("Problem size: %.2f GB\n", getProblemSize(N, nnz));
    float alpha = 1.0f;
    float beta = 0.0f;

    // Validate input
    if (N <= 0 || nnz < N || density > 1) {
        usage(argv);
    }

    // Generate a random sparse matrix and vectors.
    srand(time(NULL));
    float *h_values;
    int *h_col_indices;
    int *h_row_indices;
    float *h_x;
    float *h_y;
    float *h_cpu_y;

    CHECK_CUDA(cudaMallocManaged((void **)&h_values, nnz * sizeof(float), cudaMemAttachGlobal));
    CHECK_CUDA(cudaMallocManaged((void **)&h_col_indices, nnz * sizeof(int), cudaMemAttachGlobal));
    CHECK_CUDA(cudaMallocManaged((void **)&h_row_indices, nnz * sizeof(int), cudaMemAttachGlobal));
    CHECK_CUDA(cudaMallocManaged((void **)&h_x, N * sizeof(float), cudaMemAttachGlobal));
    CHECK_CUDA(cudaMallocManaged((void **)&h_y, N * sizeof(float), cudaMemAttachGlobal));
    if (CPU) {
        CHECK_CUDA(cudaMallocManaged((void **)&h_cpu_y, N * sizeof(float), cudaMemAttachGlobal));
    }
    printf("Start address of h_values:\t%p\n", &(h_values[0]));
    printf("Start address of h_col_indices:\t%p\n", &(h_col_indices[0]));
    printf("Start address of h_row_indices:\t%p\n", &(h_row_indices[0]));
    printf("Start address of h_x:\t%p\n", &(h_x[0]));
    printf("Start address of h_y:\t%p\n", &(h_y[0]));

    printf("Size of h_values:\t%.2f\n", (float)nnz * sizeof(float) / (1024. * 1024. * 1024.));
    printf("Size of h_col_indices:\t%.2f\n", (float)nnz * sizeof(int) / (1024. * 1024. * 1024.));
    printf("Size of h_row_indices:\t%.2f\n", (float)nnz * sizeof(int) / (1024. * 1024. * 1024.));
    printf("Size of h_x:\t%.2f\n", (float)N * sizeof(float) / (1024. * 1024. * 1024.));
    printf("Size of h_y:\t%.2f\n", (float)N * sizeof(float) / (1024. * 1024. * 1024.));


    for (size_t i = 0; i < nnz; ++i) {
        h_values[i] = (float)rand() / RAND_MAX;
        h_col_indices[i] = rand() % N;
        h_row_indices[i] = rand() % N;
    }
    for (size_t i = 0; i < N; ++i) {
        h_x[i] = (float)rand() / RAND_MAX;
        h_y[i] = 0.0; // Initialize output vector
    }
    qsort(h_row_indices, nnz, sizeof(int), compare);

    // CUSPARSE APIs
    cusparseHandle_t handle = NULL;
    cusparseSpMatDescr_t matA;
    cusparseDnVecDescr_t vecX, vecY;
    void *dBuffer = NULL;
    size_t bufferSize = 0;
    CHECK_CUSPARSE(cusparseCreate(&handle))
    // Create sparse matrix A in COO format
    CHECK_CUSPARSE(cusparseCreateCoo(&matA, N, N, nnz,
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
    printf("Start address of buffer:\t%p\n", &(dBuffer));

    // Execute SpMV
    CHECK_CUSPARSE(cusparseSpMV(handle, CUSPARSE_OPERATION_NON_TRANSPOSE,
                                &alpha, matA, vecX, &beta, vecY, CUDA_R_32F,
                                ALGORITHM, dBuffer))

    CHECK_CUDA(cudaDeviceSynchronize())
    // Destroy matrix/vector descriptors
    CHECK_CUSPARSE(cusparseDestroySpMat(matA))
    CHECK_CUSPARSE(cusparseDestroyDnVec(vecX))
    CHECK_CUSPARSE(cusparseDestroyDnVec(vecY))
    CHECK_CUSPARSE(cusparseDestroy(handle))

    int correct = 1;

// Wait for GPU to finish and check for any errors
 
   if(CPU){
         cpu_spmv(N, nnz, h_values, h_col_indices, h_row_indices, h_x, h_cpu_y);
        // device result check
        for (size_t i = 0; i < N; i++) {
		double abs_err = fabs(h_y[i] - h_cpu_y[i]);
		double relative_err = abs_err / (double)h_cpu_y[i];
            if (relative_err > ERROR_THRESHOLD) { // direct floating point comparison is not
                correct = 0;            // reliable
                printf("%zu differs by %.4f. Device: %.4f CPU: %.4f\n",i,abs_err, h_y[i], h_cpu_y[i]);
                //break;

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
