/*
 * SPDX-FileCopyrightText: Copyright (c) 1993-2022 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 *
 * Licensed under the Apache License, Version 2.0 (the "License");
 * you may not use this file except in compliance with the License.
 * You may obtain a copy of the License at
 *
 * http://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS,
 * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 * See the License for the specific language governing permissions and
 * limitations under the License.
 */

#include <cuda_runtime_api.h> // cudaMalloc, cudaMemcpy, etc.
#include <cusparse.h>         // cusparseSpMM
#include <stdio.h>            // printf
#include <stdlib.h>           // EXIT_FAILURE
#include <time.h>

#define ERROR_THRESHOLD 0.001

#define CHECK_CUDA(func)                                                       \
{                                                                              \
    cudaError_t status = (func);                                               \
    if (status != cudaSuccess) {                                               \
        printf("CUDA API failed at line %d with error: %s (%d)\n",             \
               __LINE__, cudaGetErrorString(status), status);                  \
        return EXIT_FAILURE;                                                   \
    }                                                                          \
}

#define CHECK_CUSPARSE(func)                                                   \
{                                                                              \
    cusparseStatus_t status = (func);                                          \
    if (status != CUSPARSE_STATUS_SUCCESS) {                                   \
        printf("CUSPARSE API failed at line %d with error: %s (%d)\n",         \
               __LINE__, cusparseGetErrorString(status), status);              \
        return EXIT_FAILURE;                                                   \
    }                                                                          \
}

// CPU-based SPMM (CSR * Dense)
//	cuSparse Dense matrices are col major btw
void cpu_spmm(const size_t num_rows, const size_t num_nonzero, const size_t num_cols_B,
              const float* values, const int* col_indices, const int* row_offsets,
              const float* B, float* C, const int B_ld) {
    for (size_t i = 0; i < num_rows; ++i) {
        for (size_t j = row_offsets[i]; j < row_offsets[i + 1]; ++j) {
            size_t col_A = col_indices[j];
            float val_A = values[j];
            for (size_t k = 0; k < num_cols_B; ++k) {
                //C[i * num_cols_B + k] += val_A * B[col_A * num_cols_B + k];
                C[i + k * num_rows] += val_A * B[col_A + k * B_ld];
            }
        }
    }
}

float get_size(int A_r, int A_nnz, int Bs, int Cs) {
    double alloc0 = (double) (A_r + 1) * sizeof(int);	
    double alloc1 = (double) A_nnz * sizeof(int);	
    double alloc2 = (double) A_nnz * sizeof(float);	
    double alloc3 = (double) Bs * sizeof(float);	
    double alloc4 = (double) Cs * sizeof(float);	
    return (alloc0 + alloc1 + alloc2 + alloc3 + alloc4) / (1024. * 1024. * 1024.);
}

int main(int argc, char *argv[]) {
    if (argc != 6) {
        fprintf(stderr,
                "Usage: %s <A_num_rows> <A_num_cols> <A_nnz> <B_num_cols> <CPU>\n",
                argv[0]);
        return EXIT_FAILURE;
    }

    int CPU = atoi(argv[5]);

    int   A_num_rows      = atoi(argv[1]);
    int   A_num_cols      = atoi(argv[2]);
    int   A_nnz           = atoi(argv[3]);
    int   B_num_rows      = A_num_cols;
    int   B_num_cols      = atoi(argv[4]);
    int   ldb             = B_num_rows;
    int   ldc             = A_num_rows;
    int   B_size          = ldb * B_num_cols;
    int   C_size          = ldc * B_num_cols;

    printf("A: %d x %d, nnz = %d, density: %f\n", A_num_rows, A_num_cols, A_nnz, (float)A_nnz / (float)(A_num_rows * A_num_cols));
    printf("B: %d x %d\n", B_num_rows, B_num_cols);
    printf("C: %d x %d\n", A_num_rows, B_num_cols);

    printf("size: %f\n", get_size(A_num_rows, A_nnz, B_size, C_size));

    if (A_nnz > (long long)A_num_rows * A_num_cols) {
        fprintf(stderr,
				"Invalid nnz: %d > %d * %d\n",
			A_nnz, A_num_rows, A_num_cols);
        return EXIT_FAILURE;
    }
    if (A_nnz < A_num_rows) {
	fprintf(stderr,
				"Invalid nnz: %d > %d * %d\n",
			A_nnz, A_num_rows, A_num_cols);
        return EXIT_FAILURE;
    }

    srand(time(NULL));
    int*   hA_csrOffsets;
    int*   hA_columns;
    float* hA_values;
    float* hB;
    float* hC;
    float* hC_result;
    float alpha           = 1.0f;
    float beta            = 0.0f;
    //--------------------------------------------------------------------------
    // Device memory management
    int   *dA_csrOffsets, *dA_columns;
    float *dA_values, *dB, *dC;

    int before = (A_num_rows + 1);
    hA_csrOffsets = (int*)malloc(before * sizeof(int));
    hA_columns    = (int*)malloc(A_nnz * sizeof(int));
    hA_values     = (float*)malloc(A_nnz * sizeof(float));
    hB            = (float*)malloc(B_size * sizeof(float));
    hC            = (float*)malloc(C_size * sizeof(float));

    if (CPU) {
        hC_result = (float*)malloc(C_size * sizeof(float));
    }

    
    CHECK_CUDA(cudaMalloc((void**)&dA_csrOffsets, before * sizeof(int)));
    CHECK_CUDA(cudaMalloc((void**)&dA_columns,    A_nnz * sizeof(int)));
    CHECK_CUDA(cudaMalloc((void**)&dA_values,     A_nnz * sizeof(float)));
    CHECK_CUDA(cudaMalloc((void**)&dB,            B_size * sizeof(float)));
    CHECK_CUDA(cudaMalloc((void**)&dC,            C_size * sizeof(float)));


    int nnz_per_row[A_num_rows];
    int remaining = A_nnz;

    for (int i = 0; i < A_num_rows; i++) {
	nnz_per_row[i] = 1;
	remaining--;
    }

    for (int i = 0; i< remaining; i++) {
	int r = rand() % A_num_rows;
	nnz_per_row[r]++;
    }

    hA_csrOffsets[0] = 0;
    for (int i = 0; i < A_num_rows; i++) {
	hA_csrOffsets[i+1] = hA_csrOffsets[i] + nnz_per_row[i];
    }

    for (int i = 0; i < A_num_rows; i++) {
        int start = hA_csrOffsets[i];
        int end   = hA_csrOffsets[i+1];
        for (int j = start; j < end; j++) {
			hA_values[j] = (float)rand() / RAND_MAX;
		hA_columns[j] = rand() % A_num_cols; // must be < A_num_cols, not A_num_rows
        }
    }

    for (size_t i = 0; i < B_size; ++i) {
        hB[i] = (float)rand() / RAND_MAX;
    }
    for (size_t i = 0; i < C_size; ++i) {
        hC[i] = 0.0; // Initialize output vector
    }

    CHECK_CUDA(cudaMemcpy(dA_csrOffsets, hA_csrOffsets, (A_num_rows + 1) * sizeof(int), cudaMemcpyHostToDevice));
    CHECK_CUDA(cudaMemcpy(dA_columns, hA_columns, A_nnz * sizeof(int), cudaMemcpyHostToDevice));
    CHECK_CUDA(cudaMemcpy(dA_values, hA_values, A_nnz * sizeof(float), cudaMemcpyHostToDevice));
    CHECK_CUDA(cudaMemcpy(dB, hB, B_size * sizeof(float), cudaMemcpyHostToDevice));
    CHECK_CUDA(cudaMemcpy(dC, hC, C_size * sizeof(float), cudaMemcpyHostToDevice));


    //--------------------------------------------------------------------------
    // CUSPARSE APIs
    cusparseHandle_t     handle = NULL;
    cusparseSpMatDescr_t matA;
    cusparseDnMatDescr_t matB, matC;
    void*                dBuffer    = NULL;
    size_t               bufferSize = 0;
    CHECK_CUSPARSE( cusparseCreate(&handle) )
    // Create sparse matrix A in CSR format
    CHECK_CUSPARSE( cusparseCreateCsr(&matA, A_num_rows, A_num_cols, A_nnz,
                                      dA_csrOffsets, dA_columns, dA_values,
                                      CUSPARSE_INDEX_32I, CUSPARSE_INDEX_32I,
                                      CUSPARSE_INDEX_BASE_ZERO, CUDA_R_32F) )
    // Create dense matrix B
    CHECK_CUSPARSE( cusparseCreateDnMat(&matB, A_num_cols, B_num_cols, ldb, dB,
                                        CUDA_R_32F, CUSPARSE_ORDER_COL) )
    // Create dense matrix C
    CHECK_CUSPARSE( cusparseCreateDnMat(&matC, A_num_rows, B_num_cols, ldc, dC,
                                        CUDA_R_32F, CUSPARSE_ORDER_COL) )
    // allocate an external buffer if needed
    CHECK_CUSPARSE( cusparseSpMM_bufferSize(
                                 handle,
                                 CUSPARSE_OPERATION_NON_TRANSPOSE,
                                 CUSPARSE_OPERATION_NON_TRANSPOSE,
                                 &alpha, matA, matB, &beta, matC, CUDA_R_32F,
                                 CUSPARSE_SPMM_ALG_DEFAULT, &bufferSize) )
    CHECK_CUDA( cudaMalloc(&dBuffer, bufferSize) )

    // execute preprocess (optional)
    CHECK_CUSPARSE( cusparseSpMM_preprocess(
                                 handle,
                                 CUSPARSE_OPERATION_NON_TRANSPOSE,
                                 CUSPARSE_OPERATION_NON_TRANSPOSE,
                                 &alpha, matA, matB, &beta, matC, CUDA_R_32F,
                                 CUSPARSE_SPMM_ALG_DEFAULT, dBuffer) )

    // execute SpMM
    CHECK_CUSPARSE( cusparseSpMM(handle,
                                 CUSPARSE_OPERATION_NON_TRANSPOSE,
                                 CUSPARSE_OPERATION_NON_TRANSPOSE,
                                 &alpha, matA, matB, &beta, matC, CUDA_R_32F,
                                 CUSPARSE_SPMM_ALG_DEFAULT, dBuffer) )

    // destroy matrix/vector descriptors
    CHECK_CUSPARSE( cusparseDestroySpMat(matA) )
    CHECK_CUSPARSE( cusparseDestroyDnMat(matB) )
    CHECK_CUSPARSE( cusparseDestroyDnMat(matC) )
    CHECK_CUSPARSE( cusparseDestroy(handle) )
    //--------------------------------------------------------------------------
    // device result check
    CHECK_CUDA( cudaMemcpy(hC, dC, C_size * sizeof(float),
                           cudaMemcpyDeviceToHost) )
    int correct = 1;

    
    cudaDeviceSynchronize();

    if (CPU) {
		for (size_t i = 0; i < C_size; i++)
			hC_result[i] = 0.0f;
		cpu_spmm(A_num_rows, A_nnz, B_num_cols, hA_values, hA_columns, hA_csrOffsets,
			hB, hC_result, B_num_rows);
		for (int i = 0; i < A_num_rows; i++) {
		for (int j = 0; j < B_num_cols; j++) {
			double abs_err = fabs(hC[i + j * ldc] - hC_result[i + j * ldc]);
			if (abs_err > ERROR_THRESHOLD) {
			correct = 0; // direct floating point comparison is not reliable
                        printf("%f . Device: %.4f CPU: %.4f\n",abs_err, hC[i + j * ldc], hC_result[i + j * ldc]);
			//break;
			}
		}
		}
    }
    if (correct)
        printf("spmm_csr_example test PASSED\n");
    else
        printf("spmm_csr_example test FAILED: wrong result\n");
    //--------------------------------------------------------------------------
    // device memory deallocation
    
    cudaFree(dA_csrOffsets);
	cudaFree(dA_columns);
	cudaFree(dA_values);
	cudaFree(dB);
	cudaFree(dC);

	free(hA_csrOffsets);
	free(hA_columns);
	free(hA_values);
	free(hB);
    free(hC);

    if (CPU) free(hC_result);

    return EXIT_SUCCESS;
}
