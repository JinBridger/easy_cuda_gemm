#include "easy_cuda_gemm.h"

#include <cuda_runtime_api.h>
#include <vector_types.h>

template <int BLOCK>
__global__ void ecgemm(int m, int n, int k, float* a, int lda, float* b, int ldb, float* c, int ldc) {
    int bx = blockIdx.x;
    int by = blockIdx.y;
    int tx = threadIdx.x;
    int ty = threadIdx.y;

    float* begin_a = a + by * BLOCK * k;
    float* begin_b = b + bx * BLOCK;
    float* end_a   = begin_a + k;

    float sum = 0.0f;

    for (float *submat_a = begin_a, *submat_b = begin_b; submat_a < end_a; submat_a += BLOCK, submat_b += n * BLOCK) {
        __shared__ float mat_a[BLOCK][BLOCK];
        __shared__ float mat_b[BLOCK][BLOCK];

        // Move to shared memory
        mat_a[ty][tx] = submat_a[ty * k + tx];
        mat_b[ty][tx] = submat_b[ty * n + tx];
        __syncthreads();

#pragma unroll
        for (int kk = 0; kk < BLOCK; ++kk) {
            sum += mat_a[ty][kk] * mat_b[kk][tx];
        }
        __syncthreads();
    }

    c[(by * BLOCK + ty) * n + bx * BLOCK + tx] = sum;
}

void easy_cuda_gemm(cublasHandle_t handle, int m, int n, int k, float* d_A, int lda, float* d_B, int ldb, float* d_C,
                    int ldc) {
    constexpr int BLOCK = 16;

    dim3 block(BLOCK, BLOCK);
    dim3 grid((m + BLOCK - 1) / BLOCK, (n + BLOCK - 1) / BLOCK);

    ecgemm<BLOCK><<<grid, block>>>(m, n, k, d_A, lda, d_B, ldb, d_C, ldc);
}