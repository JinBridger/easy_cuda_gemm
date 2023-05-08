#include "easy_cuda_gemm.h"

#include <cuda_runtime_api.h>
#include <vector_types.h>

template <int BLOCK>
__global__ void ecgemm(int m, int n, int k, const float* a, int lda, const float* b, int ldb, float* c, int ldc) {
    int tx = blockIdx.x * BLOCK + threadIdx.x;
    int ty = blockIdx.y * BLOCK + threadIdx.y;

    if (ty < m && tx < n) {
        float sum = 0;
        for (int i = 0; i < k; ++i) {
            sum += a[ty * k + i] * b[i * n + tx];
        }
        c[ty * n + tx] = sum;
    }
}

void easy_cuda_gemm(cublasHandle_t handle, int m, int n, int k, float* d_A, int lda, float* d_B, int ldb, float* d_C,
                    int ldc) {
    constexpr int BLOCK = 16;

    dim3 block(BLOCK, BLOCK);
    dim3 grid((m + BLOCK - 1) / BLOCK, (n + BLOCK - 1) / BLOCK);

    ecgemm<BLOCK><<<grid, block>>>(m, n, k, d_A, lda, d_B, ldb, d_C, ldc);
}