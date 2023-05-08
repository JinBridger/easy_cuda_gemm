#pragma once

#include "cublas_v2.h"
#include "cuda_runtime.h"
#include "device_launch_parameters.h"

void easy_cuda_gemm(cublasHandle_t handle, int m, int n, int k, float* d_A, int lda, float* d_B, int ldb, float* d_C,
                    int ldc);