#include "benchmark.h"
#include "cublas_v2.h"
#include "cuda_runtime.h"
#include "device_launch_parameters.h"
#include "easy_cuda_gemm.h"

#include <chrono>
#include <cuda_runtime_api.h>
#include <driver_types.h>
#include <iostream>
#include <omp.h>

float* generate_float_matrix(int size) {
    float* mat = new float[size * size];

#pragma omp parallel for
    for (int i = 0; i < size; ++i) {
        for (int j = 0; j < size; ++j) {
            mat[i * size + j] = float(rand()) / RAND_MAX;
        }
    }

    return mat;
}

void generate_float_timer(unsigned int size) {
    auto start = std::chrono::system_clock::now();

    float* mat = generate_float_matrix(size);

    auto end      = std::chrono::system_clock::now();
    auto duration = std::chrono::duration_cast<std::chrono::milliseconds>(end - start);
    std::cout << "Generate " << size << " matrix cost " << duration.count() << " ms" << std::endl;

    delete[] mat;
}

void matrix_differ(float* correct_val, float* test_val, unsigned int size) {
    float max_diff = 0.0f;

    for (unsigned int i = 0; i < size * size; ++i) {
        float diff = fabs(correct_val[i] - test_val[i]);
        max_diff   = std::max(max_diff, diff);
    }

    std::cout << "Max diff: " << max_diff << std::endl;
}

void gflops_benchmark(unsigned int size, bool is_baseline) {
    size_t size_of_matrix = sizeof(float) * size * size;

    float* host_mat_A = generate_float_matrix(size);
    float* host_mat_B = generate_float_matrix(size);

    float* device_mat_A;
    float* device_mat_B;
    float* device_mat_C;
    cudaMalloc(( void** )&device_mat_A, size_of_matrix);
    cudaMalloc(( void** )&device_mat_B, size_of_matrix);
    cudaMalloc(( void** )&device_mat_C, size_of_matrix);

    cudaMemcpy(( void* )device_mat_A, ( void* )host_mat_A, size_of_matrix, cudaMemcpyHostToDevice);
    cudaMemcpy(( void* )device_mat_B, ( void* )host_mat_B, size_of_matrix, cudaMemcpyHostToDevice);

    cublasHandle_t handle;
    cublasCreate(&handle);

    float alpha = 1;
    float beta  = 0;

    cudaEvent_t start, end;
    cudaEventCreate(&start);
    cudaEventCreate(&end);

    cudaEventRecord(start);

    if (is_baseline) {
        for (int i = 0; i < benchmark_cycle; ++i) {
            cublasSgemm(handle, CUBLAS_OP_T, CUBLAS_OP_T, size, size, size, &alpha, device_mat_A, size, device_mat_B,
                        size, &beta, device_mat_C, size);
        }
    }
    else {
        for (int i = 0; i < benchmark_cycle; ++i) {
            easy_cuda_gemm(handle, size, size, size, device_mat_A, size, device_mat_B, size, device_mat_C, size);
        }
    }

    cudaEventRecord(end);
    cudaEventSynchronize(end);

    float elapsed_time;
    cudaEventElapsedTime(&elapsed_time, start, end);
    float avg_time = elapsed_time / benchmark_cycle;

    double flops_per_matmul = 2.0 * size * size * size;
    double avg_gflops       = (flops_per_matmul * 1.0e-9f) / (avg_time / 1000.0f);

    std::cout << "Size: " << size << "\t";
    std::cout << "GFLOPS: " << avg_gflops << "\t";
    std::cout << "Time: " << avg_time << std::endl;

    cudaFree(( void* )device_mat_A);
    cudaFree(( void* )device_mat_B);
    cudaFree(( void* )device_mat_C);

    delete[] host_mat_A;
    delete[] host_mat_B;
}

void correctness_benchmark(unsigned int size) {
    size_t size_of_matrix = sizeof(float) * size * size;

    float* host_mat_A = generate_float_matrix(size);
    float* host_mat_B = generate_float_matrix(size);

    float* host_mat_correct = new float[size * size];
    float* host_mat_test    = new float[size * size];

    float* device_mat_A;
    float* device_mat_B;
    float* device_mat_C;
    cudaMalloc(( void** )&device_mat_A, size_of_matrix);
    cudaMalloc(( void** )&device_mat_B, size_of_matrix);
    cudaMalloc(( void** )&device_mat_C, size_of_matrix);

    cudaMemcpy(( void* )device_mat_A, ( void* )host_mat_A, size_of_matrix, cudaMemcpyHostToDevice);
    cudaMemcpy(( void* )device_mat_B, ( void* )host_mat_B, size_of_matrix, cudaMemcpyHostToDevice);

    cublasHandle_t handle;
    cublasCreate(&handle);

    float alpha = 1;
    float beta  = 0;

    cublasSgemm(handle, CUBLAS_OP_N, CUBLAS_OP_N, size, size, size, &alpha, device_mat_B, size, device_mat_A, size,
                &beta, device_mat_C, size);

    cudaMemcpy(( void* )host_mat_correct, ( void* )device_mat_C, size_of_matrix, cudaMemcpyDeviceToHost);

    easy_cuda_gemm(handle, size, size, size, device_mat_A, size, device_mat_B, size, device_mat_C, size);

    cudaMemcpy(( void* )host_mat_test, ( void* )device_mat_C, size_of_matrix, cudaMemcpyDeviceToHost);

    cudaFree(( void* )device_mat_A);
    cudaFree(( void* )device_mat_B);
    cudaFree(( void* )device_mat_C);

    matrix_differ(host_mat_correct, host_mat_test, size);

    delete[] host_mat_A;
    delete[] host_mat_B;
    delete[] host_mat_correct;
    delete[] host_mat_test;
}