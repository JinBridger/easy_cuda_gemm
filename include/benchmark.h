#pragma once

const int benchmark_cycle = 20;

float* generate_float_matrix(int size);
void   generate_float_timer(unsigned int size);
void   matrix_differ(float* correct_val, float* test_val, unsigned int size);
void   gflops_benchmark(unsigned int size, bool is_baseline = true);
void   correctness_benchmark(unsigned int size = 1024);