#include "benchmark.h"

#include <iostream>
#include <omp.h>

int main() {
    for (int s = 1024; s <= 16384; s += 1024) {
        gflops_benchmark(s);
        // generate_float_timer(s);
    }

    // correctness_benchmark();

    return 0;
}