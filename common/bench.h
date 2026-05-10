#pragma once

#include <algorithm>
#include <utility>

#include "cuda_utils.h"

// Run `fn` once to warm up, then `iters` more times, returning the *minimum* GPU time
// observed (in milliseconds). Min is more robust than mean for kernel benchmarking:
// it represents what the hardware can do when the OS/driver isn't interfering.
template <typename Fn>
float bench_min_ms(int iters, Fn&& fn) {
    fn();
    CUDA_CHECK(cudaDeviceSynchronize());

    float best = 1e30f;
    for (int i = 0; i < iters; ++i) {
        GpuTimer t;
        t.start();
        fn();
        float ms = t.stop_ms();
        best = std::min(best, ms);
    }
    return best;
}
