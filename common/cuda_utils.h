#pragma once

#include <cstdio>
#include <cstdlib>
#include <cuda_runtime.h>

#define CUDA_CHECK(call)                                                    \
    do {                                                                    \
        cudaError_t _err = (call);                                          \
        if (_err != cudaSuccess) {                                          \
            std::fprintf(stderr, "CUDA error %s:%d: %s\n",                  \
                         __FILE__, __LINE__, cudaGetErrorString(_err));     \
            std::exit(EXIT_FAILURE);                                        \
        }                                                                   \
    } while (0)

#define CUDA_CHECK_LAST() CUDA_CHECK(cudaGetLastError())

class GpuTimer {
public:
    GpuTimer() {
        cudaEventCreate(&start_);
        cudaEventCreate(&stop_);
    }
    ~GpuTimer() {
        cudaEventDestroy(start_);
        cudaEventDestroy(stop_);
    }
    GpuTimer(const GpuTimer&) = delete;
    GpuTimer& operator=(const GpuTimer&) = delete;

    void start(cudaStream_t s = 0) { cudaEventRecord(start_, s); }

    float stop_ms(cudaStream_t s = 0) {
        cudaEventRecord(stop_, s);
        cudaEventSynchronize(stop_);
        float ms = 0.0f;
        cudaEventElapsedTime(&ms, start_, stop_);
        return ms;
    }

private:
    cudaEvent_t start_, stop_;
};
