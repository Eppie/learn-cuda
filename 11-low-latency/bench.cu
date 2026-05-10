// Module 11 — measurements: launch overhead, CUDA Graphs, pinned vs pageable
// memcpy, persistent-kernel doorbell round-trip.

#include <chrono>
#include <cstdio>
#include <cstdlib>
#include <vector>

#include "cuda_utils.h"

using Clock = std::chrono::high_resolution_clock;

static double now_us() {
    return std::chrono::duration<double, std::micro>(
               Clock::now().time_since_epoch())
        .count();
}

// ============================================================================
// Empty kernel: serves as the baseline for launch overhead.
// ============================================================================
__global__ void empty_kernel() {}

// ============================================================================
// Tiny kernel for graph capture: just bumps a counter.
// ============================================================================
__global__ void increment_kernel(int* counter, int n) {
    int gid = blockIdx.x * blockDim.x + threadIdx.x;
    if (gid < n) atomicAdd(counter, 1);
}

// ============================================================================
static void bench_launch_overhead() {
    constexpr int LAUNCHES = 10000;

    empty_kernel<<<1, 1>>>();
    CUDA_CHECK(cudaDeviceSynchronize());

    double t0 = now_us();
    for (int i = 0; i < LAUNCHES; ++i) empty_kernel<<<1, 1>>>();
    CUDA_CHECK(cudaDeviceSynchronize());
    double t1 = now_us();

    std::printf("  Empty-kernel launch:        %.2f us / launch  (over %d launches)\n",
                (t1 - t0) / LAUNCHES, LAUNCHES);
}

static void bench_cuda_graph() {
    constexpr int LAUNCHES = 10000;
    constexpr int N = 1024;

    int* d_counter;
    CUDA_CHECK(cudaMalloc(&d_counter, sizeof(int)));
    CUDA_CHECK(cudaMemset(d_counter, 0, sizeof(int)));

    cudaStream_t stream;
    CUDA_CHECK(cudaStreamCreate(&stream));

    // Sequential baseline: 3 increments per "iteration", LAUNCHES iterations.
    increment_kernel<<<1, N, 0, stream>>>(d_counter, N);
    CUDA_CHECK(cudaStreamSynchronize(stream));

    double t0 = now_us();
    for (int i = 0; i < LAUNCHES; ++i) {
        increment_kernel<<<1, N, 0, stream>>>(d_counter, N);
        increment_kernel<<<1, N, 0, stream>>>(d_counter, N);
        increment_kernel<<<1, N, 0, stream>>>(d_counter, N);
    }
    CUDA_CHECK(cudaStreamSynchronize(stream));
    double seq_us = now_us() - t0;

    // Now build a graph.
    cudaGraph_t graph;
    cudaGraphExec_t gx;
    CUDA_CHECK(cudaStreamBeginCapture(stream, cudaStreamCaptureModeGlobal));
    increment_kernel<<<1, N, 0, stream>>>(d_counter, N);
    increment_kernel<<<1, N, 0, stream>>>(d_counter, N);
    increment_kernel<<<1, N, 0, stream>>>(d_counter, N);
    CUDA_CHECK(cudaStreamEndCapture(stream, &graph));
    CUDA_CHECK(cudaGraphInstantiate(&gx, graph, nullptr, nullptr, 0));

    // Warm up.
    CUDA_CHECK(cudaGraphLaunch(gx, stream));
    CUDA_CHECK(cudaStreamSynchronize(stream));

    t0 = now_us();
    for (int i = 0; i < LAUNCHES; ++i) {
        CUDA_CHECK(cudaGraphLaunch(gx, stream));
    }
    CUDA_CHECK(cudaStreamSynchronize(stream));
    double graph_us = now_us() - t0;

    std::printf("  3-kernel sequential:        %.2f us / iter  (LAUNCHES=%d)\n",
                seq_us / LAUNCHES, LAUNCHES);
    std::printf("  3-kernel CUDA Graph replay: %.2f us / iter  (%.1fx faster)\n",
                graph_us / LAUNCHES, seq_us / graph_us);

    CUDA_CHECK(cudaGraphExecDestroy(gx));
    CUDA_CHECK(cudaGraphDestroy(graph));
    CUDA_CHECK(cudaStreamDestroy(stream));
    CUDA_CHECK(cudaFree(d_counter));
}

static void bench_pinned_vs_pageable() {
    constexpr int N = 1 << 24;          // 64 MB
    constexpr int ITERS = 20;

    float* d_buf;
    CUDA_CHECK(cudaMalloc(&d_buf, N * sizeof(float)));

    std::vector<float> h_pageable(N);
    float* h_pinned;
    CUDA_CHECK(cudaMallocHost(&h_pinned, N * sizeof(float)));

    auto bench = [&](float* h_ptr) {
        // Warm up.
        CUDA_CHECK(cudaMemcpy(d_buf, h_ptr, N * sizeof(float), cudaMemcpyHostToDevice));
        CUDA_CHECK(cudaDeviceSynchronize());
        double t0 = now_us();
        for (int i = 0; i < ITERS; ++i) {
            CUDA_CHECK(cudaMemcpy(d_buf, h_ptr, N * sizeof(float), cudaMemcpyHostToDevice));
        }
        CUDA_CHECK(cudaDeviceSynchronize());
        return (now_us() - t0) / ITERS;
    };

    double pageable_us = bench(h_pageable.data());
    double pinned_us   = bench(h_pinned);
    double bytes = N * sizeof(float);

    std::printf("  H2D pageable (%dMB):        %.0f us  (%.1f GB/s)\n",
                (int)(bytes / 1.0e6), pageable_us, bytes / pageable_us / 1e3);
    std::printf("  H2D pinned   (%dMB):        %.0f us  (%.1f GB/s)\n",
                (int)(bytes / 1.0e6), pinned_us,   bytes / pinned_us   / 1e3);
    std::printf("  Speedup:                    %.1fx\n", pageable_us / pinned_us);

    CUDA_CHECK(cudaFreeHost(h_pinned));
    CUDA_CHECK(cudaFree(d_buf));
}


int main() {
    cudaDeviceProp prop{};
    CUDA_CHECK(cudaGetDeviceProperties(&prop, 0));
    std::printf("Device: %s\n\n", prop.name);

    std::printf("Launch overhead\n");
    bench_launch_overhead();

    std::printf("\nCUDA Graphs\n");
    bench_cuda_graph();

    std::printf("\nPinned vs pageable host memory\n");
    bench_pinned_vs_pageable();

    std::printf("\nPersistent kernel + host doorbell\n");
    std::printf("  See persistent_demo.cu and `make persistent_demo`. Skipped\n"
                "  here because the live-test pattern requires host↔device\n"
                "  memory coherence that's flaky under some virtualization\n"
                "  setups (WSL2 GPU passthrough). The demo binary measures\n"
                "  it on bare-metal Linux.\n");
    return 0;
}
