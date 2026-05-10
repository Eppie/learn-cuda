// Module 1 — benchmarks.
//
// Measures:
//   1. Kernel launch overhead two ways:
//        a. *Queued* — fire N launches back-to-back, sync once. The per-launch number
//           is the steady-state queueing rate (driver-bound).
//        b. *Per-launch synced* — fire one launch, sync, repeat. Includes the full
//           host→GPU→host round-trip on every iteration. This is the latency that
//           matters for low-latency request/response code (Module 11).
//   2. vector_add achieved bandwidth as a function of block size.
//
// Run after solving starter.cu to see how block size affects throughput.

#include <cstdio>
#include <cstdlib>
#include <vector>

#include "bench.h"
#include "cuda_utils.h"

__global__ void empty_kernel() {}

__global__ void vector_add(const float* __restrict__ a,
                           const float* __restrict__ b,
                           float* __restrict__ c,
                           int n) {
    int gid = blockIdx.x * blockDim.x + threadIdx.x;
    if (gid < n) {
        c[gid] = a[gid] + b[gid];
    }
}

static void measure_launch_overhead() {
    constexpr int LAUNCHES = 10000;

    // Warm up.
    empty_kernel<<<1, 1>>>();
    CUDA_CHECK(cudaDeviceSynchronize());

    // (a) Queued: pipeline LAUNCHES launches, then a single sync at the end.
    //     Per-launch time = total / LAUNCHES = steady-state driver issue rate.
    {
        GpuTimer t;
        t.start();
        for (int i = 0; i < LAUNCHES; ++i) {
            empty_kernel<<<1, 1>>>();
        }
        float total_ms = t.stop_ms();   // GpuTimer's stop_ms calls cudaEventSynchronize
        float per_launch_us = (total_ms / LAUNCHES) * 1000.0f;
        std::printf("Empty-kernel launch (queued)        : %.3f us / launch  (over %d launches)\n",
                    per_launch_us, LAUNCHES);
    }

    // (b) Per-launch sync: each iteration issues one launch and waits for it. This is
    //     the actual host→GPU→host round-trip — what a synchronous request/response
    //     loop sees. Use a smaller iteration count: each iteration does its own sync,
    //     so the wall-clock cost is genuinely high.
    {
        constexpr int SYNC_LAUNCHES = 1000;
        GpuTimer t;
        t.start();
        for (int i = 0; i < SYNC_LAUNCHES; ++i) {
            empty_kernel<<<1, 1>>>();
            CUDA_CHECK(cudaDeviceSynchronize());
        }
        float total_ms = t.stop_ms();
        float per_launch_us = (total_ms / SYNC_LAUNCHES) * 1000.0f;
        std::printf("Empty-kernel launch (per-launch sync): %.3f us / launch  (over %d launches)\n",
                    per_launch_us, SYNC_LAUNCHES);
    }

    std::printf("(Gap between the two ≈ host↔GPU round-trip latency. CUDA Graphs in M11 closes it.)\n");
}

static void sweep_vector_add() {
    constexpr int N     = 1 << 24;          // 16 M elements
    constexpr int ITERS = 50;
    const     long bytes = static_cast<long>(N) * 3 * sizeof(float); // 2 reads + 1 write

    float *d_a, *d_b, *d_c;
    CUDA_CHECK(cudaMalloc(&d_a, N * sizeof(float)));
    CUDA_CHECK(cudaMalloc(&d_b, N * sizeof(float)));
    CUDA_CHECK(cudaMalloc(&d_c, N * sizeof(float)));

    // Touch the buffers so the allocator backs them with real pages.
    CUDA_CHECK(cudaMemset(d_a, 0, N * sizeof(float)));
    CUDA_CHECK(cudaMemset(d_b, 0, N * sizeof(float)));

    const int block_sizes[] = {32, 64, 128, 256, 512, 1024};

    std::printf("\nvector_add bandwidth sweep (N = %d, %.1f MB moved per launch):\n",
                N, bytes / 1.0e6);
    std::printf("  %-7s  %-10s  %-10s\n", "block", "time (ms)", "GB/s");
    std::printf("  %-7s  %-10s  %-10s\n", "-----", "---------", "----");

    for (int blk : block_sizes) {
        int grd = (N + blk - 1) / blk;
        float ms = bench_min_ms(ITERS, [=] {
            vector_add<<<grd, blk>>>(d_a, d_b, d_c, N);
        });
        float gbs = bytes / (ms * 1.0e6);
        std::printf("  %-7d  %-10.3f  %-10.1f\n", blk, ms, gbs);
    }

    std::printf("\nRTX 4090 peak DRAM bandwidth: ~1008 GB/s.\n");

    CUDA_CHECK(cudaFree(d_a));
    CUDA_CHECK(cudaFree(d_b));
    CUDA_CHECK(cudaFree(d_c));
}

int main() {
    int dev = 0;
    cudaDeviceProp prop{};
    CUDA_CHECK(cudaGetDeviceProperties(&prop, dev));
    std::printf("Device: %s  (CC %d.%d, %d SMs)\n",
                prop.name, prop.major, prop.minor, prop.multiProcessorCount);

    measure_launch_overhead();
    sweep_vector_add();
    return 0;
}
