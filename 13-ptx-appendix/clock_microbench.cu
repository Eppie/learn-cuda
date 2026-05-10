// Module 13 — `clock64()` cycle-counter microbenches.
//
// `clock64()` reads the SM's 64-bit cycle counter (PTX `mov.u64 %0, %clock64`).
// Wrapping a tight loop in start/stop reads gives you per-instruction cycle
// numbers at the granularity that SASS does — useful for understanding why
// your hot loop runs at the rate it does.
//
// Caveats:
//   1. The clock counter advances per SM, not globally. Don't compare across
//      blocks unless you know they ran on the same SM (which you don't).
//   2. The compiler is *very* aggressive about hoisting/eliminating timed
//      code. We use `asm volatile` and write to global memory to keep work in.
//   3. Issue-rate vs latency: a tight FMA loop with no dependencies hits
//      issue rate; a chained-FMA loop hits FMA latency. We measure both.
//
// Reference numbers on Ada (sm_89), per CUDA-binary-utilities / Volta+ SASS docs:
//   FMA throughput: 4 cycles / warp / FP32 instruction (16 lanes/cycle on
//                   2 partitions; full warp issues every 4 cycles per pipe)
//   FMA latency:    ~4 cycles back-to-back (with reuse cache)
//
// These numbers are guidelines — actual measured cycles depend on warp
// scheduling, partition stalls, and surrounding code. The point of this bench
// is to get you a number you can compare against a code change.

#include <cstdint>
#include <cstdio>
#include <cuda_runtime.h>

#include "cuda_utils.h"

constexpr int ITERS = 1024;

// ----------------------------------------------------------------------------
// Throughput: independent FMAs (no dependency between iterations).
// ----------------------------------------------------------------------------
__global__ void fma_throughput(float* out, unsigned long long* cycles) {
    float a = (float)threadIdx.x * 0.5f;
    float b = (float)threadIdx.x * 0.25f + 1.0f;
    float c0 = 1.0f, c1 = 1.0f, c2 = 1.0f, c3 = 1.0f;

    unsigned long long t0, t1;
    asm volatile("mov.u64 %0, %%clock64;" : "=l"(t0));

    #pragma unroll 1
    for (int i = 0; i < ITERS; ++i) {
        // Four independent chains keep the FMA pipe busy and avoid stalling
        // on a single-chain dependency.
        c0 = c0 * a + b;
        c1 = c1 * a + b;
        c2 = c2 * a + b;
        c3 = c3 * a + b;
    }

    asm volatile("mov.u64 %0, %%clock64;" : "=l"(t1));

    if (threadIdx.x == 0) cycles[blockIdx.x] = t1 - t0;
    // sink so the compiler keeps c0..c3 alive
    out[blockIdx.x * blockDim.x + threadIdx.x] = c0 + c1 + c2 + c3;
}

// ----------------------------------------------------------------------------
// Latency: chained FMAs (each depends on prior).
// ----------------------------------------------------------------------------
__global__ void fma_latency(float* out, unsigned long long* cycles) {
    float a = (float)threadIdx.x * 0.5f;
    float b = (float)threadIdx.x * 0.25f + 1.0f;
    float c = 1.0f;

    unsigned long long t0, t1;
    asm volatile("mov.u64 %0, %%clock64;" : "=l"(t0));

    #pragma unroll 1
    for (int i = 0; i < ITERS; ++i) {
        c = c * a + b;     // single chain — next iter waits for this
    }

    asm volatile("mov.u64 %0, %%clock64;" : "=l"(t1));

    if (threadIdx.x == 0) cycles[blockIdx.x] = t1 - t0;
    out[blockIdx.x * blockDim.x + threadIdx.x] = c;
}

int main() {
    cudaDeviceProp prop{};
    CUDA_CHECK(cudaGetDeviceProperties(&prop, 0));
    std::printf("Device: %s  (sm_%d%d)\n", prop.name, prop.major, prop.minor);
    std::printf("Cycle-counter microbench: clock64()-bracketed FMA loops\n\n");

    constexpr int BLK = 32;
    constexpr int GRID = 1;

    float *d_out;  unsigned long long *d_cyc;
    CUDA_CHECK(cudaMalloc(&d_out, GRID * BLK * sizeof(float)));
    CUDA_CHECK(cudaMalloc(&d_cyc, GRID * sizeof(unsigned long long)));

    // throughput
    fma_throughput<<<GRID, BLK>>>(d_out, d_cyc);
    CUDA_CHECK_LAST();
    unsigned long long h_cyc_thr = 0;
    CUDA_CHECK(cudaMemcpy(&h_cyc_thr, d_cyc, sizeof(h_cyc_thr), cudaMemcpyDeviceToHost));

    // latency
    fma_latency<<<GRID, BLK>>>(d_out, d_cyc);
    CUDA_CHECK_LAST();
    unsigned long long h_cyc_lat = 0;
    CUDA_CHECK(cudaMemcpy(&h_cyc_lat, d_cyc, sizeof(h_cyc_lat), cudaMemcpyDeviceToHost));

    int total_thr_fmas = ITERS * 4;     // 4 chains
    int total_lat_fmas = ITERS;

    std::printf("Throughput (4 independent chains):\n");
    std::printf("  cycles=%llu, FMAs=%d, cycles/FMA = %.2f\n",
                h_cyc_thr, total_thr_fmas, (double)h_cyc_thr / total_thr_fmas);
    std::printf("Latency (single chain):\n");
    std::printf("  cycles=%llu, FMAs=%d, cycles/FMA = %.2f\n",
                h_cyc_lat, total_lat_fmas, (double)h_cyc_lat / total_lat_fmas);
    std::printf("\nInterpretation: throughput / cycle-per-FMA tells you how\n"
                "  close the loop gets to the FMA pipe rate; latency tells\n"
                "  you the back-to-back issue interval.\n");

    CUDA_CHECK(cudaFree(d_out));
    CUDA_CHECK(cudaFree(d_cyc));
    return 0;
}
