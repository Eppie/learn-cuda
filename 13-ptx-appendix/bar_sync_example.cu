// Module 13 — `bar.sync` variants.
//
// The familiar `__syncthreads()` is just `bar.sync 0;` — barrier ID 0,
// all threads in the block participate. PTX exposes more variants:
//
//   bar.sync     0           - block-wide barrier, all threads (the default)
//   bar.sync     N           - barrier ID N (0..15), all threads
//   bar.sync     N, COUNT    - barrier ID N, only COUNT threads required
//   bar.red.{and,or,popc} N, COUNT, p   - barrier-and-reduce
//
// Why care:
//   - "Named barriers" (different IDs) let producer/consumer sets sync
//     independently. `__syncthreads()` is a club barrier — every thread
//     in the block must hit it. With named barriers you can have a
//     subgroup wait without blocking the whole block.
//   - `bar.red` performs a reduction (and/or/popc) across the count of
//     participating threads while syncing — handy for "did anyone fail?"
//     fast paths.
//
// We demonstrate two patterns: independent named barriers (groups A/B
// sync separately) and `bar.red.or` for a block-wide OR-reduce.

#include <cstdio>
#include <cuda_runtime.h>

#include "cuda_utils.h"

// ----------------------------------------------------------------------------
// Two named-barrier groups: half the block hits barrier 1, the other half
// barrier 2. They sync independently of one another (and of barrier 0).
// ----------------------------------------------------------------------------
__global__ void named_barriers_kernel(int* out) {
    __shared__ int produced_a;
    __shared__ int produced_b;

    int tid = threadIdx.x;
    int half = blockDim.x / 2;

    if (tid == 0) { produced_a = 0; produced_b = 0; }

    // Whole-block barrier 0: full sync, ensure produced_a/b initialized.
    asm volatile("bar.sync 0;");

    if (tid < half) {
        // Group A: produces 'produced_a'.
        if (tid == 0) produced_a = 1234;
        // Group A waits at named barrier 1, count = half threads.
        asm volatile("bar.sync 1, %0;" :: "r"(half));
        // After sync, every thread in group A can read produced_a.
        out[tid] = produced_a;
    } else {
        // Group B: produces 'produced_b'.
        if (tid == half) produced_b = 5678;
        // Group B waits at named barrier 2, count = half threads.
        asm volatile("bar.sync 2, %0;" :: "r"(half));
        out[tid] = produced_b;
    }
    // Note: no whole-block sync needed at end — each group already
    // synchronized on its own ID and writes to disjoint output slots.
}

// ----------------------------------------------------------------------------
// bar.red.or: barrier + OR-reduce. Each thread contributes one predicate;
// after the barrier, every thread sees the OR of all predicates. Useful for
// "did any lane detect a problem?" early-exit logic.
// ----------------------------------------------------------------------------
__global__ void bar_red_or_kernel(int trigger_tid, int* anyone_flagged) {
    int tid = threadIdx.x;
    // Set p = 1 only on the trigger thread.
    int p_in = (tid == trigger_tid) ? 1 : 0;
    int p_out;

    asm volatile(
        "{\n"
        "    .reg .pred  %%pin, %%pout;\n"
        "    setp.ne.s32 %%pin, %1, 0;\n"
        "    bar.red.or.pred %%pout, 4, %%pin;\n"
        "    selp.s32    %0, 1, 0, %%pout;\n"
        "}\n"
        : "=r"(p_out) : "r"(p_in));

    if (tid == 0) *anyone_flagged = p_out;
}

int main() {
    cudaDeviceProp prop{};
    CUDA_CHECK(cudaGetDeviceProperties(&prop, 0));
    std::printf("Device: %s\n", prop.name);
    std::printf("bar.sync variants demo\n\n");

    constexpr int BLK = 64;       // half = 32 = one warp per group
    int *d_out;
    CUDA_CHECK(cudaMalloc(&d_out, BLK * sizeof(int)));

    // --- named barriers ---
    named_barriers_kernel<<<1, BLK>>>(d_out);
    CUDA_CHECK_LAST();
    int h_out[BLK];
    CUDA_CHECK(cudaMemcpy(h_out, d_out, BLK * sizeof(int), cudaMemcpyDeviceToHost));

    int errs = 0;
    for (int i = 0; i < BLK / 2; ++i) if (h_out[i] != 1234) ++errs;
    for (int i = BLK / 2; i < BLK; ++i) if (h_out[i] != 5678) ++errs;
    std::printf("named_barriers (bar.sync 1 / bar.sync 2):  errors=%d %s\n",
                errs, errs == 0 ? "(PASS)" : "(FAIL)");

    // --- bar.red.or with trigger=on ---
    int *d_flag; CUDA_CHECK(cudaMalloc(&d_flag, sizeof(int)));
    bar_red_or_kernel<<<1, BLK>>>(/*trigger_tid=*/17, d_flag);
    CUDA_CHECK_LAST();
    int h_flag = 0;
    CUDA_CHECK(cudaMemcpy(&h_flag, d_flag, sizeof(int), cudaMemcpyDeviceToHost));
    int e1 = (h_flag == 1) ? 0 : 1;
    std::printf("bar.red.or (one lane true):  result=%d %s\n",
                h_flag, e1 == 0 ? "(PASS)" : "(FAIL)");

    // --- bar.red.or with trigger=off (no lane true) ---
    bar_red_or_kernel<<<1, BLK>>>(/*trigger_tid=*/-1, d_flag);
    CUDA_CHECK_LAST();
    CUDA_CHECK(cudaMemcpy(&h_flag, d_flag, sizeof(int), cudaMemcpyDeviceToHost));
    int e2 = (h_flag == 0) ? 0 : 1;
    std::printf("bar.red.or (no lanes true):  result=%d %s\n",
                h_flag, e2 == 0 ? "(PASS)" : "(FAIL)");

    CUDA_CHECK(cudaFree(d_out));
    CUDA_CHECK(cudaFree(d_flag));
    int total_errs = errs + e1 + e2;
    return total_errs == 0 ? 0 : 1;
}
