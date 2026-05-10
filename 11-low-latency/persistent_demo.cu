// Module 11 — persistent-kernel doorbell demo.
//
// Best on bare-metal Linux. Under some virtualization setups (WSL2 GPU
// passthrough) the host↔device coherence on host-mapped memory can be unreliable
// and the spin loop deadlocks; if you hit that, run this on a non-virtualized
// system. The pattern itself is correct.

#include <chrono>
#include <cstdio>

#include "cuda_utils.h"

using Clock = std::chrono::high_resolution_clock;

static double now_us() {
    return std::chrono::duration<double, std::micro>(
               Clock::now().time_since_epoch())
        .count();
}

__global__ void persistent_worker(volatile int* doorbell,
                                  volatile int* response,
                                  const float* in, float* out, int n) {
    int gid = threadIdx.x;
    __shared__ int cmd_s;

    while (true) {
        if (threadIdx.x == 0) {
            int c;
            // Spin on the doorbell. `volatile` forces a fresh load every
            // iteration; without it the compiler hoists the load out of
            // the loop and we deadlock.
            //
            // __nanosleep is a small back-off (sm_70+); it lets the SM
            // gate clocks briefly, reducing power/heat. Tune the
            // argument: 0 ns = pure spin (lowest latency), 1000 ns =
            // friendlier to the rest of the GPU. 200 is a reasonable
            // default for a hot path.
            do {
                c = *doorbell;
                if (c == 0) __nanosleep(200);
            } while (c == 0);
            cmd_s = c;
        }
        __syncthreads();
        int cmd = cmd_s;
        if (cmd == -1) return;

        if (gid < n) out[gid] = in[gid] * 2.0f;

        __syncthreads();
        if (threadIdx.x == 0) {
            // __threadfence() makes the writes to `out` (and any other
            // device-side stores above) globally visible *before* the
            // host can observe the response/doorbell stores below. On
            // host-mapped memory the system-scope fence is what the host
            // CPU's coherent fabric needs; on integrated/managed memory
            // you'd want __threadfence_system() — kept as __threadfence
            // here because `out` is plain device memory and the host
            // doesn't read it (only the response flag).
            __threadfence();
            *response = 1;
            *doorbell = 0;
        }
    }
}

int main() {
    constexpr int N = 1024;
    constexpr int ITERS = 5000;

    int *dbell_raw, *resp_raw;
    CUDA_CHECK(cudaHostAlloc(&dbell_raw, sizeof(int), cudaHostAllocMapped));
    CUDA_CHECK(cudaHostAlloc(&resp_raw,  sizeof(int), cudaHostAllocMapped));
    volatile int* dbell_host = dbell_raw;
    volatile int* resp_host  = resp_raw;
    *dbell_host = 0;
    *resp_host  = 0;

    int *dbell_dev, *resp_dev;
    CUDA_CHECK(cudaHostGetDevicePointer(&dbell_dev, dbell_raw, 0));
    CUDA_CHECK(cudaHostGetDevicePointer(&resp_dev,  resp_raw,  0));

    float *d_in, *d_out;
    CUDA_CHECK(cudaMalloc(&d_in,  N * sizeof(float)));
    CUDA_CHECK(cudaMalloc(&d_out, N * sizeof(float)));
    CUDA_CHECK(cudaMemset(d_in, 0, N * sizeof(float)));

    cudaStream_t stream;
    CUDA_CHECK(cudaStreamCreate(&stream));
    persistent_worker<<<1, N, 0, stream>>>(dbell_dev, resp_dev, d_in, d_out, N);

    // Ordering rationale for the host side `*resp_host = 0; *dbell_host = 1`
    // sequence below.
    //
    // We MUST clear the response *before* arming the doorbell — otherwise
    // the GPU could finish its last iteration and write `*resp_host = 1`
    // between our clear and the write of the new doorbell, and we'd skip
    // the next round-trip (or worse, see a stale "done" before the kernel
    // even sees the new request).
    //
    // We rely on x86's TSO (total-store-ordering): plain stores to
    // different cache lines remain in *program order* as observed by other
    // agents (here, the GPU reading host-mapped pinned memory over the
    // PCIe coherent fabric). So `resp = 0` is observed by the GPU before
    // `doorbell = 1`. No host-side fence needed.
    //
    // On a weakly-ordered CPU (AArch64, POWER) you would insert a
    // `std::atomic_thread_fence(std::memory_order_release)` between the
    // two stores, and use std::atomic with release semantics for the
    // doorbell store. Keep that in mind if you port this pattern off x86.
    for (int w = 0; w < 50; ++w) {
        *resp_host  = 0;
        *dbell_host = 1;
        while (*resp_host == 0) {}
    }

    double t0 = now_us();
    for (int i = 0; i < ITERS; ++i) {
        *resp_host  = 0;
        *dbell_host = 1;
        while (*resp_host == 0) {}
    }
    double per_iter = (now_us() - t0) / ITERS;
    std::printf("Persistent-kernel round-trip: %.2f us  (over %d iterations)\n",
                per_iter, ITERS);

    *dbell_host = -1;
    CUDA_CHECK(cudaStreamSynchronize(stream));
    CUDA_CHECK(cudaStreamDestroy(stream));
    CUDA_CHECK(cudaFreeHost(dbell_raw));
    CUDA_CHECK(cudaFreeHost(resp_raw));
    CUDA_CHECK(cudaFree(d_in));
    CUDA_CHECK(cudaFree(d_out));
    return 0;
}
