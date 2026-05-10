// Module 11 — CUDA events as a stream-sync primitive.
//
// Events are the bridge between streams and graphs: a lightweight
// "barrier object" you can record on one stream and wait on from another.
// They cost ~hundreds of nanoseconds and are the canonical way to express
// fine-grained inter-stream dependencies that don't fit a static DAG (use
// graphs for static DAGs).
//
// This demo shows three things:
//   1. Timing a kernel with cudaEventRecord + cudaEventElapsedTime
//      (the same primitive GpuTimer in cuda_utils.h wraps).
//   2. Cross-stream wait: stream B waits for an event recorded on stream A.
//      (cudaStreamWaitEvent is *non-blocking* on the host — it inserts a
//      wait node into stream B's queue.)
//   3. cudaMemPrefetchAsync on managed (unified) memory: a hint to migrate
//      pages to a target device before they're touched. Critical for
//      keeping the first-touch fault overhead off the critical path.

#include <cstdio>

#include "cuda_utils.h"

__global__ void produce(float* out, int n) {
    int gid = blockIdx.x * blockDim.x + threadIdx.x;
    if (gid < n) out[gid] = (float)gid * 0.5f;
}

__global__ void consume(const float* in, float* out, int n) {
    int gid = blockIdx.x * blockDim.x + threadIdx.x;
    if (gid < n) out[gid] = in[gid] + 1.0f;
}

__global__ void scale(float* x, float s, int n) {
    int gid = blockIdx.x * blockDim.x + threadIdx.x;
    if (gid < n) x[gid] *= s;
}

int main() {
    constexpr int N = 1 << 20;
    constexpr int BLOCK = 256;
    int grid = (N + BLOCK - 1) / BLOCK;

    // ------------------------------------------------------------------------
    // 1. Event-based timing (alternative to GpuTimer).
    // ------------------------------------------------------------------------
    float* d_a;
    CUDA_CHECK(cudaMalloc(&d_a, N * sizeof(float)));

    cudaEvent_t t0, t1;
    CUDA_CHECK(cudaEventCreate(&t0));
    CUDA_CHECK(cudaEventCreate(&t1));

    // Warm up (first launch pays JIT/context costs).
    produce<<<grid, BLOCK>>>(d_a, N);
    CUDA_CHECK(cudaDeviceSynchronize());

    CUDA_CHECK(cudaEventRecord(t0));            // default stream
    produce<<<grid, BLOCK>>>(d_a, N);
    CUDA_CHECK(cudaEventRecord(t1));
    CUDA_CHECK(cudaEventSynchronize(t1));       // host waits

    float elapsed_ms = 0.0f;
    CUDA_CHECK(cudaEventElapsedTime(&elapsed_ms, t0, t1));
    std::printf("produce: %.3f ms (event-timed, post-warmup)\n", elapsed_ms);

    // ------------------------------------------------------------------------
    // 2. Cross-stream sync: B waits on an event recorded in A.
    //    Pattern: produce on stream A, then independently consume on B; we
    //    want B's kernel to start as soon as A's produce is done — no host
    //    sync. cudaStreamWaitEvent is the right tool.
    // ------------------------------------------------------------------------
    cudaStream_t sA, sB;
    CUDA_CHECK(cudaStreamCreate(&sA));
    CUDA_CHECK(cudaStreamCreate(&sB));

    float* d_b;
    CUDA_CHECK(cudaMalloc(&d_b, N * sizeof(float)));

    cudaEvent_t produced;
    // cudaEventDisableTiming makes it cheaper (~30% faster record) when you
    // only need ordering, not elapsed-time measurement.
    CUDA_CHECK(cudaEventCreateWithFlags(&produced, cudaEventDisableTiming));

    produce<<<grid, BLOCK, 0, sA>>>(d_a, N);
    CUDA_CHECK(cudaEventRecord(produced, sA));      // mark "produce done"
    CUDA_CHECK(cudaStreamWaitEvent(sB, produced, 0)); // sB will wait for it

    // Now sB can launch the consumer; the runtime will hold it until the
    // event fires. The host call returns immediately.
    consume<<<grid, BLOCK, 0, sB>>>(d_a, d_b, N);

    // Meanwhile sA can race ahead with unrelated work — independent of sB.
    scale<<<grid, BLOCK, 0, sA>>>(d_a, 2.0f, N);

    CUDA_CHECK(cudaStreamSynchronize(sB));

    // Verify: d_b[i] should be (i*0.5) + 1 (consume saw the value produce
    // wrote, before scale doubled it on stream A).
    float h0 = 0.0f, hlast = 0.0f;
    CUDA_CHECK(cudaMemcpy(&h0, d_b, sizeof(float), cudaMemcpyDeviceToHost));
    CUDA_CHECK(cudaMemcpy(&hlast, d_b + (N - 1), sizeof(float), cudaMemcpyDeviceToHost));
    bool ok = (h0 == 1.0f) && (hlast == (float)(N - 1) * 0.5f + 1.0f);
    std::printf("cross-stream wait: d_b[0]=%.1f d_b[N-1]=%.1f  %s\n",
                h0, hlast, ok ? "PASS" : "FAIL");

    CUDA_CHECK(cudaStreamSynchronize(sA));

    // ------------------------------------------------------------------------
    // 3. cudaMemPrefetchAsync on unified (managed) memory.
    //    Without prefetch, first-touch on the GPU triggers a page-fault
    //    migration on every accessed page (slow). Prefetching ahead of the
    //    kernel hides that cost.
    // ------------------------------------------------------------------------
    int dev = 0;
    CUDA_CHECK(cudaGetDevice(&dev));

    int attr_supported = 0;
    CUDA_CHECK(cudaDeviceGetAttribute(&attr_supported,
                                      cudaDevAttrConcurrentManagedAccess, dev));

    if (!attr_supported) {
        std::printf("managed prefetch: device lacks concurrent managed access; skipping\n");
    } else {
        float* m;
        CUDA_CHECK(cudaMallocManaged(&m, N * sizeof(float)));
        for (int i = 0; i < N; ++i) m[i] = 1.0f;            // host-touched pages

        // Hint: migrate to GPU before the kernel runs.
        CUDA_CHECK(cudaMemPrefetchAsync(m, N * sizeof(float), dev, sA));
        scale<<<grid, BLOCK, 0, sA>>>(m, 3.0f, N);
        // Hint: migrate back to host before we read it.
        CUDA_CHECK(cudaMemPrefetchAsync(m, N * sizeof(float), cudaCpuDeviceId, sA));
        CUDA_CHECK(cudaStreamSynchronize(sA));

        bool mok = (m[0] == 3.0f) && (m[N - 1] == 3.0f);
        std::printf("managed prefetch: m[0]=%.1f m[N-1]=%.1f  %s\n",
                    m[0], m[N - 1], mok ? "PASS" : "FAIL");
        CUDA_CHECK(cudaFree(m));
    }

    CUDA_CHECK(cudaEventDestroy(produced));
    CUDA_CHECK(cudaEventDestroy(t0));
    CUDA_CHECK(cudaEventDestroy(t1));
    CUDA_CHECK(cudaStreamDestroy(sA));
    CUDA_CHECK(cudaStreamDestroy(sB));
    CUDA_CHECK(cudaFree(d_a));
    CUDA_CHECK(cudaFree(d_b));
    return 0;
}
