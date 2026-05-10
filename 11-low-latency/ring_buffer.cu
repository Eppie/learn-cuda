// Module 11 — SPSC kernel-queue / ring-buffer pattern.
//
// One persistent kernel reads work items from a host-pinned ring buffer.
// Multiple host threads enqueue work concurrently. This is the shape of a
// real low-latency dispatcher: instead of one launch per task (5+ us each)
// or one kernel per request (kernel launch + parameter parsing), the GPU
// stays "hot" and pulls work as the host posts it.
//
// Layout of the ring (all in host-mapped pinned memory, polled by the
// device):
//
//     head_dev: producer cursor, written by host, read by device.
//     tail_dev: consumer cursor, written by device, read by host (for
//               flow-control, so producers can stall when full).
//     slots[CAP]: each slot has {seq, x, y, out}. `seq` is set last by the
//                 producer so the consumer sees a fully-populated slot
//                 (release/acquire-style handshake — see ordering notes).
//
// We use one block / one warp; with multiple producer threads we serialize
// enqueue with std::atomic on head_host (multiple-producer / single-
// consumer kernel = MPSC; the GPU consumer side is single-threaded).
//
// IMPORTANT — memory-model caveats:
//   - All shared state lives in host-mapped pinned memory. The device
//     accesses it over PCIe; reads bypass the L1/L2 caches if the
//     pointer is volatile, which we ensure with the volatile qualifier.
//   - Producer ordering on x86: writes to the slot fields *before* the
//     write to seq are observed in program order (TSO). On AArch64 you
//     would need an explicit release-store on seq.
//   - Consumer (GPU) ordering: __threadfence_system() on the device is
//     what makes the device's stores to `out` and `seq=DONE` visible to
//     the host before the host observes the head update via tail.

#include <atomic>
#include <chrono>
#include <cstdio>
#include <thread>
#include <vector>

#include "cuda_utils.h"

constexpr int CAP = 64;            // ring capacity (power of two)
constexpr int CAP_MASK = CAP - 1;

// Slot states.
enum : int {
    SLOT_EMPTY = 0,    // consumer can advance past it (after work=DONE)
    SLOT_READY = 1,    // producer has populated it; consumer may pick up
    SLOT_DONE  = 2,    // consumer has finished; producer may reuse
};

struct Slot {
    int seq;       // SLOT_* state
    int x;
    int y;
    int out;       // result written by consumer
};

// ----------------------------------------------------------------------------
// Persistent consumer kernel.
//
// Single block, single warp; only thread 0 polls the queue and dispatches
// "work" (here: out = x + y). Multi-thread consumers would need extra
// coordination; this matches the SPSC framing. Replace the work payload
// with a real per-task kernel for production use.
// ----------------------------------------------------------------------------
__global__ void consumer_kernel(volatile Slot* slots,
                                volatile int* head,
                                volatile int* tail,
                                volatile int* shutdown) {
    if (threadIdx.x != 0) return;

    int local_tail = 0;
    while (true) {
        // Poll for a populated slot or shutdown.
        for (;;) {
            if (*shutdown) return;
            int h = *head;
            if (h != local_tail) break;
            // __nanosleep reduces SM heat / power in spin-wait. 100-1000 ns
            // is typical; pick based on observed contention. Available on
            // sm_70+ (compute capability 7.0). This is a no-op on older arch.
            __nanosleep(200);
        }

        int idx = local_tail & CAP_MASK;
        // Wait for producer's release of this slot.
        while (slots[idx].seq != SLOT_READY) {
            __nanosleep(100);
        }
        // Acquire fence: ensure x/y reads happen-after the seq read.
        __threadfence_system();

        int x = slots[idx].x;
        int y = slots[idx].y;
        slots[idx].out = x + y;

        // Release the result + state to host. The system fence covers
        // host-mapped memory (vs. __threadfence which is device-only).
        __threadfence_system();
        slots[idx].seq = SLOT_DONE;

        local_tail++;
        *tail = local_tail;
    }
}

// ----------------------------------------------------------------------------
// Host helpers.
// ----------------------------------------------------------------------------
struct Queue {
    volatile Slot* slots_host;
    Slot*          slots_dev;
    volatile int*  head_host;
    int*           head_dev;
    volatile int*  tail_host;
    int*           tail_dev;
    volatile int*  shutdown_host;
    int*           shutdown_dev;

    std::atomic<int> head_atomic{0};
};

static int submit(Queue& q, int x, int y) {
    // Reserve a slot.
    int idx_seq = q.head_atomic.fetch_add(1, std::memory_order_relaxed);
    int idx     = idx_seq & CAP_MASK;

    // Wait until the consumer has freed (or never used) this physical slot.
    // Either it's the very first time (seq == 0/EMPTY) or the consumer
    // posted DONE for the previous occupant.
    while (true) {
        int s = q.slots_host[idx].seq;
        // Either fresh slot (initial 0 == EMPTY) or previously-completed (DONE).
        if (s == SLOT_EMPTY || s == SLOT_DONE) break;
        std::this_thread::yield();
    }

    q.slots_host[idx].x = x;
    q.slots_host[idx].y = y;
    // Release: the seq=READY store must happen-after x/y stores. On x86
    // this is guaranteed by TSO; on weaker hardware use std::atomic_thread_fence
    // (release) before the seq store. The host write is a normal store
    // because the device polls volatile.
    std::atomic_thread_fence(std::memory_order_release);
    q.slots_host[idx].seq = SLOT_READY;

    // Publish head. Single producer per slot (we serialized via fetch_add)
    // but we need the head value to monotonically advance; do that in
    // sequence number order.
    while (*q.head_host != idx_seq) std::this_thread::yield();
    *q.head_host = idx_seq + 1;

    // Wait for the result.
    while (q.slots_host[idx].seq != SLOT_DONE) {
        // Light back-off; on a real HFT path you'd spin tight.
    }
    int out = q.slots_host[idx].out;

    // Mark slot as recyclable: clear seq back to EMPTY so the next user
    // of this physical slot can populate fresh fields safely.
    std::atomic_thread_fence(std::memory_order_release);
    q.slots_host[idx].seq = SLOT_EMPTY;
    return out;
}

int main() {
    Queue q;

    Slot* slots_raw;
    CUDA_CHECK(cudaHostAlloc(&slots_raw, CAP * sizeof(Slot), cudaHostAllocMapped));
    int *head_raw, *tail_raw, *shutdown_raw;
    CUDA_CHECK(cudaHostAlloc(&head_raw,     sizeof(int), cudaHostAllocMapped));
    CUDA_CHECK(cudaHostAlloc(&tail_raw,     sizeof(int), cudaHostAllocMapped));
    CUDA_CHECK(cudaHostAlloc(&shutdown_raw, sizeof(int), cudaHostAllocMapped));

    for (int i = 0; i < CAP; ++i) {
        slots_raw[i].seq = SLOT_EMPTY;
        slots_raw[i].x = 0;
        slots_raw[i].y = 0;
        slots_raw[i].out = 0;
    }
    *head_raw = 0;
    *tail_raw = 0;
    *shutdown_raw = 0;

    q.slots_host = slots_raw;
    q.head_host = head_raw;
    q.tail_host = tail_raw;
    q.shutdown_host = shutdown_raw;

    CUDA_CHECK(cudaHostGetDevicePointer((void**)&q.slots_dev,    slots_raw,    0));
    CUDA_CHECK(cudaHostGetDevicePointer((void**)&q.head_dev,     head_raw,     0));
    CUDA_CHECK(cudaHostGetDevicePointer((void**)&q.tail_dev,     tail_raw,     0));
    CUDA_CHECK(cudaHostGetDevicePointer((void**)&q.shutdown_dev, shutdown_raw, 0));

    cudaStream_t s;
    CUDA_CHECK(cudaStreamCreate(&s));
    consumer_kernel<<<1, 32, 0, s>>>(q.slots_dev, q.head_dev,
                                     q.tail_dev,  q.shutdown_dev);

    // 4 host-thread producers, each submitting 100 items.
    constexpr int NTHREADS = 4;
    constexpr int PER = 100;
    std::vector<std::thread> ths;
    std::atomic<int> ok_count{0};
    auto t0 = std::chrono::steady_clock::now();
    for (int t = 0; t < NTHREADS; ++t) {
        ths.emplace_back([&, t]() {
            for (int i = 0; i < PER; ++i) {
                int x = t * 1000 + i;
                int y = i;
                int got = submit(q, x, y);
                if (got == x + y) ok_count.fetch_add(1, std::memory_order_relaxed);
            }
        });
    }
    for (auto& th : ths) th.join();
    auto t1 = std::chrono::steady_clock::now();

    *q.shutdown_host = 1;
    CUDA_CHECK(cudaStreamSynchronize(s));

    double total_us = std::chrono::duration<double, std::micro>(t1 - t0).count();
    int total = NTHREADS * PER;
    std::printf("ring buffer: %d/%d items correct  (%.2f us/item amortized "
                "across %d host threads)  %s\n",
                ok_count.load(), total, total_us / total, NTHREADS,
                ok_count.load() == total ? "PASS" : "FAIL");

    CUDA_CHECK(cudaStreamDestroy(s));
    CUDA_CHECK(cudaFreeHost(slots_raw));
    CUDA_CHECK(cudaFreeHost(head_raw));
    CUDA_CHECK(cudaFreeHost(tail_raw));
    CUDA_CHECK(cudaFreeHost(shutdown_raw));
    return ok_count.load() == total ? 0 : 1;
}
