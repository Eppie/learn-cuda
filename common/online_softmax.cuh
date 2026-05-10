#pragma once

#include <cuda_runtime.h>

// =========================================================================
// Online softmax primitives — single-pass (max, sum-of-exp) recurrence.
//
// For a stream x_1, x_2, ..., x_N, the running pair (m, s) is updated as:
//
//     m_new = max(m_old, x)
//     s_new = s_old · exp(m_old - m_new) + 1 · exp(x - m_new)
//
// This is associative — combine() generalizes it to merging two streams'
// (m, s) — so it parallelizes through warp shuffles and shared-memory
// reductions.  Numerical safety: both arguments to exp are <= 0 by
// construction; the -INFINITY identity is special-cased so combining with it
// is exact (not NaN).
//
// FlashAttention uses the same recurrence at *tile* granularity, and also
// rescales a running output O by the per-side factor alpha.  For that case,
// use `online_softmax_combine_with_factors`, which returns alpha and beta
// alongside the new (m, s).
//
// All math here uses `__expf` (the fast intrinsic) for consistency with the
// rest of this course; if you need IEEE-correct `expf` swap it in your local
// copy.
// =========================================================================

struct ms_pair {
    float m;   // running max
    float s;   // running sum of exp(x_i - m)
};

__device__ __forceinline__ ms_pair online_softmax_identity() {
    return {-INFINITY, 0.0f};
}

__device__ __forceinline__ ms_pair online_softmax_singleton(float x) {
    return {x, 1.0f};
}

__device__ __forceinline__ ms_pair online_softmax_combine(ms_pair a, ms_pair b) {
    float m  = fmaxf(a.m, b.m);
    float ea = (a.m == -INFINITY) ? 0.0f : __expf(a.m - m);
    float eb = (b.m == -INFINITY) ? 0.0f : __expf(b.m - m);
    return {m, a.s * ea + b.s * eb};
}

// FA-style combine: returns the new (m, s) plus the per-side rescale factors
// alpha (apply to running output O and to the old (m, s) sum-of-exps) and beta
// (apply to the incoming partial).  Use this when you also need to rescale a
// running output O alongside (m, s).
struct os_combine_factors {
    ms_pair p;
    float   alpha;   // factor for the "a" (running) side
    float   beta;    // factor for the "b" (incoming partial) side
};

__device__ __forceinline__ os_combine_factors
online_softmax_combine_with_factors(ms_pair a, ms_pair b) {
    float m     = fmaxf(a.m, b.m);
    float alpha = (a.m == -INFINITY) ? 0.0f : __expf(a.m - m);
    float beta  = (b.m == -INFINITY) ? 0.0f : __expf(b.m - m);
    return {{m, a.s * alpha + b.s * beta}, alpha, beta};
}

// -------------------------------------------------------------------------
// Warp-scope reduction.  Returns the result valid in lane 0; other lanes
// hold partial states (broadcast yourself via __shfl_sync if needed, or call
// `online_softmax_warp_broadcast` below).
// -------------------------------------------------------------------------
__device__ __forceinline__ ms_pair online_softmax_warp(ms_pair p) {
    for (int offset = 16; offset > 0; offset >>= 1) {
        ms_pair other;
        other.m = __shfl_down_sync(0xffffffff, p.m, offset);
        other.s = __shfl_down_sync(0xffffffff, p.s, offset);
        p = online_softmax_combine(p, other);
    }
    return p;
}

__device__ __forceinline__ ms_pair online_softmax_warp(float x) {
    return online_softmax_warp(online_softmax_singleton(x));
}

// -------------------------------------------------------------------------
// Block-scope reduction.  Caller supplies shared scratch sized [BLK / 32].
// Result valid in lane 0 of warp 0.  For a result visible to every thread,
// either broadcast yourself through shared memory or use
// `online_softmax_block_broadcast` below.
//
// BLK must be a compile-time multiple of 32.
// -------------------------------------------------------------------------
template <int BLK>
__device__ __forceinline__ ms_pair
online_softmax_block(ms_pair p, ms_pair* warp_scratch) {
    static_assert(BLK % 32 == 0, "block size must be a multiple of warp size");
    constexpr int N_WARPS = BLK / 32;

    int lane    = threadIdx.x & 31;
    int warp_id = threadIdx.x >> 5;

    p = online_softmax_warp(p);
    if (lane == 0) warp_scratch[warp_id] = p;
    __syncthreads();

    if (warp_id == 0) {
        ms_pair q = (lane < N_WARPS) ? warp_scratch[lane] : online_softmax_identity();
        p = online_softmax_warp(q);
    }
    return p;
}

template <int BLK>
__device__ __forceinline__ ms_pair
online_softmax_block(float x, ms_pair* warp_scratch) {
    return online_softmax_block<BLK>(online_softmax_singleton(x), warp_scratch);
}

// Block reduction with broadcast — every thread receives the same result.
// Reuses `warp_scratch[0]` for the broadcast, so by the time this returns
// the contents of `warp_scratch` are not meaningful.
template <int BLK>
__device__ __forceinline__ ms_pair
online_softmax_block_broadcast(ms_pair p, ms_pair* warp_scratch) {
    p = online_softmax_block<BLK>(p, warp_scratch);
    if (threadIdx.x == 0) warp_scratch[0] = p;
    __syncthreads();
    return warp_scratch[0];
}
