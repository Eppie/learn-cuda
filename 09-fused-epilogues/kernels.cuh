#pragma once

#include <cuda_runtime.h>
#include <cuda_fp16.h>
#include <cmath>

#include "online_softmax.cuh"   // ms_pair, online_softmax_combine, online_softmax_block

constexpr int BLK     = 256;
constexpr int N_WARPS = BLK / 32;

// ============================================================================
// Block reduction helpers (mirrors of the M5 primitives — copied here to keep
// this module self-contained).
//
// Shared-buffer reuse hazard: block_reduce_sum / block_reduce_max both write
// `smem[0]` and exit through `__syncthreads()`. Callers may reuse the same
// `smem` buffer for back-to-back reductions (e.g. mean then variance) only
// because the trailing __syncthreads() inside the helper guarantees every
// thread has finished reading the previous result before the next call starts
// writing. If you ever delete that final sync, two consecutive reductions
// will race on smem[lane] of the warp scratch slot.
// ============================================================================
__device__ __forceinline__ float warp_reduce_sum(float v) {
    for (int o = 16; o > 0; o >>= 1) v += __shfl_xor_sync(0xffffffff, v, o);
    return v;
}

__device__ __forceinline__ float warp_reduce_max(float v) {
    for (int o = 16; o > 0; o >>= 1) v = fmaxf(v, __shfl_xor_sync(0xffffffff, v, o));
    return v;
}

__device__ __forceinline__ float block_reduce_sum(float v, float* smem) {
    int lane = threadIdx.x & 31;
    int warp = threadIdx.x >> 5;
    v = warp_reduce_sum(v);
    if (lane == 0) smem[warp] = v;
    __syncthreads();
    if (warp == 0) {
        v = (lane < N_WARPS) ? smem[lane] : 0.0f;
        v = warp_reduce_sum(v);
        if (lane == 0) smem[0] = v;
    }
    __syncthreads();
    return smem[0];
}

__device__ __forceinline__ float block_reduce_max(float v, float* smem) {
    int lane = threadIdx.x & 31;
    int warp = threadIdx.x >> 5;
    v = warp_reduce_max(v);
    if (lane == 0) smem[warp] = v;
    __syncthreads();
    if (warp == 0) {
        v = (lane < N_WARPS) ? smem[lane] : -INFINITY;
        v = warp_reduce_max(v);
        if (lane == 0) smem[0] = v;
    }
    __syncthreads();
    return smem[0];
}

// ============================================================================
// Fused 3-pass softmax: one block per row.
// ============================================================================
__global__ void softmax_fused(const float* __restrict__ in,
                              float* __restrict__ out,
                              int rows, int cols) {
    int row = blockIdx.x;
    if (row >= rows) return;

    const float* x = in + row * cols;
    float* y = out + row * cols;

    __shared__ float smem[N_WARPS];

    float m = -INFINITY;
    for (int i = threadIdx.x; i < cols; i += BLK) m = fmaxf(m, x[i]);
    float row_max = block_reduce_max(m, smem);

    float s = 0.0f;
    for (int i = threadIdx.x; i < cols; i += BLK) s += expf(x[i] - row_max);
    float row_sum = block_reduce_sum(s, smem);
    float inv = 1.0f / row_sum;

    for (int i = threadIdx.x; i < cols; i += BLK) {
        y[i] = expf(x[i] - row_max) * inv;
    }
}

// ============================================================================
// Online softmax: 2 reads + 1 write (vs 3+1 for the 3-pass version above).
// Per-thread running pair (m, l) is combined across the block via the
// associative merge formula; the output pass writes y[i] in one shot.
//
// This is the same recurrence M10 uses for FlashAttention.  The (m, l) pair
// and the merge formula live in `common/online_softmax.cuh` (lifted out of
// M5/M9/M10 to a single source of truth).
// ============================================================================
__global__ void softmax_online(const float* __restrict__ in,
                               float* __restrict__ out,
                               int rows, int cols) {
    int row = blockIdx.x;
    if (row >= rows) return;

    const float* x = in + row * cols;
    float* y = out + row * cols;

    __shared__ ms_pair os_scratch[N_WARPS];

    // Pass 1: walk x once, accumulating per-thread (m, s) via the recurrence.
    ms_pair acc = online_softmax_identity();
    for (int i = threadIdx.x; i < cols; i += BLK) {
        acc = online_softmax_combine(acc, online_softmax_singleton(x[i]));
    }

    // Block-reduce to get the global (row_max, row_sum), broadcast to every thread.
    ms_pair total = online_softmax_block_broadcast<BLK>(acc, os_scratch);
    float row_max = total.m;
    float inv     = 1.0f / total.s;

    // Pass 2: write normalized output.
    for (int i = threadIdx.x; i < cols; i += BLK) {
        y[i] = expf(x[i] - row_max) * inv;
    }
}

// ============================================================================
// Unfused softmax — 4 separate kernels.
// ============================================================================
__global__ void softmax_unfused_max(const float* in, float* row_max,
                                    int rows, int cols) {
    int row = blockIdx.x;
    if (row >= rows) return;
    const float* x = in + row * cols;

    __shared__ float smem[N_WARPS];

    float m = -INFINITY;
    for (int i = threadIdx.x; i < cols; i += BLK) m = fmaxf(m, x[i]);
    float r = block_reduce_max(m, smem);
    if (threadIdx.x == 0) row_max[row] = r;
}

__global__ void softmax_unfused_exp(const float* in, const float* row_max,
                                    float* exp_out, int rows, int cols) {
    int row = blockIdx.x;
    if (row >= rows) return;
    const float* x = in  + row * cols;
    float* y       = exp_out + row * cols;
    float m = row_max[row];
    for (int i = threadIdx.x; i < cols; i += BLK) y[i] = expf(x[i] - m);
}

__global__ void softmax_unfused_sum(const float* exp_in, float* row_sum,
                                    int rows, int cols) {
    int row = blockIdx.x;
    if (row >= rows) return;
    const float* y = exp_in + row * cols;

    __shared__ float smem[N_WARPS];

    float s = 0.0f;
    for (int i = threadIdx.x; i < cols; i += BLK) s += y[i];
    float r = block_reduce_sum(s, smem);
    if (threadIdx.x == 0) row_sum[row] = r;
}

__global__ void softmax_unfused_norm(float* exp_inout, const float* row_sum,
                                     int rows, int cols) {
    int row = blockIdx.x;
    if (row >= rows) return;
    float* y = exp_inout + row * cols;
    float inv = 1.0f / row_sum[row];
    for (int i = threadIdx.x; i < cols; i += BLK) y[i] *= inv;
}

// ============================================================================
// Fused LayerNorm: single read for sum(x) + sum(x²); second pass writes y.
// (E[x²] - μ² formulation — fast but loses precision for very large rows.)
// ============================================================================
__global__ void layernorm_fused(const float* __restrict__ in,
                                float* __restrict__ out,
                                const float* __restrict__ gamma,
                                const float* __restrict__ beta,
                                int rows, int cols, float eps) {
    int row = blockIdx.x;
    if (row >= rows) return;

    const float* x = in  + row * cols;
    float* y       = out + row * cols;

    __shared__ float smem[N_WARPS];
    __shared__ float mean_s, rstd_s;

    float s = 0.0f, sq = 0.0f;
    for (int i = threadIdx.x; i < cols; i += BLK) {
        float v = x[i];
        s  += v;
        sq += v * v;
    }
    s  = block_reduce_sum(s,  smem);
    sq = block_reduce_sum(sq, smem);

    if (threadIdx.x == 0) {
        float mean = s  / cols;
        float var  = sq / cols - mean * mean;
        mean_s = mean;
        rstd_s = rsqrtf(var + eps);
    }
    __syncthreads();
    float mean = mean_s;
    float rstd = rstd_s;

    for (int i = threadIdx.x; i < cols; i += BLK) {
        y[i] = gamma[i] * (x[i] - mean) * rstd + beta[i];
    }
}

// ============================================================================
// Welford LayerNorm: numerically stable mean/variance via Welford's online
// recurrence. Per-thread running (count, mean, M2); merge across threads with
// the parallel-Welford combine. Practically irrelevant at typical hidden dims
// (E[x²]-μ² is fine to ~1e-6 there) but a clean illustration of streaming
// algorithms in CUDA.
// ============================================================================
__device__ __forceinline__ void welford_combine(float& m_a, float& v_a, float& n_a,
                                                float  m_b, float  v_b, float  n_b) {
    if (n_b == 0.0f) return;
    float n  = n_a + n_b;
    float d  = m_b - m_a;
    float m  = m_a + d * (n_b / n);
    float vv = v_a + v_b + d * d * (n_a * n_b / n);
    m_a = m;  v_a = vv;  n_a = n;
}

__global__ void layernorm_welford(const float* __restrict__ in,
                                  float* __restrict__ out,
                                  const float* __restrict__ gamma,
                                  const float* __restrict__ beta,
                                  int rows, int cols, float eps) {
    int row = blockIdx.x;
    if (row >= rows) return;

    const float* x = in  + row * cols;
    float* y       = out + row * cols;

    // Per-thread Welford accumulators.
    float mean_t = 0.0f, M2_t = 0.0f, n_t = 0.0f;
    for (int i = threadIdx.x; i < cols; i += BLK) {
        float v = x[i];
        n_t += 1.0f;
        float d = v - mean_t;
        mean_t += d / n_t;
        M2_t   += d * (v - mean_t);
    }

    // Tree-reduce (mean, M2, n) within warp via shuffles on the combine rule.
    int lane = threadIdx.x & 31;
    int warp = threadIdx.x >> 5;
    for (int o = 16; o > 0; o >>= 1) {
        float m_other = __shfl_xor_sync(0xffffffff, mean_t, o);
        float v_other = __shfl_xor_sync(0xffffffff, M2_t,   o);
        float n_other = __shfl_xor_sync(0xffffffff, n_t,    o);
        welford_combine(mean_t, M2_t, n_t, m_other, v_other, n_other);
    }

    // Now lane 0 of each warp has that warp's combined (mean, M2, n).
    __shared__ float ws_mean[N_WARPS], ws_M2[N_WARPS], ws_n[N_WARPS];
    if (lane == 0) {
        ws_mean[warp] = mean_t;
        ws_M2[warp]   = M2_t;
        ws_n[warp]    = n_t;
    }
    __syncthreads();

    __shared__ float row_mean, row_rstd;
    if (warp == 0) {
        float m = (lane < N_WARPS) ? ws_mean[lane] : 0.0f;
        float v = (lane < N_WARPS) ? ws_M2[lane]   : 0.0f;
        float n = (lane < N_WARPS) ? ws_n[lane]    : 0.0f;
        for (int o = 16; o > 0; o >>= 1) {
            float m_other = __shfl_xor_sync(0xffffffff, m, o);
            float v_other = __shfl_xor_sync(0xffffffff, v, o);
            float n_other = __shfl_xor_sync(0xffffffff, n, o);
            welford_combine(m, v, n, m_other, v_other, n_other);
        }
        if (lane == 0) {
            row_mean = m;
            row_rstd = rsqrtf(v / n + eps);
        }
    }
    __syncthreads();

    float mean = row_mean, rstd = row_rstd;
    for (int i = threadIdx.x; i < cols; i += BLK) {
        y[i] = gamma[i] * (x[i] - mean) * rstd + beta[i];
    }
}

// ============================================================================
// LayerNorm + residual-add fused (transformer block pattern).
//   sum   = x + residual
//   y     = γ · (sum - μ(sum)) · rstd(sum) + β
// Outputs both the post-add `sum` (used as the residual for the *next* sublayer)
// and the post-norm `y` (fed into the sublayer). Common in pre-norm and
// post-norm transformer blocks alike.
// ============================================================================
__global__ void layernorm_residual_add_v0(const float* __restrict__ x,
                                          const float* __restrict__ residual,
                                          float* __restrict__ sum_out,
                                          float* __restrict__ y_out,
                                          const float* __restrict__ gamma,
                                          const float* __restrict__ beta,
                                          int rows, int cols, float eps) {
    int row = blockIdx.x;
    if (row >= rows) return;

    const float* xr = x        + row * cols;
    const float* rr = residual + row * cols;
    float* sr       = sum_out  + row * cols;
    float* yr       = y_out    + row * cols;

    __shared__ float smem[N_WARPS];
    __shared__ float mean_s, rstd_s;

    // Pass 1: x + residual → sum_out, accumulating Σsum and Σsum².
    float s = 0.0f, sq = 0.0f;
    for (int i = threadIdx.x; i < cols; i += BLK) {
        float v = xr[i] + rr[i];
        sr[i]   = v;             // residual stream for the next sublayer
        s      += v;
        sq     += v * v;
    }
    s  = block_reduce_sum(s,  smem);
    sq = block_reduce_sum(sq, smem);

    if (threadIdx.x == 0) {
        float mean = s  / cols;
        float var  = sq / cols - mean * mean;
        mean_s = mean;
        rstd_s = rsqrtf(var + eps);
    }
    __syncthreads();
    float mean = mean_s;
    float rstd = rstd_s;

    // Pass 2: read sr (or recompute xr+rr — same bytes; use sr for clarity).
    for (int i = threadIdx.x; i < cols; i += BLK) {
        yr[i] = gamma[i] * (sr[i] - mean) * rstd + beta[i];
    }
}

// ============================================================================
// FP16 LayerNorm — bandwidth-bound, so halving precision halves runtime.
// Inputs/outputs are __half; reductions still done in FP32 to stay numerically
// honest. cols must be even (we vectorize loads via __half2 internally for the
// stat pass, but to keep the pedagogy clean we read element-by-element here).
// ============================================================================
__global__ void layernorm_fused_fp16(const __half* __restrict__ in,
                                     __half* __restrict__ out,
                                     const __half* __restrict__ gamma,
                                     const __half* __restrict__ beta,
                                     int rows, int cols, float eps) {
    int row = blockIdx.x;
    if (row >= rows) return;

    const __half* x = in  + row * cols;
    __half*       y = out + row * cols;

    __shared__ float smem[N_WARPS];
    __shared__ float mean_s, rstd_s;

    float s = 0.0f, sq = 0.0f;
    for (int i = threadIdx.x; i < cols; i += BLK) {
        float v = __half2float(x[i]);
        s  += v;
        sq += v * v;
    }
    s  = block_reduce_sum(s,  smem);
    sq = block_reduce_sum(sq, smem);

    if (threadIdx.x == 0) {
        float mean = s  / cols;
        float var  = sq / cols - mean * mean;
        mean_s = mean;
        rstd_s = rsqrtf(var + eps);
    }
    __syncthreads();
    float mean = mean_s;
    float rstd = rstd_s;

    for (int i = threadIdx.x; i < cols; i += BLK) {
        float v = __half2float(x[i]);
        float g = __half2float(gamma[i]);
        float b = __half2float(beta[i]);
        y[i] = __float2half(g * (v - mean) * rstd + b);
    }
}

// ============================================================================
// Unfused LayerNorm — 2 separate kernels (stats + apply).
// ============================================================================
__global__ void layernorm_unfused_stats(const float* in, float* mean,
                                        float* rstd, int rows, int cols, float eps) {
    int row = blockIdx.x;
    if (row >= rows) return;
    const float* x = in + row * cols;

    __shared__ float smem[N_WARPS];

    float s = 0.0f;
    for (int i = threadIdx.x; i < cols; i += BLK) s += x[i];
    s = block_reduce_sum(s, smem);
    float m = s / cols;

    float sq = 0.0f;
    for (int i = threadIdx.x; i < cols; i += BLK) {
        float d = x[i] - m;
        sq += d * d;
    }
    sq = block_reduce_sum(sq, smem);

    if (threadIdx.x == 0) {
        mean[row] = m;
        rstd[row] = rsqrtf(sq / cols + eps);
    }
}

__global__ void layernorm_unfused_apply(const float* in, float* out,
                                        const float* gamma, const float* beta,
                                        const float* mean, const float* rstd,
                                        int rows, int cols) {
    int row = blockIdx.x;
    if (row >= rows) return;
    const float* x = in  + row * cols;
    float*       y = out + row * cols;
    float m = mean[row], r = rstd[row];
    for (int i = threadIdx.x; i < cols; i += BLK) {
        y[i] = gamma[i] * (x[i] - m) * r + beta[i];
    }
}

// ============================================================================
// GEMM + bias + GELU.
//
// We use the same warp-tiled GEMM as M06 v6 as the underlying matmul. v0 is
// "post-loop epilogue": GEMM writes raw C, then a separate pointwise kernel
// applies bias + GELU. v1 is the *fused* form where bias is added in registers
// after the MMA accumulation and GELU is evaluated before the global store —
// no intermediate DRAM round-trip for the GEMM result.
//
// Activation: tanh-approximation GELU (matches PyTorch `gelu(approximate='tanh')`).
//   gelu(x) ≈ 0.5 * x * (1 + tanh(√(2/π) * (x + 0.044715 * x³)))
// ============================================================================
__device__ __forceinline__ float gelu_tanh(float x) {
    constexpr float k0 = 0.7978845608028654f;       // √(2/π)
    constexpr float k1 = 0.044715f;
    float x3 = x * x * x;
    return 0.5f * x * (1.0f + tanhf(k0 * (x + k1 * x3)));
}

// Standalone post-pass: y[i] = gelu(C[i] + bias[col(i)]).
__global__ void bias_gelu_epilogue(float* __restrict__ C,
                                   const float* __restrict__ bias,
                                   int M, int N) {
    int col = blockIdx.x * blockDim.x + threadIdx.x;
    int row = blockIdx.y * blockDim.y + threadIdx.y;
    if (row < M && col < N) {
        float v = C[row * N + col] + bias[col];
        C[row * N + col] = gelu_tanh(v);
    }
}

// v0 GEMM: same kernel as M06 v6 (no fusion); runs followed by a separate
// bias_gelu_epilogue launch (see solution.cu / bench.cu for the launcher pair).
template <int BM, int BN, int BK, int WM, int WN, int TM, int TN>
__global__ void gemm_bias_gelu_v0(const float* __restrict__ A,
                                  const float* __restrict__ B,
                                  float* __restrict__ C,
                                  int M, int N, int K) {
    constexpr int NUM_WARPS   = (BM * BN) / (WM * WN);
    constexpr int NUM_THREADS = NUM_WARPS * 32;

    int cRow = blockIdx.y;
    int cCol = blockIdx.x;

    int warpIdx = threadIdx.x / 32;
    int warpRow = warpIdx / (BN / WN);
    int warpCol = warpIdx % (BN / WN);

    int laneIdx          = threadIdx.x % 32;
    int threadColInWarp  = laneIdx % (WN / TN);
    int threadRowInWarp  = laneIdx / (WN / TN);

    __shared__ float As[BK * BM];
    __shared__ float Bs[BK * BN];

    A += cRow * BM * K;
    B += cCol * BN;
    C += (cRow * BM + warpRow * WM) * N + cCol * BN + warpCol * WN;

    int innerRowA = threadIdx.x / (BK / 4);
    int innerColA = threadIdx.x % (BK / 4);
    constexpr int strideA = NUM_THREADS / (BK / 4);
    int innerRowB = threadIdx.x / (BN / 4);
    int innerColB = threadIdx.x % (BN / 4);
    constexpr int strideB = NUM_THREADS / (BN / 4);

    float threadResults[TM * TN] = {0.0f};
    float regM[TM] = {0.0f};
    float regN[TN] = {0.0f};

    for (int kBlock = 0; kBlock < K; kBlock += BK) {
        for (int off = 0; off < BM; off += strideA) {
            float4 t = reinterpret_cast<const float4*>(A + (innerRowA + off) * K)[innerColA];
            As[(innerColA * 4 + 0) * BM + innerRowA + off] = t.x;
            As[(innerColA * 4 + 1) * BM + innerRowA + off] = t.y;
            As[(innerColA * 4 + 2) * BM + innerRowA + off] = t.z;
            As[(innerColA * 4 + 3) * BM + innerRowA + off] = t.w;
        }
        for (int off = 0; off < BK; off += strideB) {
            reinterpret_cast<float4*>(Bs + (innerRowB + off) * BN)[innerColB] =
                reinterpret_cast<const float4*>(B + (innerRowB + off) * N)[innerColB];
        }
        __syncthreads();

        A += BK;
        B += BK * N;

        for (int dotIdx = 0; dotIdx < BK; ++dotIdx) {
            for (int i = 0; i < TM; ++i)
                regM[i] = As[dotIdx * BM + warpRow * WM + threadRowInWarp * TM + i];
            for (int i = 0; i < TN; ++i)
                regN[i] = Bs[dotIdx * BN + warpCol * WN + threadColInWarp * TN + i];
            for (int rm = 0; rm < TM; ++rm)
                for (int rn = 0; rn < TN; ++rn)
                    threadResults[rm * TN + rn] += regM[rm] * regN[rn];
        }
        __syncthreads();
    }

    for (int rm = 0; rm < TM; ++rm) {
        for (int rn = 0; rn < TN; rn += 4) {
            float4 t;
            t.x = threadResults[rm * TN + rn + 0];
            t.y = threadResults[rm * TN + rn + 1];
            t.z = threadResults[rm * TN + rn + 2];
            t.w = threadResults[rm * TN + rn + 3];
            reinterpret_cast<float4*>(
                &C[(threadRowInWarp * TM + rm) * N + threadColInWarp * TN + rn])[0] = t;
        }
    }
}

// v1: fused. bias is added in registers after MMA accumulation; GELU is
// evaluated before the global store. The MMA result never visits DRAM in raw
// form — one fewer (M·N)·sizeof(float) DRAM round-trip than v0.
template <int BM, int BN, int BK, int WM, int WN, int TM, int TN>
__global__ void gemm_bias_gelu_v1(const float* __restrict__ A,
                                  const float* __restrict__ B,
                                  const float* __restrict__ bias,
                                  float* __restrict__ C,
                                  int M, int N, int K) {
    constexpr int NUM_WARPS   = (BM * BN) / (WM * WN);
    constexpr int NUM_THREADS = NUM_WARPS * 32;

    int cRow = blockIdx.y;
    int cCol = blockIdx.x;

    int warpIdx = threadIdx.x / 32;
    int warpRow = warpIdx / (BN / WN);
    int warpCol = warpIdx % (BN / WN);

    int laneIdx          = threadIdx.x % 32;
    int threadColInWarp  = laneIdx % (WN / TN);
    int threadRowInWarp  = laneIdx / (WN / TN);

    __shared__ float As[BK * BM];
    __shared__ float Bs[BK * BN];

    A += cRow * BM * K;
    B += cCol * BN;
    C += (cRow * BM + warpRow * WM) * N + cCol * BN + warpCol * WN;
    // Column offset of this thread's TN-wide stripe within the global N axis.
    int biasColBase = cCol * BN + warpCol * WN + threadColInWarp * TN;

    int innerRowA = threadIdx.x / (BK / 4);
    int innerColA = threadIdx.x % (BK / 4);
    constexpr int strideA = NUM_THREADS / (BK / 4);
    int innerRowB = threadIdx.x / (BN / 4);
    int innerColB = threadIdx.x % (BN / 4);
    constexpr int strideB = NUM_THREADS / (BN / 4);

    float threadResults[TM * TN] = {0.0f};
    float regM[TM] = {0.0f};
    float regN[TN] = {0.0f};

    for (int kBlock = 0; kBlock < K; kBlock += BK) {
        for (int off = 0; off < BM; off += strideA) {
            float4 t = reinterpret_cast<const float4*>(A + (innerRowA + off) * K)[innerColA];
            As[(innerColA * 4 + 0) * BM + innerRowA + off] = t.x;
            As[(innerColA * 4 + 1) * BM + innerRowA + off] = t.y;
            As[(innerColA * 4 + 2) * BM + innerRowA + off] = t.z;
            As[(innerColA * 4 + 3) * BM + innerRowA + off] = t.w;
        }
        for (int off = 0; off < BK; off += strideB) {
            reinterpret_cast<float4*>(Bs + (innerRowB + off) * BN)[innerColB] =
                reinterpret_cast<const float4*>(B + (innerRowB + off) * N)[innerColB];
        }
        __syncthreads();

        A += BK;
        B += BK * N;

        for (int dotIdx = 0; dotIdx < BK; ++dotIdx) {
            for (int i = 0; i < TM; ++i)
                regM[i] = As[dotIdx * BM + warpRow * WM + threadRowInWarp * TM + i];
            for (int i = 0; i < TN; ++i)
                regN[i] = Bs[dotIdx * BN + warpCol * WN + threadColInWarp * TN + i];
            for (int rm = 0; rm < TM; ++rm)
                for (int rn = 0; rn < TN; ++rn)
                    threadResults[rm * TN + rn] += regM[rm] * regN[rn];
        }
        __syncthreads();
    }

    // Fused epilogue: load TN bias values once into registers, then for each
    // output row apply bias + GELU before the float4 store.
    float regBias[TN];
    for (int rn = 0; rn < TN; ++rn) regBias[rn] = bias[biasColBase + rn];

    for (int rm = 0; rm < TM; ++rm) {
        for (int rn = 0; rn < TN; rn += 4) {
            float4 t;
            t.x = gelu_tanh(threadResults[rm * TN + rn + 0] + regBias[rn + 0]);
            t.y = gelu_tanh(threadResults[rm * TN + rn + 1] + regBias[rn + 1]);
            t.z = gelu_tanh(threadResults[rm * TN + rn + 2] + regBias[rn + 2]);
            t.w = gelu_tanh(threadResults[rm * TN + rn + 3] + regBias[rn + 3]);
            reinterpret_cast<float4*>(
                &C[(threadRowInWarp * TM + rm) * N + threadColInWarp * TN + rn])[0] = t;
        }
    }
}

// Concrete launcher templates the chosen tile sizes. Same shape as M06 v6.
inline void launch_gemm_bias_gelu_v0(const float* A, const float* B,
                                     const float* bias, float* C,
                                     int M, int N, int K) {
    constexpr int BM = 128, BN = 128, BK = 16;
    constexpr int WM = 64,  WN = 32;
    constexpr int TM = 8,   TN = 8;
    dim3 block(((BM * BN) / (WM * WN)) * 32);
    dim3 grid((N + BN - 1) / BN, (M + BM - 1) / BM);
    gemm_bias_gelu_v0<BM, BN, BK, WM, WN, TM, TN><<<grid, block>>>(A, B, C, M, N, K);
    dim3 ebl(32, 8);
    dim3 egr((N + ebl.x - 1) / ebl.x, (M + ebl.y - 1) / ebl.y);
    bias_gelu_epilogue<<<egr, ebl>>>(C, bias, M, N);
}

inline void launch_gemm_bias_gelu_v1(const float* A, const float* B,
                                     const float* bias, float* C,
                                     int M, int N, int K) {
    constexpr int BM = 128, BN = 128, BK = 16;
    constexpr int WM = 64,  WN = 32;
    constexpr int TM = 8,   TN = 8;
    dim3 block(((BM * BN) / (WM * WN)) * 32);
    dim3 grid((N + BN - 1) / BN, (M + BM - 1) / BM);
    gemm_bias_gelu_v1<BM, BN, BK, WM, WN, TM, TN><<<grid, block>>>(A, B, bias, C, M, N, K);
}
