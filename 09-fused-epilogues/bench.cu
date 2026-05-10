// Module 9 — perf comparison: fused vs unfused kernels.
//
// IMPORTANT: we initialize d_x with deterministic *random* values rather than
// cudaMemset(0) — a zero input makes softmax / layernorm numerically trivial
// (uniform output, exp(0)=1) and lets fundamentally broken kernels still
// produce plausible-looking timing results. Use real values so any drift in
// the fused kernels surfaces during the bench, not just in the verify pass.

#include <cstdint>
#include <cstdio>
#include <vector>

#include "bench.h"
#include "cuda_utils.h"
#include "kernels.cuh"

constexpr int ROWS = 4096;
constexpr int COLS = 4096;

// Deterministic LCG (Numerical Recipes 32-bit). Stable across runs/machines so
// bench numbers are reproducible without depending on rand().
static void fill_lcg(std::vector<float>& v, uint32_t seed) {
    uint32_t s = seed;
    for (auto& x : v) {
        s = s * 1664525u + 1013904223u;
        // Map to roughly [-1, 1] — numerically interesting but not extreme.
        x = ((int32_t)(s >> 1) - (int32_t)(1u << 30)) * (2.0f / (float)(1u << 30));
    }
}

int main() {
    cudaDeviceProp prop{};
    CUDA_CHECK(cudaGetDeviceProperties(&prop, 0));
    std::printf("Device: %s\nShape: %d x %d (%.0f MB)\n\n",
                prop.name, ROWS, COLS, ROWS * COLS * sizeof(float) / 1.0e6);

    float *d_x, *d_y, *d_g, *d_b, *d_r, *d_sum;
    float *d_tmp1, *d_tmp2, *d_tmp3;
    CUDA_CHECK(cudaMalloc(&d_x,    ROWS * COLS * sizeof(float)));
    CUDA_CHECK(cudaMalloc(&d_y,    ROWS * COLS * sizeof(float)));
    CUDA_CHECK(cudaMalloc(&d_r,    ROWS * COLS * sizeof(float)));
    CUDA_CHECK(cudaMalloc(&d_sum,  ROWS * COLS * sizeof(float)));
    CUDA_CHECK(cudaMalloc(&d_g,    COLS * sizeof(float)));
    CUDA_CHECK(cudaMalloc(&d_b,    COLS * sizeof(float)));
    CUDA_CHECK(cudaMalloc(&d_tmp1, ROWS * COLS * sizeof(float))); // exp buffer / nothing
    CUDA_CHECK(cudaMalloc(&d_tmp2, ROWS * sizeof(float)));        // row_max / mean
    CUDA_CHECK(cudaMalloc(&d_tmp3, ROWS * sizeof(float)));        // row_sum / rstd

    {
        std::vector<float> hx(ROWS * COLS), hr(ROWS * COLS), hg(COLS), hb(COLS);
        fill_lcg(hx, 0xc0ffee01u);
        fill_lcg(hr, 0xc0ffee02u);
        fill_lcg(hg, 0xc0ffee03u);
        fill_lcg(hb, 0xc0ffee04u);
        CUDA_CHECK(cudaMemcpy(d_x, hx.data(), ROWS * COLS * sizeof(float), cudaMemcpyHostToDevice));
        CUDA_CHECK(cudaMemcpy(d_r, hr.data(), ROWS * COLS * sizeof(float), cudaMemcpyHostToDevice));
        CUDA_CHECK(cudaMemcpy(d_g, hg.data(), COLS * sizeof(float), cudaMemcpyHostToDevice));
        CUDA_CHECK(cudaMemcpy(d_b, hb.data(), COLS * sizeof(float), cudaMemcpyHostToDevice));
    }

    constexpr int ITERS = 30, WARMUP = 5;

    auto bench = [&](auto&& fn) {
        for (int i = 0; i < WARMUP; ++i) fn();
        CUDA_CHECK(cudaDeviceSynchronize());
        float best = 1e30f;
        for (int i = 0; i < ITERS; ++i) {
            GpuTimer t;
            t.start();
            fn();
            best = std::min(best, t.stop_ms());
        }
        return best;
    };

    auto softmax_fused_run  = [&] { softmax_fused <<<ROWS, BLK>>>(d_x, d_y, ROWS, COLS); };
    auto softmax_online_run = [&] { softmax_online<<<ROWS, BLK>>>(d_x, d_y, ROWS, COLS); };
    auto softmax_unfused_run = [&] {
        softmax_unfused_max <<<ROWS, BLK>>>(d_x, d_tmp2, ROWS, COLS);
        softmax_unfused_exp <<<ROWS, BLK>>>(d_x, d_tmp2, d_tmp1, ROWS, COLS);
        softmax_unfused_sum <<<ROWS, BLK>>>(d_tmp1, d_tmp3, ROWS, COLS);
        softmax_unfused_norm<<<ROWS, BLK>>>(d_tmp1, d_tmp3, ROWS, COLS);
    };

    auto ln_fused_run    = [&] { layernorm_fused  <<<ROWS, BLK>>>(d_x, d_y, d_g, d_b, ROWS, COLS, 1e-5f); };
    auto ln_welford_run  = [&] { layernorm_welford<<<ROWS, BLK>>>(d_x, d_y, d_g, d_b, ROWS, COLS, 1e-5f); };
    auto ln_unfused_run  = [&] {
        layernorm_unfused_stats<<<ROWS, BLK>>>(d_x, d_tmp2, d_tmp3, ROWS, COLS, 1e-5f);
        layernorm_unfused_apply<<<ROWS, BLK>>>(d_x, d_y, d_g, d_b, d_tmp2, d_tmp3,
                                                ROWS, COLS);
    };
    auto ln_resadd_run = [&] {
        layernorm_residual_add_v0<<<ROWS, BLK>>>(d_x, d_r, d_sum, d_y, d_g, d_b,
                                                 ROWS, COLS, 1e-5f);
    };

    long bytes_in_out = (long)ROWS * COLS * 2 * sizeof(float);  // 1 read + 1 write
    auto report = [&](const char* name, float ms) {
        float gbs = bytes_in_out / (ms * 1.0e6);
        std::printf("  %-32s %.3f ms   ~%.0f GB/s of in+out\n", name, ms, gbs);
    };

    std::printf("Softmax\n");
    report("softmax_fused (3-pass)",     bench(softmax_fused_run));
    report("softmax_online (2-pass)",    bench(softmax_online_run));
    report("softmax_unfused (4 kernels)", bench(softmax_unfused_run));

    std::printf("\nLayerNorm\n");
    report("layernorm_fused (1 launch)",      bench(ln_fused_run));
    report("layernorm_welford (1 launch)",    bench(ln_welford_run));
    report("layernorm_unfused (2 launches)",  bench(ln_unfused_run));
    report("layernorm_residual_add (fused)",  bench(ln_resadd_run));

    // GEMM + bias + GELU at a small/medium shape — perf isn't the focus here,
    // but we want the fused-vs-post comparison visible.
    {
        constexpr int M = 1024, N = 1024, K = 1024;
        float *dA, *dB, *dBias, *dC;
        CUDA_CHECK(cudaMalloc(&dA,    M * K * sizeof(float)));
        CUDA_CHECK(cudaMalloc(&dB,    K * N * sizeof(float)));
        CUDA_CHECK(cudaMalloc(&dBias, N     * sizeof(float)));
        CUDA_CHECK(cudaMalloc(&dC,    M * N * sizeof(float)));
        std::vector<float> hA(M*K), hB(K*N), hBias(N);
        fill_lcg(hA,    0xfeedu);
        fill_lcg(hB,    0xbeefu);
        fill_lcg(hBias, 0xdeafu);
        CUDA_CHECK(cudaMemcpy(dA,    hA.data(),    M*K*sizeof(float), cudaMemcpyHostToDevice));
        CUDA_CHECK(cudaMemcpy(dB,    hB.data(),    K*N*sizeof(float), cudaMemcpyHostToDevice));
        CUDA_CHECK(cudaMemcpy(dBias, hBias.data(), N  *sizeof(float), cudaMemcpyHostToDevice));

        auto v0 = [&] { launch_gemm_bias_gelu_v0(dA, dB, dBias, dC, M, N, K); };
        auto v1 = [&] { launch_gemm_bias_gelu_v1(dA, dB, dBias, dC, M, N, K); };

        std::printf("\nGEMM+bias+GELU (M=N=K=%d)\n", M);
        std::printf("  %-32s %.3f ms\n", "v0 (gemm + epilogue kernel)", bench(v0));
        std::printf("  %-32s %.3f ms\n", "v1 (fused epilogue)",         bench(v1));

        CUDA_CHECK(cudaFree(dA));
        CUDA_CHECK(cudaFree(dB));
        CUDA_CHECK(cudaFree(dBias));
        CUDA_CHECK(cudaFree(dC));
    }

    CUDA_CHECK(cudaFree(d_x));
    CUDA_CHECK(cudaFree(d_y));
    CUDA_CHECK(cudaFree(d_r));
    CUDA_CHECK(cudaFree(d_sum));
    CUDA_CHECK(cudaFree(d_g));
    CUDA_CHECK(cudaFree(d_b));
    CUDA_CHECK(cudaFree(d_tmp1));
    CUDA_CHECK(cudaFree(d_tmp2));
    CUDA_CHECK(cudaFree(d_tmp3));
    return 0;
}
