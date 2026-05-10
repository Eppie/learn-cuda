// Module 2 — bandwidth comparison.
//
// Three sub-benches, all printing PASS/FAIL on a verify pass before reporting any
// timings:
//   (1) copy_scalar vs copy_strided vs copy_vec4 at N = 32 M (defeats L2)
//   (2) Working-set size sweep on copy_scalar: N ∈ {1, 4, 16, 64} M to expose the
//       L2-fits → DRAM-bound transition.
//   (3) AoS-vs-SoA: scale_aos uses struct{float x,y,z}[]; scale_soa uses three
//       float[] arrays. Same workload, completely different bandwidth.

#include <cstdio>
#include <vector>

#include "bench.h"
#include "cuda_utils.h"

// --- (1) and (2) kernels ---------------------------------------------------

__global__ void copy_scalar(const float* __restrict__ a,
                            float* __restrict__ b,
                            int n) {
    int gid = blockIdx.x * blockDim.x + threadIdx.x;
    if (gid < n) b[gid] = a[gid];
}

__global__ void copy_strided(const float* __restrict__ a,
                             float* __restrict__ b,
                             int n) {
    int gid = blockIdx.x * blockDim.x + threadIdx.x;
    if (gid >= n) return;
    int lane    = threadIdx.x & 31;
    int warp_id = gid >> 5;
    long long idx = ((long long)warp_id * 32 + lane) * 32;
    idx = idx % n;
    b[gid] = a[idx];
}

__global__ void copy_vec4(const float4* __restrict__ a,
                          float4* __restrict__ b,
                          int n4) {
    int gid = blockIdx.x * blockDim.x + threadIdx.x;
    if (gid < n4) b[gid] = a[gid];
}

// --- (3) AoS vs SoA --------------------------------------------------------

struct Vec3 { float x, y, z; };

// Touches all three components → full utilization either way; lesson is muted.
__global__ void scale_aos_full(Vec3* __restrict__ p, float s, int n) {
    int gid = blockIdx.x * blockDim.x + threadIdx.x;
    if (gid >= n) return;
    Vec3 v = p[gid];
    v.x *= s; v.y *= s; v.z *= s;
    p[gid] = v;
}

__global__ void scale_soa_full(float* __restrict__ px,
                               float* __restrict__ py,
                               float* __restrict__ pz,
                               float s, int n) {
    int gid = blockIdx.x * blockDim.x + threadIdx.x;
    if (gid >= n) return;
    px[gid] *= s;
    py[gid] *= s;
    pz[gid] *= s;
}

// Touches only .x — the common ML case (e.g., update one feature). Now AoS pays:
// each warp reads 32 × 12 = 384 bytes spanning 4 cache lines (160 B fetched, 128
// useful → 80% efficiency *if* the compiler can issue 12-byte loads — usually it
// emits 16-byte loads and overhead grows). SoA reads 32 × 4 = 128 bytes, 1 line,
// 100%. Expect SoA to be roughly 2–3× faster.
__global__ void scale_aos_sparse(Vec3* __restrict__ p, float s, int n) {
    int gid = blockIdx.x * blockDim.x + threadIdx.x;
    if (gid >= n) return;
    p[gid].x *= s;
}

__global__ void scale_soa_sparse(float* __restrict__ px, float s, int n) {
    int gid = blockIdx.x * blockDim.x + threadIdx.x;
    if (gid >= n) return;
    px[gid] *= s;
}

// ---------------------------------------------------------------------------

static int verify_copy(const float* host_a, const float* host_b, int n) {
    int errs = 0;
    for (int i = 0; i < n; ++i) if (host_a[i] != host_b[i]) ++errs;
    return errs;
}

static void run_copy_comparison() {
    constexpr int N     = 1 << 25;          // 32M floats = 128 MB > L2 (72 MB)
    constexpr int BLK   = 256;
    constexpr int ITERS = 50;
    const     int GRD   = (N + BLK - 1) / BLK;
    const     int GRD4  = (N / 4 + BLK - 1) / BLK;
    const     long bytes = static_cast<long>(N) * 2 * sizeof(float); // read + write

    std::printf("\n=== (1) copy comparison @ N=%d (%.0f MB working set, > L2) ===\n",
                N, N * sizeof(float) / 1.0e6);

    std::vector<float> h_a(N), h_b(N);
    for (int i = 0; i < N; ++i) h_a[i] = static_cast<float>(i & 0xfff);

    float *d_a, *d_b;
    CUDA_CHECK(cudaMalloc(&d_a, N * sizeof(float)));
    CUDA_CHECK(cudaMalloc(&d_b, N * sizeof(float)));
    CUDA_CHECK(cudaMemcpy(d_a, h_a.data(), N * sizeof(float), cudaMemcpyHostToDevice));

    auto verify_then_report = [&](const char* name, auto&& launch, auto&& expect) {
        CUDA_CHECK(cudaMemset(d_b, 0, N * sizeof(float)));
        launch();
        CUDA_CHECK_LAST();
        CUDA_CHECK(cudaMemcpy(h_b.data(), d_b, N * sizeof(float), cudaMemcpyDeviceToHost));
        int errs = expect();
        const char* tag = (errs == 0) ? "PASS" : "FAIL";
        float ms = bench_min_ms(ITERS, launch);
        float gbs = bytes / (ms * 1.0e6);
        std::printf("  %-22s [%s] %.3f ms   %.1f GB/s   (%.0f%% of peak)\n",
                    name, tag, ms, gbs, gbs / 1008.0f * 100.0f);
    };

    verify_then_report("copy_scalar",
        [&] { copy_scalar<<<GRD, BLK>>>(d_a, d_b, N); },
        [&] { return verify_copy(h_a.data(), h_b.data(), N); });

    verify_then_report("copy_strided",
        [&] { copy_strided<<<GRD, BLK>>>(d_a, d_b, N); },
        [&] {
            int errs = 0;
            for (int gid = 0; gid < N; ++gid) {
                int lane    = gid & 31;
                int warp_id = gid >> 5;
                long long idx = ((long long)warp_id * 32 + lane) * 32;
                idx = idx % N;
                if (h_b[gid] != h_a[idx]) ++errs;
            }
            return errs;
        });

    verify_then_report("copy_vec4",
        [&] { copy_vec4<<<GRD4, BLK>>>(reinterpret_cast<float4*>(d_a),
                                        reinterpret_cast<float4*>(d_b),
                                        N / 4); },
        [&] { return verify_copy(h_a.data(), h_b.data(), N); });

    CUDA_CHECK(cudaFree(d_a));
    CUDA_CHECK(cudaFree(d_b));
}

static void run_l2_size_sweep() {
    constexpr int BLK   = 256;
    constexpr int ITERS = 50;
    const int sizes[] = { 1 << 20, 1 << 22, 1 << 24, 1 << 26 };  // 1, 4, 16, 64 M floats
    const long L2_SIZE_BYTES = 72L * 1024 * 1024;

    std::printf("\n=== (2) L2 effects: working-set sweep on copy_scalar ===\n");
    std::printf("  L2 = %ld MB on RTX 4090. Below ~%.0f MB (read+write), data fits in L2.\n",
                L2_SIZE_BYTES / (1L << 20), L2_SIZE_BYTES / 2.0 / 1e6);
    std::printf("  %-9s  %-9s  %-9s  %-9s\n", "N", "MB (rd+wr)", "ms", "GB/s");

    for (int N : sizes) {
        const long bytes = static_cast<long>(N) * 2 * sizeof(float);
        const int  GRD   = (N + BLK - 1) / BLK;
        float *d_a, *d_b;
        CUDA_CHECK(cudaMalloc(&d_a, N * sizeof(float)));
        CUDA_CHECK(cudaMalloc(&d_b, N * sizeof(float)));
        CUDA_CHECK(cudaMemset(d_a, 0, N * sizeof(float)));
        float ms = bench_min_ms(ITERS, [=] { copy_scalar<<<GRD, BLK>>>(d_a, d_b, N); });
        float gbs = bytes / (ms * 1.0e6);
        const char* note = (bytes <= L2_SIZE_BYTES) ? "  <- L2-resident" : "  <- DRAM-bound";
        std::printf("  %-9d  %-10.1f %-9.3f  %-9.1f%s\n",
                    N, bytes / 1.0e6, ms, gbs, note);
        CUDA_CHECK(cudaFree(d_a));
        CUDA_CHECK(cudaFree(d_b));
    }
}

static void run_aos_vs_soa() {
    constexpr int N     = 1 << 24;          // 16M points
    constexpr int BLK   = 256;
    constexpr int ITERS = 50;
    const     int GRD   = (N + BLK - 1) / BLK;

    std::printf("\n=== (3) AoS vs SoA: 3D-point scaling (N = %d points) ===\n", N);
    std::printf("  Two regimes, two stories:\n");
    std::printf("  - 'full' touches all three components per point (.x, .y, .z).\n");
    std::printf("  - 'sparse' touches only .x — the realistic update-one-feature pattern.\n");

    Vec3*  d_pts_full;   Vec3*  d_pts_sparse;
    float *d_x_full,  *d_y_full,  *d_z_full;
    float *d_x_sparse;
    CUDA_CHECK(cudaMalloc(&d_pts_full,   N * sizeof(Vec3)));
    CUDA_CHECK(cudaMalloc(&d_pts_sparse, N * sizeof(Vec3)));
    CUDA_CHECK(cudaMalloc(&d_x_full,     N * sizeof(float)));
    CUDA_CHECK(cudaMalloc(&d_y_full,     N * sizeof(float)));
    CUDA_CHECK(cudaMalloc(&d_z_full,     N * sizeof(float)));
    CUDA_CHECK(cudaMalloc(&d_x_sparse,   N * sizeof(float)));

    std::vector<Vec3>  h_pts(N);
    std::vector<float> h_x(N), h_y(N), h_z(N);
    for (int i = 0; i < N; ++i) {
        h_pts[i] = {float(i), float(2*i), float(3*i)};
        h_x[i] = float(i);
        h_y[i] = float(2*i);
        h_z[i] = float(3*i);
    }
    CUDA_CHECK(cudaMemcpy(d_pts_full,   h_pts.data(), N * sizeof(Vec3),  cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemcpy(d_pts_sparse, h_pts.data(), N * sizeof(Vec3),  cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemcpy(d_x_full,     h_x.data(),   N * sizeof(float), cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemcpy(d_y_full,     h_y.data(),   N * sizeof(float), cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemcpy(d_z_full,     h_z.data(),   N * sizeof(float), cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemcpy(d_x_sparse,   h_x.data(),   N * sizeof(float), cudaMemcpyHostToDevice));

    constexpr float SCALE = 0.5f;

    // --- Verify "full" pair --------------------------------------------------
    scale_aos_full<<<GRD, BLK>>>(d_pts_full, SCALE, N);
    CUDA_CHECK_LAST();
    CUDA_CHECK(cudaMemcpy(h_pts.data(), d_pts_full, N * sizeof(Vec3), cudaMemcpyDeviceToHost));
    int aos_full_errs = 0;
    for (int i = 0; i < N; ++i) {
        if (h_pts[i].x != float(i)*SCALE || h_pts[i].y != float(2*i)*SCALE
            || h_pts[i].z != float(3*i)*SCALE) ++aos_full_errs;
    }
    CUDA_CHECK(cudaMemcpy(d_pts_full, h_pts.data(), N * sizeof(Vec3), cudaMemcpyHostToDevice)); // restore

    scale_soa_full<<<GRD, BLK>>>(d_x_full, d_y_full, d_z_full, SCALE, N);
    CUDA_CHECK_LAST();
    CUDA_CHECK(cudaMemcpy(h_x.data(), d_x_full, N * sizeof(float), cudaMemcpyDeviceToHost));
    CUDA_CHECK(cudaMemcpy(h_y.data(), d_y_full, N * sizeof(float), cudaMemcpyDeviceToHost));
    CUDA_CHECK(cudaMemcpy(h_z.data(), d_z_full, N * sizeof(float), cudaMemcpyDeviceToHost));
    int soa_full_errs = 0;
    for (int i = 0; i < N; ++i) {
        if (h_x[i] != float(i)*SCALE || h_y[i] != float(2*i)*SCALE
            || h_z[i] != float(3*i)*SCALE) ++soa_full_errs;
    }
    // restore SoA inputs
    for (int i = 0; i < N; ++i) { h_x[i] = float(i); h_y[i] = float(2*i); h_z[i] = float(3*i); }
    CUDA_CHECK(cudaMemcpy(d_x_full, h_x.data(), N * sizeof(float), cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemcpy(d_y_full, h_y.data(), N * sizeof(float), cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemcpy(d_z_full, h_z.data(), N * sizeof(float), cudaMemcpyHostToDevice));

    // --- Verify "sparse" pair ------------------------------------------------
    scale_aos_sparse<<<GRD, BLK>>>(d_pts_sparse, SCALE, N);
    CUDA_CHECK_LAST();
    CUDA_CHECK(cudaMemcpy(h_pts.data(), d_pts_sparse, N * sizeof(Vec3), cudaMemcpyDeviceToHost));
    int aos_sparse_errs = 0;
    for (int i = 0; i < N; ++i) {
        if (h_pts[i].x != float(i)*SCALE) ++aos_sparse_errs;
    }
    for (int i = 0; i < N; ++i) h_pts[i] = {float(i), float(2*i), float(3*i)};
    CUDA_CHECK(cudaMemcpy(d_pts_sparse, h_pts.data(), N * sizeof(Vec3), cudaMemcpyHostToDevice));

    scale_soa_sparse<<<GRD, BLK>>>(d_x_sparse, SCALE, N);
    CUDA_CHECK_LAST();
    CUDA_CHECK(cudaMemcpy(h_x.data(), d_x_sparse, N * sizeof(float), cudaMemcpyDeviceToHost));
    int soa_sparse_errs = 0;
    for (int i = 0; i < N; ++i) {
        if (h_x[i] != float(i)*SCALE) ++soa_sparse_errs;
    }
    for (int i = 0; i < N; ++i) h_x[i] = float(i);
    CUDA_CHECK(cudaMemcpy(d_x_sparse, h_x.data(), N * sizeof(float), cudaMemcpyHostToDevice));

    // --- Time it -------------------------------------------------------------
    // Bytes for "full": rd+wr of 12 B per element = 24 B / elt.
    // Bytes for "sparse" — what's *useful*: rd+wr of one float = 8 B / elt.
    // (The AoS sparse variant likely moves more bytes than 'useful' — that's
    //  the point. Reported GB/s uses *useful* bytes so the gap shows up.)
    const long bytes_full   = static_cast<long>(N) * 6 * sizeof(float);
    const long bytes_sparse = static_cast<long>(N) * 2 * sizeof(float);

    auto report = [&](const char* name, int errs, float ms, long bytes_useful) {
        const char* tag = (errs == 0) ? "PASS" : "FAIL";
        float gbs = bytes_useful / (ms * 1.0e6);
        std::printf("  %-22s [%s] %.3f ms   %.1f GB/s useful   (%.0f%% of peak)\n",
                    name, tag, ms, gbs, gbs / 1008.0f * 100.0f);
    };

    std::printf("\n  -- Full (touch all 3 components) --\n");
    float ms_aos_full = bench_min_ms(ITERS, [=] { scale_aos_full<<<GRD, BLK>>>(d_pts_full, 1.0f, N); });
    float ms_soa_full = bench_min_ms(ITERS, [=] { scale_soa_full<<<GRD, BLK>>>(d_x_full, d_y_full, d_z_full, 1.0f, N); });
    report("AoS full",  aos_full_errs,  ms_aos_full,  bytes_full);
    report("SoA full",  soa_full_errs,  ms_soa_full,  bytes_full);
    std::printf("  (Both touch every byte; AoS and SoA should be ~tied here.)\n");

    std::printf("\n  -- Sparse (touch only .x) --\n");
    float ms_aos_sp = bench_min_ms(ITERS, [=] { scale_aos_sparse<<<GRD, BLK>>>(d_pts_sparse, 1.0f, N); });
    float ms_soa_sp = bench_min_ms(ITERS, [=] { scale_soa_sparse<<<GRD, BLK>>>(d_x_sparse, 1.0f, N); });
    report("AoS sparse",  aos_sparse_errs,  ms_aos_sp,  bytes_sparse);
    report("SoA sparse",  soa_sparse_errs,  ms_soa_sp,  bytes_sparse);
    std::printf("  (AoS sparse forces the warp to fetch 12 B per .x — 3× the bytes\n"
                "   it actually uses. Expect SoA to be roughly 2-3× faster on the\n"
                "   useful-bytes metric.)\n");

    CUDA_CHECK(cudaFree(d_pts_full));
    CUDA_CHECK(cudaFree(d_pts_sparse));
    CUDA_CHECK(cudaFree(d_x_full));
    CUDA_CHECK(cudaFree(d_y_full));
    CUDA_CHECK(cudaFree(d_z_full));
    CUDA_CHECK(cudaFree(d_x_sparse));
}

int main() {
    cudaDeviceProp prop{};
    CUDA_CHECK(cudaGetDeviceProperties(&prop, 0));
    std::printf("Device: %s  (CC %d.%d)\n", prop.name, prop.major, prop.minor);

    run_copy_comparison();
    run_l2_size_sweep();
    run_aos_vs_soa();

    std::printf("\nRTX 4090 peak DRAM bandwidth: ~1008 GB/s.\n");
    return 0;
}
