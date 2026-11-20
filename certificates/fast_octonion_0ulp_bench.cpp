#include <iostream>
#include <iomanip>
#include <vector>
#include <array>
#include <random>
#include <chrono>
#include <cstring>
#include "fast_octonion_kernels.hpp"

static const int T_O_REF[512] = {1,0,0,0,0,0,0,0,0,1,0,0,0,0,0,0,0,0,1,0,0,0,0,0,0,0,0,1,0,0,0,0,0,0,0,0,1,0,0,0,0,0,0,0,0,1,0,0,0,0,0,0,0,0,1,0,0,0,0,0,0,0,0,1,0,1,0,0,0,0,0,0,-1,0,0,0,0,0,0,0,0,0,0,0,1,0,0,0,0,0,0,0,0,0,0,1,0,0,-1,0,0,0,0,0,0,0,0,0,0,0,1,0,0,0,0,0,0,-1,0,0,0,0,0,-1,0,0,0,0,0,0,1,0,0,0,0,0,0,0,0,0,-1,0,0,0,-1,0,0,0,0,0,0,0,0,0,0,0,0,1,0,0,0,1,0,0,0,0,0,0,0,0,0,-1,0,0,0,0,0,0,0,0,0,0,0,1,0,0,0,0,0,0,-1,0,0,0,0,1,0,0,0,0,0,0,0,0,0,0,0,-1,0,0,0,0,0,-1,0,0,-1,0,0,0,0,0,0,0,0,0,0,0,0,0,1,0,0,0,1,0,0,0,0,0,0,0,0,0,-1,0,0,0,0,1,0,0,0,0,0,0,0,0,0,0,1,0,0,0,0,0,1,0,0,0,0,0,0,-1,0,0,0,0,0,0,0,0,0,0,0,0,-1,0,-1,0,0,0,0,0,0,0,0,0,0,0,0,0,0,1,0,0,0,1,0,0,0,0,0,0,0,0,0,-1,0,0,0,0,0,0,0,1,0,0,0,0,0,0,0,0,-1,0,0,0,0,1,0,0,0,0,0,0,-1,0,0,0,0,0,0,0,0,0,0,0,0,-1,-1,0,0,0,0,0,0,0,0,1,0,0,0,0,0,0,0,0,0,0,1,0,0,0,0,0,0,0,0,0,1,0,0,0,0,0,0,1,0,0,0,0,0,0,0,0,0,-1,0,0,0,0,1,0,0,0,0,0,0,-1,0,0,0,0,0,-1,0,0,0,0,0,0,-1,0,0,0,0,0,0,0,0,0,1,0,0,0,0,0,0,0,0,0,0,0,0,1,0,0,0,1,0,0,0,0,0,0,0,0,0,0,1,0,0,-1,0,0,0,0,0,0,0,0,0,0,0,1,0,0,0,0,0,0,-1,0,0,0,0,0,-1,0,0,0,0,0,-1,0,0,0,0,0,0,0};
static const int T_Z_REF[512] = {1,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,1,0,0,0,0,0,0,0,0,1,0,0,0,0,0,0,0,0,1,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,1,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,1,0,0,0,0,0,0,0,0,1,0,0,0,0,0,0,0,0,1,0,0,0,0,0,0,0,0,0,0,1,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,-1,0,0,0,0,0,0,1,0,-1,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,1,0,0,0,0,0,0,0,0,0,0,0,1,0,0,0,0,0,0,0,0,0,0,0,0,0,-1,0,0,0,0,0,0,0,0,0,0,-1,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,1,0,0,0,0,0,0,0,0,0,-1,0,0,0,0,0,0,1,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,-1,0,0,0,0,0,0,0,0,0,0,0,0,1,0,0,0,0,0,0,0,0,0,0,0,-1,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,-1,0,0,0,0,0,0,1,0,0,0,0,0,0,0,0,0,0,1,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,-1,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,1,0,0,0,0,0,0,0,0,0,0,0,0,0,-1,0,0,0,0,0,0,0,0,0,0,0,0,1,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,-1,0,0,0,0,0,0,0,0,0,-1,0,0,0,0,0,0,1,0,0,0,0,0,0,0,0,0,0,0,0,0};

template <typename Fn>
int check_basis_512(Fn fn, const int ref[512]) {
    int mis = 0;
    int64_t x[8], y[8], out[8];
    for (int i=0; i<8; ++i) {
        std::memset(x, 0, sizeof(x)); x[i] = 1;
        for (int j=0; j<8; ++j) {
            std::memset(y, 0, sizeof(y)); y[j] = 1;
            fn(x, y, out);
            for (int k=0; k<8; ++k) {
                if (out[k] != ref[i*64 + j*8 + k]) ++mis;
            }
        }
    }
    return mis;
}

template <typename Fn>
int check_basis_512_qi(Fn fn, const int ref[512]) {
    int mis = 0;
    int64_t xr[8], xi[8], yr[8], yi[8], outr[8], outi[8];
    // Test all 4 combinations (1, i) x (1, i) on all 512 basis entries = 2,048 checks!
    const int ph[2][2] = { {1,0}, {0,1} };
    for (int pa=0; pa<2; ++pa) {
        for (int pb=0; pb<2; ++pb) {
            int exp_re_s = ph[pa][0]*ph[pb][0] - ph[pa][1]*ph[pb][1];
            int exp_im_s = ph[pa][0]*ph[pb][1] + ph[pa][1]*ph[pb][0];
            for (int i=0; i<8; ++i) {
                std::memset(xr,0,sizeof(xr)); std::memset(xi,0,sizeof(xi));
                xr[i] = ph[pa][0]; xi[i] = ph[pa][1];
                for (int j=0; j<8; ++j) {
                    std::memset(yr,0,sizeof(yr)); std::memset(yi,0,sizeof(yi));
                    yr[j] = ph[pb][0]; yi[j] = ph[pb][1];
                    fn(xr, xi, yr, yi, outr, outi);
                    for (int k=0; k<8; ++k) {
                        int v = ref[i*64 + j*8 + k];
                        if (outr[k] != (int64_t)exp_re_s * v || outi[k] != (int64_t)exp_im_s * v) ++mis;
                    }
                }
            }
        }
    }
    return mis;
}

template <typename Fn, typename RefFn>
int check_random_100k(Fn fn, RefFn ref_fn) {
    std::mt19937_64 rng(20261120ULL);
    std::uniform_int_distribution<int64_t> dist(-1000000LL, 1000000LL);
    int64_t x[8], y[8], out[8], ref[8];
    int mis = 0;
    for (int t=0; t<100000; ++t) {
        for (int i=0; i<8; ++i) { x[i] = dist(rng); y[i] = dist(rng); }
        fn(x, y, out);
        ref_fn(x, y, ref);
        for (int i=0; i<8; ++i) if (out[i] != ref[i]) ++mis;
    }
    return mis;
}

template <typename Fn, typename RefFn>
int check_random_100k_qi(Fn fn, RefFn ref_fn) {
    std::mt19937_64 rng(20261120ULL);
    std::uniform_int_distribution<int64_t> dist(-1000000LL, 1000000LL);
    int64_t xr[8], xi[8], yr[8], yi[8], or1[8], oi1[8], or2[8], oi2[8];
    int mis = 0;
    for (int t=0; t<100000; ++t) {
        for (int i=0; i<8; ++i) {
            xr[i] = dist(rng); xi[i] = dist(rng);
            yr[i] = dist(rng); yi[i] = dist(rng);
        }
        fn(xr, xi, yr, yi, or1, oi1);
        ref_fn(xr, xi, yr, yi, or2, oi2);
        for (int i=0; i<8; ++i) if (or1[i] != or2[i] || oi1[i] != oi2[i]) ++mis;
    }
    return mis;
}

template <typename Fn>
double bench_real(Fn fn, int iters, int64_t &sink) {
    constexpr int NBUF = 256;
    alignas(64) int64_t xs[NBUF][8], ys[NBUF][8], out[8];
    std::mt19937_64 rng(777);
    for (int b=0; b<NBUF; ++b)
        for (int i=0; i<8; ++i) { xs[b][i] = (int64_t)(rng() % 2048) - 1024; ys[b][i] = (int64_t)(rng() % 2048) - 1024; }
    for (int w=0; w<20000; ++w) fn(xs[w & 255], ys[(w+1) & 255], out);
    double best_ns = 1e18;
    for (int rep=0; rep<5; ++rep) {
        int64_t acc = 0;
        auto t0 = std::chrono::steady_clock::now();
        for (int it=0; it<iters; ++it) {
            fn(xs[it & 255], ys[(it*3 + 1) & 255], out);
            acc ^= out[it & 7];
            xs[(it + 7) & 255][0] = (out[0] & 1023); // light data dependency prevents dead-code elimination
        }
        auto t1 = std::chrono::steady_clock::now();
        sink ^= acc;
        double ns = std::chrono::duration<double, std::nano>(t1 - t0).count() / iters;
        if (ns < best_ns) best_ns = ns;
    }
    return best_ns;
}

template <typename Fn>
double bench_qi(Fn fn, int iters, int64_t &sink) {
    constexpr int NBUF = 256;
    alignas(64) int64_t xr[NBUF][8], xi[NBUF][8], yr[NBUF][8], yi[NBUF][8], outr[8], outi[8];
    std::mt19937_64 rng(888);
    for (int b=0; b<NBUF; ++b)
        for (int i=0; i<8; ++i) {
            xr[b][i] = (int64_t)(rng() % 2048) - 1024; xi[b][i] = (int64_t)(rng() % 2048) - 1024;
            yr[b][i] = (int64_t)(rng() % 2048) - 1024; yi[b][i] = (int64_t)(rng() % 2048) - 1024;
        }
    for (int w=0; w<20000; ++w) fn(xr[w&255], xi[w&255], yr[(w+1)&255], yi[(w+1)&255], outr, outi);
    double best_ns = 1e18;
    for (int rep=0; rep<5; ++rep) {
        int64_t acc = 0;
        auto t0 = std::chrono::steady_clock::now();
        for (int it=0; it<iters; ++it) {
            int a = it & 255, b = (it*3 + 1) & 255;
            fn(xr[a], xi[a], yr[b], yi[b], outr, outi);
            acc ^= (outr[it & 7] + outi[it & 7]);
            xr[(it + 7) & 255][0] = (outr[0] & 1023);
        }
        auto t1 = std::chrono::steady_clock::now();
        sink ^= acc;
        double ns = std::chrono::duration<double, std::nano>(t1 - t0).count() / iters;
        if (ns < best_ns) best_ns = ns;
    }
    return best_ns;
}

int main() {
    int64_t sink = 0;
    const int ITERS = 2000000;
    std::cout << "{\n";
    auto emit_r = [&](const char* key, auto fn, const int* ref, auto ref_fn, bool comma) {
        int b_mis = check_basis_512(fn, ref);
        int r_mis = check_random_100k(fn, ref_fn);
        double ns = bench_real(fn, ITERS, sink);
        std::cout << "  \"" << key << "\": {\"basis_512_err\":" << b_mis
                  << ",\"rand_100k_err\":" << r_mis
                  << ",\"ns_per_op\":" << std::fixed << std::setprecision(3) << ns
                  << ",\"mops_per_sec\":" << std::fixed << std::setprecision(2) << (1000.0 / ns)
                  << "}" << (comma ? "," : "") << "\n";
    };
    auto emit_q = [&](const char* key, auto fn, bool comma) {
        int b_mis = check_basis_512_qi(fn, T_O_REF);
        int r_mis = check_random_100k_qi(fn, mul_qi_naive64);
        double ns = bench_qi(fn, ITERS, sink);
        std::cout << "  \"" << key << "\": {\"basis_512_err\":" << b_mis
                  << ",\"rand_100k_err\":" << r_mis
                  << ",\"ns_per_op\":" << std::fixed << std::setprecision(3) << ns
                  << ",\"mops_per_sec\":" << std::fixed << std::setprecision(2) << (1000.0 / ns)
                  << "}" << (comma ? "," : "") << "\n";
    };

    emit_r("Zorn_Dense_64", mul_zorn_dense64, T_Z_REF, mul_zorn_sparse32, true);
    emit_r("Zorn_NaiveSparse_32", mul_zorn_sparse32, T_Z_REF, mul_zorn_sparse32, true);
    emit_r("Zorn_PureZ_R23_#102", mul_zorn_r23, T_Z_REF, mul_zorn_sparse32, true);
    emit_r("Zorn_Dyadic_R22_#102", mul_zorn_r22, T_Z_REF, mul_zorn_sparse32, true);

    emit_q("Octo_Qi_Naive_64", mul_qi_naive64, true);
    emit_q("Octo_Qi_Cariow_30", mul_qi_cariow30, true);
    emit_q("Octo_Qi_ButterflyZorn_R23_#102", mul_qi_bfly_r23, true);
    emit_q("Octo_Qi_ButterflyZorn_R22_#102", mul_qi_bfly_r22, true);
    emit_q("Octo_Qi_DirectCert_R22_#102", mul_qi_direct_r22, true);

    emit_r("Octo_Z_Naive_64", mul_octo_naive64, T_O_REF, mul_octo_naive64, true);
    emit_r("Octo_Z_Makarov_32", mul_octo_makarov32, T_O_REF, mul_octo_naive64, true);
    emit_r("Octo_Z_Cariow_30", mul_octo_cariow30, T_O_REF, mul_octo_naive64, true);
    emit_r("Octo_Z_Dyadic_R26_#102", mul_octo_r26, T_O_REF, mul_octo_naive64, true);

    // Block-Matrix 0-ULP Octonion GEMM Benchmark (D=32, 8x32x32 int64 matrices: models NPU/Layer Octonion GEMM!)
    // Given rank-R real/dyadic or butterfly-Gaussian scheme: R integer D x D GEMMs + O(D^2) pre/post linear combs!
    constexpr int D = 32;
    constexpr int DD = D * D;
    auto gemm_i64 = [](const int64_t* __restrict A, const int64_t* __restrict B, int64_t* __restrict C) {
        std::memset(C, 0, DD * sizeof(int64_t));
        for (int i=0; i<D; ++i) {
            for (int k=0; k<D; ++k) {
                int64_t aik = A[i*D + k];
                for (int j=0; j<D; ++j) C[i*D + j] += aik * B[k*D + j];
            }
        }
    };
    std::vector<int64_t> Xz(8*DD), Yz(8*DD), OutRef(8*DD), OutTest(8*DD);
    std::mt19937_64 brng(999);
    for (int idx=0; idx<8*DD; ++idx) { Xz[idx] = (int64_t)(brng()%64)-32; Yz[idx] = (int64_t)(brng()%64)-32; }

    // Benchmark 0-ULP D=32 Zorn GEMM: NaiveSparse 32-GEMM vs #102 Pure-Z 23-GEMM vs Dense 64-GEMM!
    auto bench_block_zorn_naive32 = [&]() {
        std::memset(OutRef.data(), 0, 8*DD*sizeof(int64_t));
        std::vector<int64_t> P(DD);
        for (int i=0; i<8; ++i)
            for (int j=0; j<8; ++j) {
                bool any = false;
                for (int k=0; k<8; ++k) if (T_Z_REF[i*64+j*8+k] != 0) any = true;
                if (!any) continue;
                gemm_i64(&Xz[i*DD], &Yz[j*DD], P.data());
                for (int k=0; k<8; ++k) {
                    int s = T_Z_REF[i*64+j*8+k];
                    if (s == 1) for (int q=0; q<DD; ++q) OutRef[k*DD+q] += P[q];
                    else if (s == -1) for (int q=0; q<DD; ++q) OutRef[k*DD+q] -= P[q];
                }
            }
    };
    static const int UZ23[23][8] = { {1,0,0,0,0,0,0,0},{1,0,0,-1,0,0,0,1},{0,1,0,0,0,0,1,0},{0,0,0,1,-1,1,1,1},{1,1,0,0,-1,1,1,0},{0,0,0,0,0,1,0,0},{0,0,0,1,0,0,0,0},{0,0,1,0,-1,0,0,0},{0,0,0,1,0,0,0,1},{0,1,0,0,0,0,0,0},{1,0,0,-1,0,0,0,0},{0,1,0,0,-1,0,1,0},{1,0,0,-1,0,1,0,0},{0,0,1,-1,0,0,0,0},{0,0,0,0,0,0,0,1},{0,0,0,0,0,1,1,1},{1,1,0,0,-1,1,0,0},{1,0,-1,0,1,0,0,0},{0,0,1,0,0,0,0,0},{0,1,-1,0,0,0,1,-1},{0,1,1,0,-1,1,0,-1},{0,0,0,0,0,0,1,0},{0,1,0,0,0,0,0,0} };
    static const int VZ23[23][8] = { {0,0,0,0,1,0,1,0},{1,0,0,0,0,-1,0,0},{0,1,0,0,0,0,0,-1},{0,1,0,1,0,0,-1,0},{0,0,0,0,0,1,-1,1},{1,1,-1,1,0,0,-1,0},{1,-1,0,-1,1,-1,0,-1},{0,1,1,0,0,0,-1,0},{0,1,0,1,-1,0,-1,0},{0,0,0,0,0,1,1,-1},{1,0,0,-1,1,-1,1,1},{0,1,0,0,0,0,-1,0},{0,0,0,0,0,0,0,1},{0,1,1,1,-1,0,-1,0},{1,1,0,1,-1,0,-1,0},{1,1,0,1,0,0,-1,0},{0,0,0,0,0,1,0,1},{0,0,1,0,0,0,0,0},{0,1,1,0,-1,0,-1,0},{0,0,0,0,0,0,1,0},{0,0,0,0,0,1,0,0},{1,0,0,0,0,0,-1,1},{0,0,0,0,0,0,0,1} };
    static const int WZ23[8][23] = { {0,1,0,0,0,0,1,0,1,0,0,0,-1,0,-1,0,1,0,0,0,-1,0,-1},{0,0,1,0,0,1,0,0,0,0,0,0,0,0,1,-1,0,0,0,0,0,1,1},{0,0,1,0,0,0,0,1,0,0,0,-1,0,0,0,0,0,1,0,-1,0,0,1},{1,1,0,0,0,0,0,0,1,0,-1,0,1,0,-1,0,0,0,0,0,0,0,0},{1,0,1,0,1,0,0,0,0,0,0,-1,0,0,0,0,-1,0,0,0,0,0,1},{0,0,-1,-1,0,0,0,0,1,1,0,1,0,0,-1,1,0,0,0,0,0,-1,0},{0,0,1,0,0,0,0,1,0,0,0,-1,0,0,0,0,0,0,-1,0,0,1,1},{0,0,0,0,0,0,0,0,-1,0,0,0,0,-1,1,0,0,0,1,0,0,0,1} };
    auto bench_block_zorn_r23 = [&]() {
        std::memset(OutTest.data(), 0, 8*DD*sizeof(int64_t));
        std::vector<int64_t> A(DD), B(DD), M(23*DD);
        for (int r=0; r<23; ++r) {
            std::memset(A.data(), 0, DD*sizeof(int64_t));
            std::memset(B.data(), 0, DD*sizeof(int64_t));
            for (int i=0; i<8; ++i) {
                if (UZ23[r][i]==1) for (int q=0; q<DD; ++q) A[q] += Xz[i*DD+q];
                else if (UZ23[r][i]==-1) for (int q=0; q<DD; ++q) A[q] -= Xz[i*DD+q];
                if (VZ23[r][i]==1) for (int q=0; q<DD; ++q) B[q] += Yz[i*DD+q];
                else if (VZ23[r][i]==-1) for (int q=0; q<DD; ++q) B[q] -= Yz[i*DD+q];
            }
            gemm_i64(A.data(), B.data(), &M[r*DD]);
        }
        for (int k=0; k<8; ++k)
            for (int r=0; r<23; ++r) {
                if (WZ23[k][r]==1) for (int q=0; q<DD; ++q) OutTest[k*DD+q] += M[r*DD+q];
                else if (WZ23[k][r]==-1) for (int q=0; q<DD; ++q) OutTest[k*DD+q] -= M[r*DD+q];
            }
    };
    bench_block_zorn_naive32();
    bench_block_zorn_r23();
    int blk_zorn_err = 0;
    for (int idx=0; idx<8*DD; ++idx) if (OutRef[idx] != OutTest[idx]) ++blk_zorn_err;

    auto time_fn = [](auto fn, int reps) {
        fn();
        double best_us = 1e18;
        for (int r=0; r<reps; ++r) {
            auto s0 = std::chrono::steady_clock::now();
            fn();
            auto s1 = std::chrono::steady_clock::now();
            double us = std::chrono::duration<double, std::micro>(s1 - s0).count();
            if (us < best_us) best_us = us;
        }
        return best_us;
    };
    double us_zorn32 = time_fn(bench_block_zorn_naive32, 25);
    double us_zorn23 = time_fn(bench_block_zorn_r23, 25);

    // Also D=32 Gaussian-Octonion Block-GEMM: Naive 64 (192 real GEMMs) vs Cariow 30 (90 real GEMMs) vs Butterfly-R22 (66 real GEMMs)!
    static const int UZ22[22][8] = { {1,-1,-1,1,0,-4,-2,1},{0,0,1,-1,2,2,2,0},{0,0,-1,1,0,0,0,0},{0,1,0,1,0,0,0,-1},{-1,0,-1,1,0,0,0,0},{-2,0,-1,1,-2,0,0,0},{0,0,0,0,0,-2,0,1},{-1,0,0,0,0,0,0,0},{-2,0,1,-1,0,2,2,0},{0,-1,0,-1,0,0,0,1},{0,-2,-1,-2,-2,0,-2,2},{1,0,1,-1,0,0,0,1},{0,0,-1,1,0,0,0,-1},{0,0,-1,1,0,0,0,0},{0,1,1,0,2,2,2,-1},{0,0,0,-1,0,0,0,1},{0,0,0,-1,0,0,0,0},{0,1,0,0,0,0,0,0},{-2,1,2,-2,2,2,2,0},{1,1,2,-1,2,0,2,1},{-1,0,1,-1,0,2,2,0},{-1,0,1,-1,0,2,2,1} };
    static const int VZ22[22][8] = { {0,0,0,0,0,0,0,1},{1,2,0,1,0,-2,-2,1},{0,-4,3,-6,2,2,0,-3},{1,0,0,0,0,0,-1,1},{0,1,-2,3,-2,0,1,1},{1,0,1,0,0,0,0,0},{0,0,1,-1,0,0,0,-1},{-1,1,1,-2,2,-2,-3,0},{0,0,0,1,0,0,0,0},{1,-1,0,0,0,0,-1,1},{1,0,0,0,0,0,0,1},{0,-2,2,-2,4,0,-2,-1},{0,-2,1,-1,2,2,0,-1},{0,1,-1,3,0,0,0,1},{0,1,0,0,0,-1,-1,0},{1,0,-1,1,-2,-2,0,1},{1,2,0,0,-2,-2,0,1},{0,-2,0,0,0,0,0,-1},{0,-2,0,0,0,2,2,-1},{0,0,0,0,0,1,1,0},{0,1,0,-1,0,-1,0,1},{0,-2,0,0,0,2,0,-1} };
    static const int WZ22[8][22] = { {0,0,0,-2,0,0,0,2,2,-2,0,1,2,0,2,0,0,-1,1,0,2,1},{1,0,-1,0,0,0,1,0,1,0,0,0,1,-2,0,0,0,-1,0,0,2,1},{-1,0,2,2,2,0,0,0,-4,2,0,-1,-2,2,0,0,0,1,0,0,-4,-2},{-1,0,0,2,0,0,0,0,-2,2,0,0,0,0,0,0,0,1,0,0,-2,-1},{0,0,-1,-1,-1,0,0,0,1,-1,0,1,2,-1,1,0,0,0,0,1,1,1},{1,1,0,2,0,0,0,0,1,0,1,0,0,0,-2,0,-1,-1,0,0,2,1},{-1,0,1,-2,2,1,0,2,-2,0,-1,0,0,0,2,0,1,0,1,0,-2,-1},{0,0,2,-4,0,0,0,0,0,-4,0,0,-2,4,0,2,-2,-2,0,0,0,0} };
    static const int UC30[30][8] = { {1,1,0,-2,0,0,0,2},{0,0,1,0,-1,-1,-1,0},{1,-1,0,-1,0,0,0,-1},{0,0,1,0,1,0,0,0},{0,1,0,0,1,0,0,0},{0,1,0,0,-1,0,0,0},{1,0,1,1,0,-1,0,0},{1,0,-1,0,0,0,0,0},{0,0,0,1,0,0,0,-1},{0,0,0,0,0,1,1,0},{0,0,0,1,0,0,0,1},{0,0,0,0,0,1,-1,0},{0,0,0,1,0,0,0,0},{0,0,0,1,0,0,0,1},{0,0,0,1,0,-1,0,0},{1,-1,-1,0,1,1,-1,-2},{1,-1,0,0,0,0,0,0},{0,0,1,0,-1,0,0,0},{1,0,0,-1,0,0,0,0},{0,0,1,0,1,0,0,0},{0,1,0,0,1,0,-1,1},{0,1,0,0,-1,0,1,1},{1,0,-1,-1,0,-1,0,0},{1,0,1,-1,0,1,0,0},{0,0,1,1,1,0,0,1},{1,-1,0,-2,0,0,0,0},{0,0,0,0,0,1,-1,0},{0,1,0,0,1,0,1,-1},{0,1,0,0,-1,0,-1,-1},{1,0,-1,1,0,1,0,0} };
    static const int VC30[30][8] = { {1,1,0,0,0,1,-1,0},{0,0,1,0,0,0,0,0},{0,0,1,0,1,0,0,0},{1,-1,0,-1,0,0,0,-1},{0,1,1,0,0,1,0,1},{0,1,-1,0,0,-1,0,1},{1,0,0,0,-1,0,0,0},{1,0,0,1,1,0,-1,0},{2,2,-1,1,-1,2,-2,1},{0,0,1,0,1,1,1,0},{0,2,2,0,2,1,-1,0},{2,2,0,-1,0,0,0,1},{1,-1,-1,0,-1,0,0,0},{0,2,1,0,1,0,0,0},{1,0,0,-1,-1,0,-1,0},{0,0,1,0,1,0,0,0},{0,0,0,1,0,0,0,-1},{0,0,1,0,1,-1,-1,0},{0,0,0,0,0,1,-1,0},{0,0,0,1,0,0,0,1},{0,0,0,0,0,1,0,1},{0,0,0,0,0,1,0,-1},{0,0,0,1,0,0,-1,0},{0,0,0,1,0,0,1,0},{1,-1,0,0,0,0,0,0},{0,0,1,0,1,-1,1,0},{1,1,0,0,0,0,0,0},{0,1,1,0,0,0,0,0},{0,1,-1,0,0,0,0,0},{1,0,0,0,1,0,0,0} };
    static const int WC30[8][30] = { {0,-4,2,0,1,1,1,-1,0,-2,0,0,-4,-2,1,-2,0,0,0,0,-1,1,1,0,0,0,0,-2,-2,2},{2,0,2,0,-1,-1,-1,-1,2,0,0,0,0,0,1,0,0,0,-4,0,1,-1,1,0,0,-2,0,0,0,0},{0,0,2,0,1,-1,1,-1,0,0,2,0,0,-2,-1,0,0,0,0,0,-1,-1,1,0,0,0,0,0,0,0},{0,0,-2,0,-1,-1,0,1,0,0,0,0,4,2,-1,2,0,2,0,0,0,0,0,1,0,0,0,1,1,-1},{0,0,0,2,1,-1,-1,1,0,0,0,2,0,0,1,0,0,0,0,2,-1,-1,-1,0,0,0,-4,0,0,0},{0,0,-2,0,-1,1,0,-1,0,0,0,0,0,0,-1,0,0,0,4,0,0,0,0,1,0,2,0,1,-1,1},{0,0,0,0,1,-1,0,-1,0,0,0,0,0,0,-1,0,0,0,0,-2,0,0,0,1,0,0,-2,-1,1,1},{0,0,0,-2,1,1,0,1,0,0,0,0,0,0,-1,0,-2,0,0,-2,0,0,0,1,2,0,0,-1,-1,-1} };
    std::vector<int64_t> Xr(8*DD), Xi(8*DD), Yr(8*DD), Yi(8*DD), OrRef(8*DD), OiRef(8*DD), Or22(8*DD), Oi22(8*DD), Or30(8*DD), Oi30(8*DD);
    for (int idx=0; idx<8*DD; ++idx) {
        Xr[idx]=(int64_t)(brng()%32)-16; Xi[idx]=(int64_t)(brng()%32)-16;
        Yr[idx]=(int64_t)(brng()%32)-16; Yi[idx]=(int64_t)(brng()%32)-16;
    }
    auto cgemm_3mul = [&](const int64_t* Ar, const int64_t* Ai, const int64_t* Br, const int64_t* Bi, int64_t* Cr, int64_t* Ci) {
        int64_t K1[DD], K2[DD], K3[DD], S1[DD], S2[DD];
        for (int q=0; q<DD; ++q) { S1[q] = Ar[q] + Ai[q]; S2[q] = Br[q] + Bi[q]; }
        gemm_i64(Ar, Br, K1);
        gemm_i64(Ai, Bi, K2);
        gemm_i64(S1, S2, K3);
        for (int q=0; q<DD; ++q) { Cr[q] = K1[q] - K2[q]; Ci[q] = K3[q] - K1[q] - K2[q]; }
    };
    auto blk_qi_naive64 = [&]() {
        std::memset(OrRef.data(), 0, 8*DD*sizeof(int64_t));
        std::memset(OiRef.data(), 0, 8*DD*sizeof(int64_t));
        int64_t Pr[DD], Pi[DD];
        for (int i=0; i<8; ++i)
            for (int j=0; j<8; ++j) {
                cgemm_3mul(&Xr[i*DD], &Xi[i*DD], &Yr[j*DD], &Yi[j*DD], Pr, Pi);
                for (int k=0; k<8; ++k) {
                    int s = T_O_REF[i*64+j*8+k];
                    if (s == 1) for (int q=0; q<DD; ++q) { OrRef[k*DD+q] += Pr[q]; OiRef[k*DD+q] += Pi[q]; }
                    else if (s == -1) for (int q=0; q<DD; ++q) { OrRef[k*DD+q] -= Pr[q]; OiRef[k*DD+q] -= Pi[q]; }
                }
            }
    };
    auto blk_qi_cariow30 = [&]() {
        std::memset(Or30.data(), 0, 8*DD*sizeof(int64_t));
        std::memset(Oi30.data(), 0, 8*DD*sizeof(int64_t));
        std::vector<int64_t> Ar(DD), Ai(DD), Br(DD), Bi(DD), Mr(30*DD), Mi(30*DD);
        for (int r=0; r<30; ++r) {
            std::memset(Ar.data(),0,DD*8); std::memset(Ai.data(),0,DD*8);
            std::memset(Br.data(),0,DD*8); std::memset(Bi.data(),0,DD*8);
            for (int i=0; i<8; ++i) {
                int cu = UC30[r][i], cv = VC30[r][i];
                if (cu) for (int q=0; q<DD; ++q) { Ar[q] += cu*Xr[i*DD+q]; Ai[q] += cu*Xi[i*DD+q]; }
                if (cv) for (int q=0; q<DD; ++q) { Br[q] += cv*Yr[i*DD+q]; Bi[q] += cv*Yi[i*DD+q]; }
            }
            cgemm_3mul(Ar.data(), Ai.data(), Br.data(), Bi.data(), &Mr[r*DD], &Mi[r*DD]);
        }
        for (int k=0; k<8; ++k) {
            for (int r=0; r<30; ++r) {
                int cw = WC30[k][r];
                if (cw) for (int q=0; q<DD; ++q) { Or30[k*DD+q] += cw*Mr[r*DD+q]; Oi30[k*DD+q] += cw*Mi[r*DD+q]; }
            }
            for (int q=0; q<DD; ++q) { Or30[k*DD+q] >>= 1; Oi30[k*DD+q] >>= 1; }
        }
    };
    auto blk_qi_bfly22 = [&]() {
        int64_t Zxr[8*DD], Zxi[8*DD], Zyr[8*DD], Zyi[8*DD];
        const int prs[4][4] = { {0,7,0,1}, {4,5,2,5}, {2,6,3,6}, {1,3,4,7} };
        for (auto &pr : prs) {
            int p=pr[0], g=pr[1], a=pr[2], b=pr[3];
            for (int q=0; q<DD; ++q) {
                Zxr[a*DD+q] = Xr[p*DD+q] - Xi[g*DD+q]; Zxi[a*DD+q] = Xi[p*DD+q] + Xr[g*DD+q];
                Zxr[b*DD+q] = Xr[p*DD+q] + Xi[g*DD+q]; Zxi[b*DD+q] = Xi[p*DD+q] - Xr[g*DD+q];
                Zyr[a*DD+q] = Yr[p*DD+q] - Yi[g*DD+q]; Zyi[a*DD+q] = Yi[p*DD+q] + Yr[g*DD+q];
                Zyr[b*DD+q] = Yr[p*DD+q] + Yi[g*DD+q]; Zyi[b*DD+q] = Yi[p*DD+q] - Yr[g*DD+q];
            }
        }
        std::vector<int64_t> Ar(DD), Ai(DD), Br(DD), Bi(DD), Mr(22*DD), Mi(22*DD), Wr(8*DD,0), Wi(8*DD,0);
        for (int r=0; r<22; ++r) {
            std::memset(Ar.data(),0,DD*8); std::memset(Ai.data(),0,DD*8);
            std::memset(Br.data(),0,DD*8); std::memset(Bi.data(),0,DD*8);
            for (int i=0; i<8; ++i) {
                int cu = UZ22[r][i], cv = VZ22[r][i];
                if (cu) for (int q=0; q<DD; ++q) { Ar[q] += cu*Zxr[i*DD+q]; Ai[q] += cu*Zxi[i*DD+q]; }
                if (cv) for (int q=0; q<DD; ++q) { Br[q] += cv*Zyr[i*DD+q]; Bi[q] += cv*Zyi[i*DD+q]; }
            }
            cgemm_3mul(Ar.data(), Ai.data(), Br.data(), Bi.data(), &Mr[r*DD], &Mi[r*DD]);
        }
        for (int k=0; k<8; ++k)
            for (int r=0; r<22; ++r) {
                int cw = WZ22[k][r];
                if (cw) for (int q=0; q<DD; ++q) { Wr[k*DD+q] += cw*Mr[r*DD+q]; Wi[k*DD+q] += cw*Mi[r*DD+q]; }
            }
        for (auto &pr : prs) {
            int p=pr[0], g=pr[1], a=pr[2], b=pr[3];
            for (int q=0; q<DD; ++q) {
                Or22[p*DD+q] = (Wr[a*DD+q] + Wr[b*DD+q]) >> 2;
                Oi22[p*DD+q] = (Wi[a*DD+q] + Wi[b*DD+q]) >> 2;
                Or22[g*DD+q] = (Wi[a*DD+q] - Wi[b*DD+q]) >> 2;
                Oi22[g*DD+q] = (Wr[b*DD+q] - Wr[a*DD+q]) >> 2;
            }
        }
    };
    blk_qi_naive64(); blk_qi_cariow30(); blk_qi_bfly22();
    int blk_qi22_err = 0, blk_qi30_err = 0;
    for (int idx=0; idx<8*DD; ++idx) {
        if (OrRef[idx] != Or22[idx] || OiRef[idx] != Oi22[idx]) ++blk_qi22_err;
        if (OrRef[idx] != Or30[idx] || OiRef[idx] != Oi30[idx]) ++blk_qi30_err;
    }
    double us_qi64 = time_fn(blk_qi_naive64, 15);
    double us_qi30 = time_fn(blk_qi_cariow30, 15);
    double us_qi22 = time_fn(blk_qi_bfly22, 15);
    std::cout << "  \"_block_gemm_D32\": {"
              << "\"zorn_r23_err\":" << blk_zorn_err
              << ",\"zorn_sparse32_us\":" << std::fixed << std::setprecision(2) << us_zorn32
              << ",\"zorn_r23_us\":" << std::fixed << std::setprecision(2) << us_zorn23
              << ",\"zorn_r23_speedup_vs_32\": " << std::fixed << std::setprecision(3) << (us_zorn32 / us_zorn23)
              << ",\"qi_r22_err\":" << blk_qi22_err << ",\"qi_c30_err\":" << blk_qi30_err
              << ",\"qi_naive64_us\":" << std::fixed << std::setprecision(2) << us_qi64
              << ",\"qi_cariow30_us\":" << std::fixed << std::setprecision(2) << us_qi30
              << ",\"qi_bfly_r22_us\":" << std::fixed << std::setprecision(2) << us_qi22
              << ",\"qi_r22_speedup_vs_naive64\":" << std::fixed << std::setprecision(3) << (us_qi64 / us_qi22)
              << ",\"qi_r22_speedup_vs_cariow30\":" << std::fixed << std::setprecision(3) << (us_qi30 / us_qi22)
              << "}\n";

    std::cout << "}\n";
    if (sink == 123456789LL) std::cerr << sink;
    return 0;
}
