// tests/core/expert_worker_parity_test.cpp - Numerical parity test for strata-expert-worker.
//
// Connects to a running `strata-expert-worker` (e.g. on astor RX 6800 XT HIP device 0) and compares its
// real expert outputs against local reference expert execution (CPU ggml-cpu native expert evaluation +
// weighted partial-sum reduction) across single-expert, multi-expert, duplicate-token, and partial-sum cases
// using actual expert weights from the loaded GGUF model.

#include "strata/artifact/gguf_reader.hpp"
#include "strata/core/expert_worker_identity.hpp"
#include "strata/core/family_layout.hpp"
#include "strata/core/network_experts.hpp"
#include "strata/kernels/cpu/native_expert.hpp"
#include "strata/kernels/iq_kernels.hpp"

#include <cuda_runtime.h>

#include <cmath>
#include <algorithm>
#include <set>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <memory>
#include <random>
#include <string>
#include <vector>

namespace {

using namespace strata::core;
namespace K = strata::kernels;
namespace C = strata::kernels::cpu;

struct ErrorMetrics {
    double max_abs_err = 0.0;
    double rel_l2_err = 0.0;
    double cosine_sim = 1.0;
    int nan_inf_cnt = 0;
};

ErrorMetrics compare_vectors(const float *ref, const float *got, size_t n) {
    ErrorMetrics m{};
    double diff_sq = 0.0, ref_sq = 0.0, got_sq = 0.0, dot = 0.0;
    for (size_t i = 0; i < n; ++i) {
        if (!std::isfinite(ref[i]) || !std::isfinite(got[i])) {
            ++m.nan_inf_cnt;
            continue;
        }
        const double r = ref[i];
        const double g = got[i];
        const double d = std::fabs(r - g);
        if (d > m.max_abs_err)
            m.max_abs_err = d;
        diff_sq += d * d;
        ref_sq += r * r;
        got_sq += g * g;
        dot += r * g;
    }
    m.rel_l2_err = (ref_sq > 1e-20) ? std::sqrt(diff_sq / ref_sq) : std::sqrt(diff_sq);
    m.cosine_sim = (ref_sq > 1e-20 && got_sq > 1e-20) ? (dot / (std::sqrt(ref_sq) * std::sqrt(got_sq))) : 1.0;
    return m;
}

} // namespace

int main(int argc, char **argv) {
    std::string model_path;
    std::string worker_ep;
    double tol = 1e-3;
    int64_t worker_vram_mib = -1;
    bool all_experts = false;

    for (int i = 1; i < argc; ++i) {
        const std::string a = argv[i];
        auto next = [&]() -> std::string {
            if (i + 1 >= argc) {
                std::fprintf(stderr, "missing value for %s\n", a.c_str());
                std::exit(2);
            }
            return argv[++i];
        };
        if (a == "--model")
            model_path = next();
        else if (a == "--worker")
            worker_ep = next();
        else if (a == "--tol")
            tol = std::atof(next().c_str());
        else if (a == "--worker-vram-mib") worker_vram_mib = std::stoll(next());
        else if (a == "--all-experts") all_experts = true;
        else {
            std::fprintf(stderr, "usage: expert_worker_parity_test --model GGUF --worker IP:PORT [--tol 1e-3] [--all-experts] [--worker-vram-mib N]\n");
            return 2;
        }
    }

    if (model_path.empty() || worker_ep.empty()) {
        std::fprintf(stderr, "usage: expert_worker_parity_test --model GGUF --worker IP:PORT [--tol 1e-3] [--all-experts] [--worker-vram-mib N]\n");
        return 2;
    }

    auto gm_ptr = std::make_shared<strata::GgufModel>(strata::GgufModel::open(model_path));
    const strata::GgufModel &gm = *gm_ptr;
    ModelGeometry g{};
    int64_t top_k = 8;
    std::string err;
    if (!geometry_from_gguf(gm.meta(), g, top_k, err)) {
        std::fprintf(stderr, "geometry_from_gguf failed: %s\n", err.c_str());
        return 1;
    }

    const int32_t total_layers = static_cast<int32_t>(g.n_layers);
    const uint32_t N = static_cast<uint32_t>(g.n_embd);
    const uint32_t FF = static_cast<uint32_t>(g.n_ff);

    NetworkExpertExecutor net;
    if (!net.connect_and_handshake(worker_ep, static_cast<uint32_t>(g.n_layers), static_cast<uint32_t>(total_layers),
                                   static_cast<uint32_t>(g.n_expert), N, FF, static_cast<uint32_t>(top_k),
                                   expert_worker_model_sha256(gm), 30000, err)) {
        std::fprintf(stderr, "connect_and_handshake failed: %s\n", err.c_str());
        return 1;
    }

    // Select test layers covering distinct quantization formats (e.g. layers 0, 1, 19, 39) and experts 0..7
    std::vector<int32_t> test_layers = {0, 1, static_cast<int32_t>(g.n_layers / 2),
                                        static_cast<int32_t>(g.n_layers - 1)};
    std::vector<ExpertWorkerPair> plan;
    std::vector<int32_t> plan_layers = test_layers;
    if (all_experts) { plan_layers.clear(); for (int32_t l = 0; l < g.n_layers; ++l) plan_layers.push_back(l); }
    for (int32_t l : plan_layers) {
        for (int32_t e = 0; e < (all_experts ? g.n_expert : 8); ++e) {
            plan.push_back({l, e});
        }
    }
    // Independent execution oracle: static budget consumes each layer's padded
    // expert size in plan order; CPU weights are evaluated from packed reference
    // blobs, independently of the worker's RAM/mapped layout overrides.
    std::set<std::pair<int32_t, int32_t>> gpu_pairs;
    if (worker_vram_mib >= 0) {
        size_t remaining = static_cast<size_t>(worker_vram_mib) << 20;
        for (const auto p : plan) {
            const auto *tg = gm.find("blk." + std::to_string(p.layer) + ".ffn_gate_exps.weight");
            const auto *td = gm.find("blk." + std::to_string(p.layer) + ".ffn_down_exps.weight");
            if (!tg || !td) return 1;
            const auto layout = K::native_expert_layout(tg->type, td->type, N, FF);
            const auto bytes = (layout.bytes + 255u) & ~size_t(255u);
            if (bytes <= remaining) { gpu_pairs.emplace(p.layer, p.expert); remaining -= bytes; }
        }
        std::printf("Tiered oracle: %zu GPU experts; remaining owned experts use CPU reference\n", gpu_pairs.size());
    }

    if (!net.install_plan(plan, err)) {
        std::fprintf(stderr, "install_plan failed: %s\n", err.c_str());
        return 1;
    }
    std::printf("Connected to worker '%s' (backend=%u), installed %u experts (%.2f MiB VRAM)\n", net.caps().device_name,
                net.caps().backend_id, net.ready_info().loaded_count,
                static_cast<double>(net.ready_info().vram_used_bytes) / (1024.0 * 1024.0));

    // Helper to evaluate a single (layer, expert) on local CPU reference (ggml-cpu Q8_K native expert kernel)
    // and local GPU reference (native_expert_grouped Q8_1 kernel)
    uint8_t *d_blob = nullptr, *d_q8 = nullptr, *d_scr = nullptr;
    float *d_x = nullptr, *d_out = nullptr;
    unsigned long long *d_ptr = nullptr;
    int32_t *d_start = nullptr, *d_cnt = nullptr, *d_dst = nullptr, *d_tok = nullptr;
    const size_t max_blob = 4ULL << 20;
    cudaMalloc(reinterpret_cast<void **>(&d_blob), max_blob);
    cudaMalloc(reinterpret_cast<void **>(&d_x), N * sizeof(float));
    cudaMalloc(reinterpret_cast<void **>(&d_q8), (N / 32) * 36);
    cudaMalloc(reinterpret_cast<void **>(&d_scr), K::native_expert_scratch_bytes(1, FF));
    cudaMalloc(reinterpret_cast<void **>(&d_out), N * sizeof(float));
    cudaMalloc(reinterpret_cast<void **>(&d_ptr), sizeof(unsigned long long));
    cudaMalloc(reinterpret_cast<void **>(&d_start), 2 * sizeof(int32_t));
    cudaMalloc(reinterpret_cast<void **>(&d_cnt), sizeof(int32_t));
    cudaMalloc(reinterpret_cast<void **>(&d_dst), sizeof(int32_t));
    cudaMalloc(reinterpret_cast<void **>(&d_tok), sizeof(int32_t));

    auto eval_refs = [&](int32_t layer, int32_t expert, const float *act, float *out_gpu, float *out_cpu) -> bool {
        auto tname = [&](const char *s) { return "blk." + std::to_string(layer) + "." + s; };
        size_t sg = 0, su = 0, sd = 0;
        const strata::TensorInfo *tg = gm.find(tname("ffn_gate_exps.weight"), &sg);
        const strata::TensorInfo *tu = gm.find(tname("ffn_up_exps.weight"), &su);
        const strata::TensorInfo *td = gm.find(tname("ffn_down_exps.weight"), &sd);
        if (!tg || !tu || !td)
            return false;

        auto L = K::native_expert_layout(static_cast<int>(tg->type), static_cast<int>(td->type), g.n_embd, g.n_ff);
        const size_t gsz = static_cast<size_t>(g.n_ff) * L.gu_row;
        const size_t dsz = static_cast<size_t>(g.n_embd) * L.d_row;
        std::vector<uint8_t> blob(L.bytes);
        std::memcpy(blob.data(), gm.shard(sg).tensor_data(*tg) + static_cast<size_t>(expert) * gsz, gsz);
        std::memcpy(blob.data() + L.up_off, gm.shard(su).tensor_data(*tu) + static_cast<size_t>(expert) * gsz, gsz);
        std::memcpy(blob.data() + L.down_off, gm.shard(sd).tensor_data(*td) + static_cast<size_t>(expert) * dsz, dsz);

        if (out_cpu) {
            C::NativeFmt fmt{};
            std::string ferr;
            if (!C::native_fmt(L.gu_type, L.d_type, g.n_embd, g.n_ff, fmt, ferr))
                return false;

            std::vector<uint8_t> qact(C::kNativeActBytes);
            std::vector<float> ff(static_cast<size_t>(g.n_ff));
            std::vector<uint8_t> hq(C::kNativeHBytes);
            C::native_quant_act(fmt, act, qact.data());
            const void *act_ptrs[1] = {qact.data()};
            float *ff_ptrs[1] = {ff.data()};
            C::native_gu_rows(fmt, blob.data(), act_ptrs, 1, ff_ptrs, 0, static_cast<int>(g.n_ff));
            C::native_quant_h(fmt, ff.data(), hq.data());
            const void *hq_ptrs[1] = {hq.data()};
            float *out_ptrs[1] = {out_cpu};
            C::native_down_rows(fmt, blob.data(), hq_ptrs, 1, out_ptrs, 0, static_cast<int>(g.n_embd));
        }

        if (out_gpu) {
            const unsigned long long h_ptr = reinterpret_cast<unsigned long long>(d_blob);
            const int32_t h_start[2] = {0, 1};
            const int32_t h_cnt = 1, h_dst = 0, h_tok = 0;
            cudaMemcpy(d_blob, blob.data(), L.bytes, cudaMemcpyHostToDevice);
            cudaMemcpy(d_x, act, N * sizeof(float), cudaMemcpyHostToDevice);
            cudaMemcpy(d_ptr, &h_ptr, sizeof(h_ptr), cudaMemcpyHostToDevice);
            cudaMemcpy(d_start, h_start, sizeof(h_start), cudaMemcpyHostToDevice);
            cudaMemcpy(d_cnt, &h_cnt, sizeof(h_cnt), cudaMemcpyHostToDevice);
            cudaMemcpy(d_dst, &h_dst, sizeof(h_dst), cudaMemcpyHostToDevice);
            cudaMemcpy(d_tok, &h_tok, sizeof(h_tok), cudaMemcpyHostToDevice);
            K::quantize_q8_1_rows(d_x, 1, N, d_q8, nullptr);
            K::native_expert_grouped(L, d_ptr, d_start, d_cnt, d_dst, d_tok, 1, 1, d_q8, d_scr, d_out, nullptr);
            cudaDeviceSynchronize();
            cudaMemcpy(out_gpu, d_out, N * sizeof(float), cudaMemcpyDeviceToHost);
        }
        if (worker_vram_mib >= 0 && !gpu_pairs.count({layer, expert}) && out_gpu && out_cpu)
            std::copy(out_cpu, out_cpu + N, out_gpu);
        return true;
    };

    std::mt19937 rng(42);
    std::normal_distribution<float> dist(0.0f, 0.5f);

    int failures = 0;

    // Case 1: Single expert parity across all test layers
    for (int32_t l : test_layers) {
        std::vector<float> act(N);
        for (uint32_t i = 0; i < N; ++i)
            act[i] = dist(rng);

        std::vector<ExpertWorkerAssignment> assigns = {{0, 0, 3, 1.0f}};
        std::vector<float> got(N, 0.0f), ref_gpu(N, 0.0f), ref_cpu(N, 0.0f);
        uint8_t mask = 0;
        if (!net.execute_sync(l, 1, act.data(), assigns, got.data(), &mask, &err) || mask != 1) {
            std::fprintf(stderr, "FAIL: layer %d single-expert execute_sync failed: %s\n", l, err.c_str());
            return 1;
        }
        eval_refs(l, 3, act.data(), ref_gpu.data(), ref_cpu.data());
        const auto mg = compare_vectors(ref_gpu.data(), got.data(), N);
        const auto mc = compare_vectors(ref_cpu.data(), got.data(), N);
        const bool pass = (mg.nan_inf_cnt == 0 && mg.rel_l2_err <= tol && mg.cosine_sim >= 0.99999 &&
                           mc.nan_inf_cnt == 0 && mc.max_abs_err <= 2e-3 && mc.cosine_sim >= 0.9995);
        std::printf("CASE single_expert layer=%2d expert=3 vs_execution(max_abs=%.2e,rel_l2=%.2e,cos=%.7f) "
                    "vs_cpu(max_abs=%.2e,rel_l2=%.2e,cos=%.7f) nan=%d -> %s\n",
                    l, mg.max_abs_err, mg.rel_l2_err, mg.cosine_sim, mc.max_abs_err, mc.rel_l2_err, mc.cosine_sim,
                    mg.nan_inf_cnt, pass ? "PASS" : "FAIL");
        if (!pass)
            ++failures;
    }

    // Case 2: Multi-expert weighted partial sum for 1 token (4 experts with router weights)
    for (int32_t l : test_layers) {
        std::vector<float> act(N);
        for (uint32_t i = 0; i < N; ++i)
            act[i] = dist(rng);

        std::vector<ExpertWorkerAssignment> assigns = {
            {0, 0, 0, 0.32f},
            {0, 2, 2, 0.21f},
            {0, 5, 5, 0.14f},
            {0, 7, 7, 0.09f},
        };
        std::vector<float> got(N, 0.0f), ref_gpu(N, 0.0f), ref_cpu(N, 0.0f), tmp_g(N, 0.0f), tmp_c(N, 0.0f);
        uint8_t mask = 0;
        if (!net.execute_sync(l, 1, act.data(), assigns, got.data(), &mask, &err) || mask != 1) {
            std::fprintf(stderr, "FAIL: layer %d multi-expert execute_sync failed: %s\n", l, err.c_str());
            return 1;
        }
        for (const auto &a : assigns) {
            eval_refs(l, a.expert_id, act.data(), tmp_g.data(), tmp_c.data());
            for (uint32_t d = 0; d < N; ++d) {
                ref_gpu[d] = std::fma(a.weight, tmp_g[d], ref_gpu[d]);
                ref_cpu[d] = std::fma(a.weight, tmp_c[d], ref_cpu[d]);
            }
        }
        const auto mg = compare_vectors(ref_gpu.data(), got.data(), N);
        const auto mc = compare_vectors(ref_cpu.data(), got.data(), N);
        const bool pass = (mg.nan_inf_cnt == 0 && mg.rel_l2_err <= tol && mg.cosine_sim >= 0.99999 &&
                           mc.nan_inf_cnt == 0 && mc.max_abs_err <= 2e-3 && mc.cosine_sim >= 0.9995);
        std::printf("CASE multi_expert_sum layer=%2d n_exp=4 vs_execution(max_abs=%.2e,rel_l2=%.2e,cos=%.7f) "
                    "vs_cpu(max_abs=%.2e,rel_l2=%.2e,cos=%.7f) nan=%d -> %s\n",
                    l, mg.max_abs_err, mg.rel_l2_err, mg.cosine_sim, mc.max_abs_err, mc.rel_l2_err, mc.cosine_sim,
                    mg.nan_inf_cnt, pass ? "PASS" : "FAIL");
        if (!pass)
            ++failures;
    }

    // Case 3: Multi-token with shared/duplicate expert routing + sparse token (token 1 has 0 remote experts)
    {
        const int32_t l = test_layers.front();
        const uint32_t n_tok = 4;
        std::vector<float> acts(n_tok * N);
        for (float &v : acts)
            v = dist(rng);

        // Tokens 0, 2, 3 share expert 1 and 4; token 1 has NO remote experts
        std::vector<ExpertWorkerAssignment> assigns = {
            {0, 0, 1, 0.40f}, {0, 3, 4, 0.25f}, {2, 1, 1, 0.50f}, {3, 0, 4, 0.30f}, {3, 2, 6, 0.20f},
        };
        std::vector<float> got(n_tok * N, 0.0f), ref_gpu(n_tok * N, 0.0f), ref_cpu(n_tok * N, 0.0f), tmp_g(N, 0.0f),
            tmp_c(N, 0.0f);
        std::vector<uint8_t> mask(n_tok, 0);
        if (!net.execute_sync(l, n_tok, acts.data(), assigns, got.data(), mask.data(), &err)) {
            std::fprintf(stderr, "FAIL: multi_token_shared execute_sync failed: %s\n", err.c_str());
            return 1;
        }
        if (mask[0] != 1 || mask[1] != 0 || mask[2] != 1 || mask[3] != 1) {
            std::fprintf(stderr, "FAIL: multi_token_shared unexpected mask (%u,%u,%u,%u)\n", mask[0], mask[1], mask[2],
                         mask[3]);
            return 1;
        }
        for (const auto &a : assigns) {
            eval_refs(l, a.expert_id, acts.data() + static_cast<size_t>(a.token_idx) * N, tmp_g.data(), tmp_c.data());
            float *dst_g = ref_gpu.data() + static_cast<size_t>(a.token_idx) * N;
            float *dst_c = ref_cpu.data() + static_cast<size_t>(a.token_idx) * N;
            for (uint32_t d = 0; d < N; ++d) {
                dst_g[d] = std::fma(a.weight, tmp_g[d], dst_g[d]);
                dst_c[d] = std::fma(a.weight, tmp_c[d], dst_c[d]);
            }
        }
        for (uint32_t t : {0u, 2u, 3u}) {
            const auto mg = compare_vectors(ref_gpu.data() + t * N, got.data() + t * N, N);
            const auto mc = compare_vectors(ref_cpu.data() + t * N, got.data() + t * N, N);
            const bool pass = (mg.nan_inf_cnt == 0 && mg.rel_l2_err <= tol && mg.cosine_sim >= 0.99999 &&
                               mc.nan_inf_cnt == 0 && mc.max_abs_err <= 2e-3 && mc.cosine_sim >= 0.9995);
            std::printf("CASE multi_token_shared layer=%2d tok=%u vs_execution(max_abs=%.2e,rel_l2=%.2e,cos=%.7f) "
                        "vs_cpu(max_abs=%.2e,rel_l2=%.2e,cos=%.7f) nan=%d -> %s\n",
                        l, t, mg.max_abs_err, mg.rel_l2_err, mg.cosine_sim, mc.max_abs_err, mc.rel_l2_err,
                        mc.cosine_sim, mg.nan_inf_cnt, pass ? "PASS" : "FAIL");
            if (!pass)
                ++failures;
        }
    }

    cudaFree(d_blob);
    cudaFree(d_x);
    cudaFree(d_q8);
    cudaFree(d_scr);
    cudaFree(d_out);
    cudaFree(d_ptr);
    cudaFree(d_start);
    cudaFree(d_cnt);
    cudaFree(d_dst);
    cudaFree(d_tok);

    net.print_summary(stdout);
    if (failures == 0) {
        std::printf("expert_worker_parity_test: 0 failures (ALL PASS)\n");
        return 0;
    }
    std::fprintf(stderr, "expert_worker_parity_test: %d failures\n", failures);
    return 1;
}
