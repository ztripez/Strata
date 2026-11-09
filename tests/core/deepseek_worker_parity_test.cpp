// Real DeepSeek expert RPC against independent packed local GPU and CPU evaluations.
#include "strata/core/network_experts.hpp"
#include "strata/core/expert_worker_identity.hpp"
#include "strata/core/deepseek_worker_geometry.hpp"
#include "ds4_gguf.cuh"
#include "ds4_gguf_cpu.hpp"
#include <algorithm>
#include <cmath>
#include <cstdio>
#include <cstring>
#include <limits>
#include <random>
#include <set>
#include <stdexcept>
#include <vector>

namespace {
namespace G = strata::ds4::gguf;
namespace X = strata::dsv41k::exl3;
using namespace strata::core;
void ck(cudaError_t status) { if (status != cudaSuccess) throw std::runtime_error(cudaGetErrorString(status)); }
struct Memory {
    std::vector<void *> allocations;
    template<class T> T *get(size_t n) {
        T *p = nullptr; ck(cudaMalloc(reinterpret_cast<void **>(&p), n * sizeof(T)));
        allocations.push_back(p); return p;
    }
    // Cleanup after a test failure must not throw another exception.
    ~Memory() { for (auto p : allocations) (void)cudaFree(p); }
};
bool close(const std::vector<float> &ref, const std::vector<float> &got, bool cpu) {
    double diff = 0, norm = 0, dot = 0, gn = 0, max = 0;
    for (size_t i = 0; i < ref.size(); ++i) {
        if (!std::isfinite(ref[i]) || !std::isfinite(got[i])) return false;
        const double d = static_cast<double>(ref[i]) - got[i];
        diff += d*d; norm += static_cast<double>(ref[i])*ref[i]; gn += static_cast<double>(got[i])*got[i];
        dot += static_cast<double>(ref[i])*got[i]; max = std::max(max, std::abs(d));
    }
    const double rel = std::sqrt(diff / std::max(norm, 1e-30));
    const double cosine = norm > 1e-30 && gn > 1e-30 ? dot / std::sqrt(norm * gn) : 1;
    std::printf(" %s(abs=%.3g rel=%.3g cos=%.9f)", cpu ? "CPU" : "GPU", max, rel, cosine);
    return cpu ? max <= 2e-3 && cosine >= .9995 : rel <= 1e-3 && cosine >= .99999;
}
}

int main(int argc, char **argv) {
    if (argc != 3) { std::fprintf(stderr, "usage: deepseek_worker_parity_test FIRST_GGUF IPv4:port\n"); return 2; }
    try {
        auto model = strata::GgufModel::open(argv[1]);
        DeepSeekWorkerGeometry g(model);
        std::fprintf(stderr, "Hashing exact DeepSeek GGUF mappings before handshake\n");
        const auto sha = expert_worker_model_sha256(model);
        const auto arithmetic = G::iq_q8k_activations() ? ExpertWorkerArithmetic::DEEPSEEK_IQ_Q8K : ExpertWorkerArithmetic::DEEPSEEK_IQ_PER32;
        std::fprintf(stderr, "ds4-network-identity: role=parity-oracle effective_experts=%d engine=%s arithmetic=%u model_sha256=", g.picks, STRATA_VERSION, static_cast<uint32_t>(arithmetic));
        for (const auto b : sha) std::fprintf(stderr, "%02x", b);
        std::fprintf(stderr, "\n");
        NetworkExpertExecutor net;
        std::string error;
        {
            NetworkExpertExecutor mismatch;
            const auto other = arithmetic == ExpertWorkerArithmetic::DEEPSEEK_IQ_Q8K
                ? ExpertWorkerArithmetic::DEEPSEEK_IQ_PER32 : ExpertWorkerArithmetic::DEEPSEEK_IQ_Q8K;
            if (mismatch.connect_and_handshake(argv[2],g.layers,g.layers,g.experts,g.hidden,g.intermediate,g.picks,
                    sha,30000,error,other) || mismatch.has_any_owned())
                throw std::runtime_error("worker accepted a different activation arithmetic policy");
            std::puts("reject=arithmetic mismatch before READY PASS"); std::fflush(stdout);
            error.clear();
        }
        if (!net.connect_and_handshake(argv[2], g.layers, g.layers, g.experts, g.hidden, g.intermediate, g.picks,
                sha, 30000, error, arithmetic)) throw std::runtime_error(error);
        if (net.caps().max_tokens != 16) throw std::runtime_error("fixture requires the qualified 16-token worker limit");
        // Preserve the original four layers' seeded cases, then append the four
        // additional pairs present in this pack. Coverage is checked, not assumed.
        const std::vector<int> layers{0, 7, 17, 39, 10, 11, 23, 38};
        std::set<std::pair<uint32_t,uint32_t>> required, covered;
        for (int l=0; l<g.layers; ++l) {
            const auto *gu=model.find("blk."+std::to_string(l)+".ffn_gate_exps.weight");
            const auto *dn=model.find("blk."+std::to_string(l)+".ffn_down_exps.weight");
            if (!gu || !dn) throw std::runtime_error("missing format inventory tensor");
            const auto pair=std::make_pair(gu->type,dn->type);
            required.insert(pair);
            if (std::find(layers.begin(),layers.end(),l)!=layers.end()) covered.insert(pair);
        }
        if (required!=covered) throw std::runtime_error("fixture does not cover every expert-format pair in the pack");
        std::printf("expert_format_pairs=%zu all_pairs_covered=1\n",covered.size());
        std::vector<ExpertWorkerPair> plan;
        for (int l : layers) for (int e = 0; e < 8; ++e) plan.push_back({l, e});
        if (!net.install_plan(plan, error)) throw std::runtime_error(error);
        // Invalid local requests must be refused before they can reach the worker.
        std::vector<float> invalid_x(17 * g.hidden, 0.25f);
        auto reject = [&](int layer, int tokens, std::vector<ExpertWorkerAssignment> a, const char *name) {
            if (net.begin_async(layer, tokens, invalid_x.data(), a))
                throw std::runtime_error(std::string("accepted invalid request: ") + name);
            if (!net.healthy()) throw std::runtime_error("local rejection poisoned the session");
            std::printf("reject=%s PASS\n", name);
        };
        reject(0, 1, {{0,0,8,1}}, "unowned expert");
        reject(1, 1, {{0,0,0,1}}, "unowned layer");
        reject(0, 1, {{0,0,0,1},{0,0,1,1}}, "duplicate slot");
        reject(0, 1, {{0,6,0,1}}, "slot exceeds top-six");
        reject(0, 1, {{1,0,0,1}}, "token exceeds rows");
        reject(0, 17, {{0,0,0,1}}, "token count exceeds limit");
        reject(0, 1, {{0,0,0,std::numeric_limits<float>::infinity()}}, "nonfinite weight");
        invalid_x[0] = std::numeric_limits<float>::quiet_NaN();
        reject(0, 1, {{0,0,0,1}}, "nonfinite activation");
        Memory memory;
        auto *dx = memory.get<float>(g.hidden);
        auto *dh = memory.get<float>(g.intermediate);
        auto *dy = memory.get<float>(g.hidden);
        auto *dp = memory.get<unsigned long long>(1);
        auto *dw = memory.get<float>(1);
        const float one = 1;
        ck(cudaMemcpy(dw, &one, sizeof(one), cudaMemcpyHostToDevice));
        std::mt19937 rng(42);
        std::normal_distribution<float> random(0, .5f);
        int cases = 0;
        for (int layer : layers) {
            size_t sg = 0, su = 0, sd = 0;
            const auto prefix = "blk." + std::to_string(layer) + ".";
            const auto *gate = model.find(prefix + "ffn_gate_exps.weight", &sg);
            const auto *up = model.find(prefix + "ffn_up_exps.weight", &su);
            const auto *down = model.find(prefix + "ffn_down_exps.weight", &sd);
            if (!gate || !up || !down || !model.in_bounds(*gate, sg) || !model.in_bounds(*up, su) || !model.in_bounds(*down, sd))
                throw std::runtime_error("invalid reference tensors");
            if (gate->type != up->type) throw std::runtime_error("reference gate/up types differ");
            std::printf("layer=%d expert_format_pair=%u/%u\n", layer, gate->type, down->type);
            G::IqCpu cpu;
            if (!G::iq_cpu_fmt(gate->type, down->type, g.hidden, g.intermediate, cpu, error)) throw std::runtime_error(error);
            const size_t gu_bytes = cpu.gu_row * g.intermediate, down_bytes = cpu.d_row * g.hidden;
            const size_t total_bytes = 2*gu_bytes + down_bytes;
            auto *db = memory.get<uint8_t>(total_bytes);
            auto evaluate = [&](int expert, const float *x, std::vector<float> &gpu, std::vector<float> &host) {
                std::vector<uint8_t> blob(total_bytes);
                std::memcpy(blob.data(), model.shard(sg).tensor_data(*gate) + expert*gu_bytes, gu_bytes);
                std::memcpy(blob.data()+gu_bytes, model.shard(su).tensor_data(*up) + expert*gu_bytes, gu_bytes);
                std::memcpy(blob.data()+2*gu_bytes, model.shard(sd).tensor_data(*down) + expert*down_bytes, down_bytes);
                ck(cudaMemcpy(db, blob.data(), total_bytes, cudaMemcpyHostToDevice));
                ck(cudaMemcpy(dx, x, g.hidden*sizeof(float), cudaMemcpyHostToDevice));
                const auto pointer = reinterpret_cast<unsigned long long>(db);
                ck(cudaMemcpy(dp, &pointer, sizeof(pointer), cudaMemcpyHostToDevice));
                X::MoeArgs args;
                args.plan_ptr = dp; args.plan_w = dw; args.k = 1;
                args.n_embd = g.hidden; args.n_ff = g.intermediate; args.limit = g.limit;
                args.lay.fmt = G::kFmtMXFP4;
                args.lay.K[0] = args.lay.K[1] = gate->type; args.lay.K[2] = down->type;
                args.lay.trellis[0] = 0; args.lay.trellis[1] = gu_bytes; args.lay.trellis[2] = 2*gu_bytes;
                if (!G::prepare(nullptr)) throw std::runtime_error("local GPU scratch initialization failed");
                const auto errors = G::launch_errors();
                G::moe_gate_up(args, dx, dh, nullptr);
                G::moe_down(args, dh, nullptr, dy, nullptr);
                if (G::launch_errors() != errors) throw std::runtime_error("local GPU expert execution refused");
                ck(cudaDeviceSynchronize());
                ck(cudaMemcpy(gpu.data(), dy, g.hidden*sizeof(float), cudaMemcpyDeviceToHost));
                std::vector<uint8_t> q(cpu.act_bytes), hq(cpu.h_bytes);
                std::vector<float> intermediate(g.intermediate);
                G::iq_cpu_quant(cpu.gu_act, x, g.hidden, q.data());
                const void *qa[1] = {q.data()}; float *ff[1] = {intermediate.data()};
                G::iq_cpu_gate_up(cpu, blob.data(), blob.data()+gu_bytes, qa, 1, ff, 0, g.intermediate, g.limit);
                G::iq_cpu_quant(cpu.d_act, intermediate.data(), g.intermediate, hq.data());
                const void *qh[1] = {hq.data()}; float *out[1] = {host.data()};
                G::iq_cpu_down(cpu, blob.data()+2*gu_bytes, qh, 1, out, 0, g.hidden);
            };
            for (int mode = 0; mode < 6; ++mode) {
                const int tokens = mode == 3 || mode == 4 ? 16 : mode == 2 ? 3 : 1;
                std::vector<ExpertWorkerAssignment> assignments = mode == 0
                    ? std::vector<ExpertWorkerAssignment>{{0,0,3,1}}
                    : mode == 1 ? std::vector<ExpertWorkerAssignment>{{0,0,0,.4f},{0,2,2,.3f},{0,5,5,.2f}}
                    : std::vector<ExpertWorkerAssignment>{{0,0,1,.4f},{2,1,1,.5f},{2,3,4,.3f}};
                if (mode >= 3) {
                    assignments.clear();
                    for (int t = 0; t < tokens; ++t)
                        for (int slot = 0; slot < (mode == 4 ? g.picks : 1); ++slot)
                            assignments.push_back({static_cast<uint16_t>(t), static_cast<uint16_t>(slot),
                                                   (t + slot) % 8, slot == 5 ? 0.0f : 1.0f / g.picks});
                }
                std::vector<float> x(tokens*g.hidden), got(x.size()), ref_gpu(x.size()), ref_cpu(x.size());
                for (auto &v : x) v = random(rng) * (mode == 5 ? 64.0f : 1.0f);
                std::vector<uint8_t> mask(tokens);
                if (!net.execute_sync(layer, tokens, x.data(), assignments, got.data(), mask.data(), &error)) throw std::runtime_error(error);
                std::vector<uint8_t> expected(tokens);
                for (const auto a : assignments) {
                    expected[a.token_idx] = 1;
                    std::vector<float> gpu(g.hidden), host(g.hidden);
                    evaluate(a.expert_id, x.data()+a.token_idx*g.hidden, gpu, host);
                    for (int c = 0; c < g.hidden; ++c) {
                        const auto i = a.token_idx*g.hidden+c;
                        ref_gpu[i] = std::fma(a.weight, gpu[c], ref_gpu[i]);
                        ref_cpu[i] = std::fma(a.weight, host[c], ref_cpu[i]);
                    }
                }
                if (mask != expected) throw std::runtime_error("invalid token coverage");
                for (int t = 0; t < tokens; ++t)
                    if (!expected[t] && std::any_of(got.begin()+t*g.hidden, got.begin()+(t+1)*g.hidden,
                                                   [](float v) { return v != 0.0f; }))
                        throw std::runtime_error("nonzero uncovered token row");
                std::printf("layer=%d case=%d tokens=%d assignments=%zu", layer, mode, tokens, assignments.size());
                const bool gpu_ok = close(ref_gpu, got, false), cpu_ok = close(ref_cpu, got, true);
                std::printf(" %s\n", gpu_ok && cpu_ok ? "PASS" : "FAIL"); std::fflush(stdout);
                if (!gpu_ok || !cpu_ok) throw std::runtime_error("DeepSeek numerical parity failed");
                ++cases;
            }
        }
        net.print_summary();
        std::printf("DeepSeek expert parity: %d cases PASS\n", cases);
        return 0;
    } catch (const std::exception &e) { std::fprintf(stderr, "%s\n", e.what()); return 1; }
}
