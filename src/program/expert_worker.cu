// src/program/expert_worker.cu - strata-expert-worker: standalone MoE expert worker service.
//
// Runs on the worker host's GPU device 0 (e.g. AMD Radeon RX 6800 XT gfx1030 via HIP, or CUDA device 0).
// Preloads an immutable set of (layer, expert) native blobs into VRAM upon INSTALL_PLAN (or at startup via
// --placement), executes grouped expert kernels + GPU-side weighted partial-sum reduction per EXECUTE request,
// and returns at most one FP32 vector per token plus explicit coverage mask and microsecond stage timings.
// Opt-in --expert-vram-mib / --resident-budget-mib separate ownership from
// residency, using the existing native CPU pool for RAM and mapped-file experts.

#include "strata/artifact/gguf_reader.hpp"
#include "strata/core/expert_worker_identity.hpp"
#include "strata/core/expert_worker_wire.hpp"
#include "strata/core/family_layout.hpp"
#include "strata/kernels/iq_kernels.hpp"
#include "strata/kernels/native_mmvq.hpp"
#include "strata/kernels/native_moe.hpp"
#include "strata/kernels/cpu/pool.hpp"
#if defined(STRATA_WORKER_DSV41)
#include "strata/core/deepseek_worker_geometry.hpp"
#include "ds4_gguf.cuh"
#include "ds4_worker_cpu.hpp"
#endif
#ifdef STRATA_KIMI2_LIBRARY
#include "strata/kimi2/engine.hpp"
#include "strata/kimi2/model.hpp"
#include "strata/kernels/cpu/native_expert.hpp"
#include <fstream>
#endif
#include <set>
#include <thread>

#include <cuda_runtime.h>

#include <algorithm>
#include <cerrno>
#include <chrono>
#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <map>
#include <memory>
#include <string>
#include <utility>
#include <vector>

#if !defined(_WIN32)
#include <netinet/in.h>
#include <sys/socket.h>
#include <unistd.h>
#endif

namespace {

using namespace strata::core;
namespace K = strata::kernels;
using Clock = std::chrono::steady_clock;

uint64_t us_since(Clock::time_point t0) {
    return static_cast<uint64_t>(std::chrono::duration_cast<std::chrono::microseconds>(Clock::now() - t0).count());
}

constexpr int kMaxTokens = 16;
constexpr int kMaxTopK = 16;
constexpr int kMaxCap = kMaxTokens * kMaxTopK;

struct WorkerGroupMeta {
    unsigned long long ptr[kMaxCap];
    int32_t start[kMaxCap + 1];
    int32_t dst[kMaxCap];
    int32_t tok[kMaxCap];
    int32_t count;
    // Reduction table: for each token t in [0, n_tokens), up to kMaxTopK compact row indices and weights
    int32_t red_cnt[kMaxTokens];
    int32_t red_row[kMaxTokens * kMaxTopK];
    float red_wt[kMaxTokens * kMaxTopK];
};

__global__ void k_reduce_weighted_parts(const float *__restrict__ parts, const WorkerGroupMeta *__restrict__ meta,
                                        float *__restrict__ out_sum, int n_embd) {
    const int t = blockIdx.y;
    const int c = blockIdx.x * blockDim.x + threadIdx.x;
    if (c >= n_embd)
        return;
    const int cnt = meta->red_cnt[t];
    float acc = 0.0f;
    if (cnt > 0) {
        const int base = t * kMaxTopK;
        const int r0 = meta->red_row[base];
        acc = meta->red_wt[base] * parts[static_cast<size_t>(r0) * n_embd + c];
        for (int j = 1; j < cnt; ++j) {
            const int rj = meta->red_row[base + j];
            acc = fmaf(meta->red_wt[base + j], parts[static_cast<size_t>(rj) * n_embd + c], acc);
        }
    }
    out_sum[static_cast<size_t>(t) * n_embd + c] = acc;
}

struct WorkerLayerSource {
    K::NativeExpertLayout layout{};
#ifdef STRATA_KIMI2_LIBRARY
    std::string fixture;
#endif
    const uint8_t *src_g = nullptr;
    const uint8_t *src_u = nullptr;
    const uint8_t *src_d = nullptr;
    size_t src_gsz = 0;
    size_t src_dsz = 0;
    size_t slot_stride = 0;
#if defined(STRATA_WORKER_DSV41)
    strata::ds4::gguf::IqCpu deepseek_cpu{};
#endif
};

struct WorkerContext {
#ifdef STRATA_KIMI2_LIBRARY
    bool kimi2 = false, kimi2_full = false;
    std::string kimi2_identity;
    std::unique_ptr<strata::kimi2::Model> kimi2_model;
#endif
    int device = 0;
    std::string device_name;
    std::shared_ptr<strata::GgufModel> gguf;
    ModelGeometry geom{};
    int64_t top_k = 8;
    int32_t total_layers = 0; // g.n_layers + (g.n_mtp_layers > 0 ? 1 : 0)
    uint64_t fingerprint = 0;
    std::array<uint8_t, 32> model_sha256{};
    bool deepseek = false;
    ExpertWorkerArithmetic arithmetic = ExpertWorkerArithmetic::NATIVE_V1;
    float swiglu_limit = 0;
    cudaEvent_t timing[4]{};
    std::vector<WorkerLayerSource> layers;

    // Resident cache in GPU VRAM:
    // slot_of[layer * n_expert + expert] -> slot index in d_arena (or -1 if not resident)
    std::vector<int32_t> slot_of;
    std::vector<size_t> slot_offset; // byte offset of slot s in d_arena
    uint8_t *d_arena = nullptr;
    size_t arena_bytes = 0;
    uint32_t loaded_count = 0;
    // Opt-in static placement budgets. Local execution reuses the native CPU pool
    // and its mapped-GGUF NativeFmt overrides; no replacement cache is introduced.
    int64_t expert_vram_mib = -1;
    size_t resident_budget_mib = 0;
    std::vector<uint8_t> host_arena;
    std::vector<const uint8_t *> host_blob_of;
    std::unique_ptr<K::cpu::ExpertPool> cpu_pool;
#if defined(STRATA_WORKER_DSV41)
    std::unique_ptr<strata::ds4::Pool> deepseek_cpu_pool;
#endif
    uint64_t gpu_assignments = 0, ram_assignments = 0, mapped_assignments = 0;

    // Execution buffers on device
    cudaStream_t stream = nullptr;
    WorkerGroupMeta *h_meta = nullptr;
    WorkerGroupMeta *d_meta = nullptr;
    float *h_x = nullptr;
    float *d_x = nullptr;
    void *d_q8 = nullptr;
    void *d_scratch = nullptr;
    float *d_parts = nullptr;
    float *d_sum = nullptr;
    float *h_sum = nullptr;

    ~WorkerContext() { close(); }

    void close_cache() {
        if (d_arena) {
            cudaFree(d_arena);
            d_arena = nullptr;
        }
        arena_bytes = 0;
        loaded_count = 0;
        std::fill(slot_of.begin(), slot_of.end(), -1);
        slot_offset.clear();
        host_arena.clear();
        host_blob_of.clear();
        gpu_assignments = ram_assignments = mapped_assignments = 0;
    }

    void close() {
        if (stream)
            cudaStreamSynchronize(stream);
        close_cache();
        for (auto e : timing)
            if (e)
                cudaEventDestroy(e);
        if (d_meta) {
            cudaFree(d_meta);
            d_meta = nullptr;
        }
        if (h_meta) {
            cudaFreeHost(h_meta);
            h_meta = nullptr;
        }
        if (d_x) {
            cudaFree(d_x);
            d_x = nullptr;
        }
        if (h_x) {
            cudaFreeHost(h_x);
            h_x = nullptr;
        }
        if (d_q8) {
            cudaFree(d_q8);
            d_q8 = nullptr;
        }
        if (d_scratch) {
            cudaFree(d_scratch);
            d_scratch = nullptr;
        }
        if (d_parts) {
            cudaFree(d_parts);
            d_parts = nullptr;
        }
        if (d_sum) {
            cudaFree(d_sum);
            d_sum = nullptr;
        }
        if (h_sum) {
            cudaFreeHost(h_sum);
            h_sum = nullptr;
        }
        if (stream) {
            cudaStreamDestroy(stream);
            stream = nullptr;
        }
    }
};

bool check_cuda(cudaError_t e, const char *what, std::string &err) {
    if (e == cudaSuccess)
        return true;
    err = std::string(what) + ": " + cudaGetErrorString(e);
    return false;
}

void pack_expert_blob(const WorkerLayerSource &L, int32_t e, uint8_t *dst) {
#ifdef STRATA_KIMI2_LIBRARY
    if(!L.fixture.empty()){
        for(int role=0;role<3;++role){
            const char* name=role==0?"gate":role==1?"up":"down";
            const size_t bytes=role==2?L.src_dsz:L.src_gsz;
            const size_t offset=role==0?0:role==1?L.layout.up_off:L.layout.down_off;
            std::ifstream f(L.fixture+"/ffn_"+name+"_exps.weight.expert"+std::to_string(e)+".bin",std::ios::binary|std::ios::ate);
            if(!f || static_cast<size_t>(f.tellg())!=bytes)throw std::runtime_error("missing/wrong K2 expert slice");
            f.seekg(0);if(!f.read(reinterpret_cast<char*>(dst+offset),bytes))throw std::runtime_error("K2 expert slice read failed");
        }
        return;
    }
#endif
    std::memcpy(dst, L.src_g + static_cast<size_t>(e) * L.src_gsz, L.src_gsz);
    std::memcpy(dst + L.layout.up_off, L.src_u + static_cast<size_t>(e) * L.src_gsz, L.src_gsz);
    std::memcpy(dst + L.layout.down_off, L.src_d + static_cast<size_t>(e) * L.src_dsz, L.src_dsz);
}

bool init_worker_model(const std::string &model_path, int device, WorkerContext &ctx, std::string &err) {
    int count = 0;
    if (!check_cuda(cudaGetDeviceCount(&count), "cudaGetDeviceCount", err))
        return false;
    if (device < 0 || device >= count) {
        err = "requested GPU device " + std::to_string(device) + " not available (count=" + std::to_string(count) + ")";
        return false;
    }
    if (!check_cuda(cudaSetDevice(device), "cudaSetDevice", err))
        return false;
    ctx.device = device;

    cudaDeviceProp prop{};
    if (!check_cuda(cudaGetDeviceProperties(&prop, device), "cudaGetDeviceProperties", err))
        return false;
    ctx.device_name = prop.name;

#ifdef STRATA_KIMI2_LIBRARY
    if(ctx.kimi2){
        ctx.geom.n_embd=7168;ctx.geom.n_ff=2048;ctx.geom.n_layers=61;ctx.geom.n_expert=384;
        ctx.geom.n_expert_used=8;ctx.top_k=8;ctx.total_layers=61;
        if(ctx.kimi2_full)ctx.kimi2_model=std::make_unique<strata::kimi2::Model>(model_path);
        const std::string identity_path=ctx.kimi2_full?ctx.kimi2_identity:model_path+"/fixture.identity";
        std::string identity;std::ifstream id(identity_path);id>>identity;
        if(identity.size()!=64)throw std::runtime_error("K2 fixture requires exact manifest identity");
        for(int i=0;i<32;++i)ctx.model_sha256[i]=static_cast<uint8_t>(std::stoul(identity.substr(2*i,2),nullptr,16));
        ctx.fingerprint=compute_model_fingerprint(61,384,7168,2048);
        ctx.layers.resize(61);
        for(int l=1;l<61;++l){
            const std::string dir=model_path+"/experts"+std::to_string(l);
            int gu=0,down=0;std::ifstream g(dir+"/ffn_gate_exps.type"),d(dir+"/ffn_down_exps.type");
            if(ctx.kimi2_full){
                gu=ctx.kimi2_model->type("blk."+std::to_string(l)+".ffn_gate_exps.weight");
                down=ctx.kimi2_model->type("blk."+std::to_string(l)+".ffn_down_exps.weight");
            }else{
                if(!g && !d)continue;
                if(!(g>>gu) || !(d>>down))throw std::runtime_error("invalid K2 fixture types");
            }
            strata::kernels::cpu::NativeFmt fmt;std::string error;
            if(!strata::kernels::cpu::native_fmt(gu,down,7168,2048,fmt,error,true))throw std::runtime_error("unsupported K2 fixture expert format: "+error);
            auto& wl=ctx.layers[l];
            if(ctx.kimi2_full){
                const std::string prefix="blk."+std::to_string(l)+".";
                wl.src_g=ctx.kimi2_model->data(prefix+"ffn_gate_exps.weight");
                wl.src_u=ctx.kimi2_model->data(prefix+"ffn_up_exps.weight");
                wl.src_d=ctx.kimi2_model->data(prefix+"ffn_down_exps.weight");
            }else wl.fixture=dir;
            wl.layout.gu_type=gu;wl.layout.d_type=down;wl.layout.n_embd=7168;wl.layout.n_ff=2048;
            wl.layout.gu_row=fmt.gu_row;wl.layout.d_row=fmt.d_row;
            wl.layout.up_off=fmt.up_off;wl.layout.down_off=fmt.down_off;wl.layout.bytes=fmt.bytes;
            wl.src_gsz=2048*wl.layout.gu_row;wl.src_dsz=7168*wl.layout.d_row;
            wl.slot_stride=(wl.layout.bytes+255u)&~size_t(255u);
        }
    }else
#endif
    {
    try {
        ctx.gguf = std::make_shared<strata::GgufModel>(strata::GgufModel::open(model_path));
    } catch (const std::exception &ex) {
        err = std::string("failed to open GGUF model: ") + ex.what();
        return false;
    }
    const strata::GgufModel &gm = *ctx.gguf;
    const auto *arch = gm.meta().get("general.architecture");
#if defined(STRATA_WORKER_DSV41)
    if (arch && arch->s == "deepseek41") {
        const DeepSeekWorkerGeometry geometry(gm);
        ctx.deepseek = true;
        ctx.arithmetic = strata::ds4::gguf::iq_q8k_activations() ? ExpertWorkerArithmetic::DEEPSEEK_IQ_Q8K
                                                              : ExpertWorkerArithmetic::DEEPSEEK_IQ_PER32;
        ctx.geom.n_layers = geometry.layers; ctx.geom.n_expert = geometry.experts;
        ctx.geom.n_embd = geometry.hidden; ctx.geom.n_ff = geometry.intermediate;
        ctx.top_k = geometry.picks; ctx.swiglu_limit = geometry.limit;
        if (ctx.expert_vram_mib >= 0 && ctx.arithmetic != ExpertWorkerArithmetic::DEEPSEEK_IQ_Q8K) {
            err = "DeepSeek tiered worker requires qualified DS4_IQ_Q8K=1 arithmetic";
            return false;
        }
    } else
#endif
    if (!geometry_from_gguf(gm.meta(), ctx.geom, ctx.top_k, err)) return false;
    const auto &g = ctx.geom;
    if (!arch || (!ctx.deepseek && arch->s != "qwen35moe") || ctx.top_k < 1 || ctx.top_k > kMaxTopK) {
        err = "network worker v1 requires qwen35moe with top-k in 1..16";
        return false;
    }
    ctx.total_layers = static_cast<int32_t>(g.n_layers);
    if (ctx.deepseek)
        std::fprintf(stderr, "ds4-network-startup: role=expert-worker effective_experts=%d; hashing exact GGUF mappings\n", ctx.top_k);
    ctx.model_sha256 = expert_worker_model_sha256(gm);
    if (ctx.deepseek) {
        std::fprintf(stderr, "ds4-network-identity: role=expert-worker effective_experts=%d engine=%s arithmetic=%u model_sha256=", ctx.top_k, STRATA_VERSION, static_cast<uint32_t>(ctx.arithmetic));
        for (const auto b : ctx.model_sha256) std::fprintf(stderr, "%02x", b);
        std::fprintf(stderr, "\n");
    }
    ctx.fingerprint = compute_model_fingerprint(static_cast<uint32_t>(g.n_layers), static_cast<uint32_t>(g.n_expert),
                                                static_cast<uint32_t>(g.n_embd), static_cast<uint32_t>(g.n_ff));

    ctx.layers.resize(static_cast<size_t>(ctx.total_layers));
    for (int32_t l = 0; l < ctx.total_layers; ++l) {
        auto tname = [&](const char *s) { return "blk." + std::to_string(l) + "." + s; };
        size_t sg = 0, su = 0, sd = 0;
        const strata::TensorInfo *tg = gm.find(tname("ffn_gate_exps.weight"), &sg);
        const strata::TensorInfo *tu = gm.find(tname("ffn_up_exps.weight"), &su);
        const strata::TensorInfo *td = gm.find(tname("ffn_down_exps.weight"), &sd);
        if (!tg || !tu || !td) {
            err = "missing routed expert tensors at layer " + std::to_string(l);
            return false;
        }
        const std::vector<uint64_t> gu_shape{static_cast<uint64_t>(g.n_embd), static_cast<uint64_t>(g.n_ff),
                                             static_cast<uint64_t>(g.n_expert)};
        const std::vector<uint64_t> dn_shape{static_cast<uint64_t>(g.n_ff), static_cast<uint64_t>(g.n_embd),
                                             static_cast<uint64_t>(g.n_expert)};
        if (tg->shape != gu_shape || tu->shape != gu_shape || td->shape != dn_shape || !gm.in_bounds(*tg, sg) ||
            !gm.in_bounds(*tu, su) || !gm.in_bounds(*td, sd)) {
            err = "invalid expert tensor shape or bounds at layer " + std::to_string(l);
            return false;
        }
        if (tg->type != tu->type) {
            err = "ffn_gate_exps / ffn_up_exps type mismatch at layer " + std::to_string(l);
            return false;
        }
        if (!ctx.deepseek && !K::native_expert_supported(static_cast<int>(tg->type), static_cast<int>(td->type), g.n_embd, g.n_ff)) {
            err = "unsupported expert quantization types at layer " + std::to_string(l);
            return false;
        }
        WorkerLayerSource &wl = ctx.layers[static_cast<size_t>(l)];
#if defined(STRATA_WORKER_DSV41)
        if (ctx.deepseek) {
            namespace G = strata::ds4::gguf;
            if (!G::type_on_device(tg->type) || !G::type_on_device(td->type)) { err = "unsupported DeepSeek expert type"; return false; }
            wl.layout.gu_type = tg->type; wl.layout.d_type = td->type;
            wl.layout.n_embd = g.n_embd; wl.layout.n_ff = g.n_ff;
            wl.layout.gu_row = G::type_row_bytes(tg->type, g.n_embd);
            wl.layout.d_row = G::type_row_bytes(td->type, g.n_ff);
            wl.layout.up_off = wl.layout.gu_row * g.n_ff;
            wl.layout.down_off = 2 * wl.layout.up_off;
            wl.layout.bytes = wl.layout.down_off + wl.layout.d_row * g.n_embd;
            if (ctx.expert_vram_mib >= 0 && !G::iq_cpu_fmt(tg->type, td->type, g.n_embd, g.n_ff, wl.deepseek_cpu, err))
                return false;
        } else
#endif
        wl.layout = K::native_expert_layout(static_cast<int>(tg->type), static_cast<int>(td->type), g.n_embd, g.n_ff);
        wl.src_gsz = static_cast<size_t>(g.n_ff) * wl.layout.gu_row;
        wl.src_dsz = static_cast<size_t>(g.n_embd) * wl.layout.d_row;
        wl.slot_stride = (wl.layout.bytes + 255u) & ~size_t(255u);
        wl.src_g = gm.shard(sg).tensor_data(*tg);
        wl.src_u = gm.shard(su).tensor_data(*tu);
        wl.src_d = gm.shard(sd).tensor_data(*td);
    }

    } // model-specific source binding; allocations and protocol remain shared
    for (auto &e : ctx.timing)
        if (!check_cuda(cudaEventCreate(&e), "cudaEventCreate", err))
            return false;
    ctx.slot_of.assign(static_cast<size_t>(ctx.total_layers) * static_cast<size_t>(ctx.geom.n_expert), -1);

    const size_t N = static_cast<size_t>(ctx.geom.n_embd);
    const size_t FF = static_cast<size_t>(ctx.geom.n_ff);
    const size_t q8_bytes = K::native_q8_1_bytes(static_cast<int>(N), 1) * kMaxTokens;
    const size_t scratch_bytes = K::native_expert_scratch_bytes(kMaxCap, static_cast<int64_t>(FF));

    if (!check_cuda(cudaStreamCreateWithFlags(&ctx.stream, cudaStreamNonBlocking), "cudaStreamCreate", err) ||
        !check_cuda(
            cudaHostAlloc(reinterpret_cast<void **>(&ctx.h_meta), sizeof(WorkerGroupMeta), cudaHostAllocDefault),
            "cudaHostAlloc(h_meta)", err) ||
        !check_cuda(cudaMalloc(reinterpret_cast<void **>(&ctx.d_meta), sizeof(WorkerGroupMeta)), "cudaMalloc(d_meta)",
                    err) ||
        !check_cuda(
            cudaHostAlloc(reinterpret_cast<void **>(&ctx.h_x), kMaxTokens * N * sizeof(float), cudaHostAllocDefault),
            "cudaHostAlloc(h_x)", err) ||
        !check_cuda(cudaMalloc(reinterpret_cast<void **>(&ctx.d_x), kMaxTokens * N * sizeof(float)), "cudaMalloc(d_x)",
                    err) ||
        !check_cuda(cudaMalloc(&ctx.d_q8, q8_bytes), "cudaMalloc(d_q8)", err) ||
        !check_cuda(cudaMalloc(&ctx.d_scratch, scratch_bytes), "cudaMalloc(d_scratch)", err) ||
        !check_cuda(cudaMalloc(reinterpret_cast<void **>(&ctx.d_parts), kMaxCap * N * sizeof(float)),
                    "cudaMalloc(d_parts)", err) ||
        !check_cuda(cudaMalloc(reinterpret_cast<void **>(&ctx.d_sum), kMaxTokens * N * sizeof(float)),
                    "cudaMalloc(d_sum)", err) ||
        !check_cuda(
            cudaHostAlloc(reinterpret_cast<void **>(&ctx.h_sum), kMaxTokens * N * sizeof(float), cudaHostAllocDefault),
            "cudaHostAlloc(h_sum)", err)) {
        ctx.close();
        return false;
    }

    return true;
}

bool install_gpu_plan(WorkerContext &ctx, const ExpertWorkerPair *pairs, uint32_t n_pairs,
                         ExpertWorkerReadyPayload &out_ready, std::string &err) {
    const auto t0 = Clock::now();
    ctx.close_cache();
    out_ready = {};

    const int32_t n_exp = static_cast<int32_t>(ctx.geom.n_expert);
    std::vector<ExpertWorkerPair> unique_pairs;
    unique_pairs.reserve(n_pairs);

    size_t needed_bytes = 0;
    for (uint32_t i = 0; i < n_pairs; ++i) {
        const int32_t l = pairs[i].layer;
        const int32_t e = pairs[i].expert;
        if (l < 0 || l >= ctx.total_layers || e < 0 || e >= n_exp || !ctx.layers[l].slot_stride) {
            err = "pair (" + std::to_string(l) + "," + std::to_string(e) + ") out of bounds";
            out_ready.status = static_cast<uint32_t>(ExpertWorkerErrorCode::PLAN_LOAD_FAILED);
            return false;
        }
        const size_t idx = static_cast<size_t>(l) * static_cast<size_t>(n_exp) + static_cast<size_t>(e);
        if (ctx.slot_of[idx] >= 0)
            continue; // duplicate in list
        ctx.slot_of[idx] = static_cast<int32_t>(unique_pairs.size());
        ctx.slot_offset.push_back(needed_bytes);
        needed_bytes += ctx.layers[static_cast<size_t>(l)].slot_stride;
        unique_pairs.push_back({l, e});
    }

    size_t free_b = 0, total_b = 0;
    if (!check_cuda(cudaMemGetInfo(&free_b, &total_b), "cudaMemGetInfo", err)) {
        ctx.close_cache();
        out_ready.status = static_cast<uint32_t>(ExpertWorkerErrorCode::PLAN_LOAD_FAILED);
        return false;
    }

    if (unique_pairs.empty()) {
        out_ready.status = static_cast<uint32_t>(ExpertWorkerErrorCode::OK);
        out_ready.loaded_count = 0;
        out_ready.vram_used_bytes = 0;
        out_ready.vram_free_bytes = free_b;
        out_ready.load_time_us = us_since(t0);
        return true;
    }

    const size_t reserve = 256ULL << 20; // keep at least 256 MiB free
    if (needed_bytes + reserve > free_b) {
        err = "plan requires " + std::to_string(needed_bytes >> 20) + " MiB, but only " + std::to_string(free_b >> 20) +
              " MiB free on GPU";
        ctx.close_cache();
        out_ready.status = static_cast<uint32_t>(ExpertWorkerErrorCode::PLAN_LOAD_FAILED);
        return false;
    }

    if (!check_cuda(cudaMalloc(reinterpret_cast<void **>(&ctx.d_arena), needed_bytes), "cudaMalloc(d_arena)", err)) {
        ctx.close_cache();
        out_ready.status = static_cast<uint32_t>(ExpertWorkerErrorCode::PLAN_LOAD_FAILED);
        return false;
    }
    ctx.arena_bytes = needed_bytes;

    std::vector<uint8_t> host_blob;
    for (size_t s = 0; s < unique_pairs.size(); ++s) {
        const int32_t l = unique_pairs[s].layer;
        const int32_t e = unique_pairs[s].expert;
        const WorkerLayerSource &wl = ctx.layers[static_cast<size_t>(l)];
        if (host_blob.size() < wl.layout.bytes)
            host_blob.resize(wl.layout.bytes);
        pack_expert_blob(wl, e, host_blob.data());
        if (!check_cuda(
                cudaMemcpy(ctx.d_arena + ctx.slot_offset[s], host_blob.data(), wl.layout.bytes, cudaMemcpyHostToDevice),
                "cudaMemcpy(expert_blob)", err)) {
            ctx.close_cache();
            out_ready.status = static_cast<uint32_t>(ExpertWorkerErrorCode::PLAN_LOAD_FAILED);
            return false;
        }
    }

    // Verify every acknowledged slot byte-for-byte from VRAM.
    for (size_t s = 0; s < unique_pairs.size(); ++s) {
        const int32_t l = unique_pairs[s].layer;
        const int32_t e = unique_pairs[s].expert;
        const WorkerLayerSource &wl = ctx.layers[static_cast<size_t>(l)];
        std::vector<uint8_t> expected(wl.layout.bytes), actual(wl.layout.bytes);
        pack_expert_blob(wl, e, expected.data());
        if (!check_cuda(
                cudaMemcpy(actual.data(), ctx.d_arena + ctx.slot_offset[s], wl.layout.bytes, cudaMemcpyDeviceToHost),
                "cudaMemcpy(verify_slot)", err) ||
            std::memcmp(expected.data(), actual.data(), wl.layout.bytes) != 0) {
            err = "VRAM verification failed for slot " + std::to_string(s);
            ctx.close_cache();
            out_ready.status = static_cast<uint32_t>(ExpertWorkerErrorCode::PLAN_LOAD_FAILED);
            return false;
        }
    }

    ctx.loaded_count = static_cast<uint32_t>(unique_pairs.size());
    if (!check_cuda(cudaMemGetInfo(&free_b, &total_b), "cudaMemGetInfo", err)) {
        ctx.close_cache();
        out_ready.status = static_cast<uint32_t>(ExpertWorkerErrorCode::PLAN_LOAD_FAILED);
        return false;
    }
    out_ready.status = static_cast<uint32_t>(ExpertWorkerErrorCode::OK);
    out_ready.loaded_count = ctx.loaded_count;
    out_ready.vram_used_bytes = ctx.arena_bytes;
    out_ready.vram_free_bytes = free_b;
    out_ready.load_time_us = us_since(t0);
    return true;
}

bool install_worker_plan(WorkerContext &ctx, const ExpertWorkerPair *pairs, uint32_t n_pairs,
                         ExpertWorkerReadyPayload &ready, std::string &err) {
    if (ctx.expert_vram_mib < 0) return install_gpu_plan(ctx, pairs, n_pairs, ready, err);
    const auto start = Clock::now();
    ready = {};
    ready.status = static_cast<uint32_t>(ExpertWorkerErrorCode::PLAN_LOAD_FAILED);
    std::vector<ExpertWorkerPair> gpu, cpu;
    std::set<std::pair<int32_t, int32_t>> seen;
    size_t gpu_bytes = 0, host_bytes = 0;
    const size_t gpu_budget = static_cast<size_t>(ctx.expert_vram_mib) << 20;
    const size_t host_budget = ctx.resident_budget_mib << 20;
    for (uint32_t i = 0; i < n_pairs; ++i) {
        const auto p = pairs[i];
        if (p.layer < 0 || p.layer >= ctx.total_layers || p.expert < 0 || p.expert >= ctx.geom.n_expert ||
            !seen.emplace(p.layer, p.expert).second) {
            err = "invalid or duplicate tiered ownership pair";
            return false;
        }
        const auto &wl = ctx.layers[p.layer];
#if defined(STRATA_WORKER_DSV41)
        if (ctx.deepseek) {
            if (wl.deepseek_cpu.gu_row != wl.layout.gu_row || wl.deepseek_cpu.d_row != wl.layout.d_row) {
                err = "DeepSeek worker CPU/GPU row layouts differ";return false;
            }
        } else
#endif
        {
            K::cpu::NativeFmt fmt;
            if (!K::cpu::native_fmt(wl.layout.gu_type, wl.layout.d_type, ctx.geom.n_embd, ctx.geom.n_ff, fmt, err))
                return false;
            if (fmt.bytes != wl.layout.bytes || fmt.up_off != wl.layout.up_off || fmt.down_off != wl.layout.down_off ||
                fmt.act_bytes > K::cpu::kNativeActBytes || fmt.h_bytes > K::cpu::kNativeHBytes) {
                err = "worker CPU/GPU expert layouts differ";
                return false;
            }
        }
        if (wl.slot_stride <= gpu_budget - gpu_bytes) {
            gpu.push_back(p);
            gpu_bytes += wl.slot_stride;
        } else cpu.push_back(p);
    }
    if (!install_gpu_plan(ctx, gpu.data(), static_cast<uint32_t>(gpu.size()), ready, err)) return false;
    for (const auto p : cpu) {
        const auto bytes = ctx.layers[p.layer].slot_stride;
        if (bytes <= host_budget - host_bytes) host_bytes += bytes;
    }
    ctx.host_arena.resize(host_bytes);
    ctx.host_blob_of.assign(ctx.slot_of.size(), nullptr);
    size_t off = 0, host_experts = 0;
    for (const auto p : cpu) {
        const auto &wl = ctx.layers[p.layer];
        const auto idx = static_cast<size_t>(p.layer) * ctx.geom.n_expert + p.expert;
        ctx.slot_of[idx] = -2; // acknowledged CPU-owned; -1 remains unowned
        if (wl.slot_stride <= host_bytes - off) {
            auto *dst = ctx.host_arena.data() + off;
            pack_expert_blob(wl, p.expert, dst);
            ctx.host_blob_of[idx] = dst;
            off += wl.slot_stride;
            ++host_experts;
        }
    }
    if (!cpu.empty()) {
#if defined(STRATA_WORKER_DSV41)
        if (ctx.deepseek) {
            // Same native pool as the layer owner; one worker plus its caller
            // bounds this first tiered profile to two CPU threads.
            if (!ctx.deepseek_cpu_pool) ctx.deepseek_cpu_pool = std::make_unique<strata::ds4::Pool>(1);
        } else
#endif
        if (!ctx.cpu_pool) ctx.cpu_pool = std::make_unique<K::cpu::ExpertPool>(0, false);
    }
    ctx.loaded_count = n_pairs;
    ready.loaded_count = n_pairs; // acknowledge logical ownership, including mapped experts
    ready.load_time_us = us_since(start);
    std::fprintf(stderr, "strata-worker-tiers: owned=%u gpu=%zu ram=%zu mapped=%zu gpu_bytes=%zu ram_bytes=%zu\n",
                 n_pairs, gpu.size(), host_experts, cpu.size() - host_experts, gpu_bytes, host_bytes);
    return true;
}

bool execute_gpu_request(WorkerContext &ctx, const ExpertWorkerExecuteFixed &ef,
                            const ExpertWorkerAssignment *assigns, const float *acts, ExpertWorkerResultFixed &rf,
                            uint8_t *out_mask, float *out_sums, std::string &err) {
    const auto t0 = Clock::now();
    const int32_t N = static_cast<int32_t>(ctx.geom.n_embd);
    const int32_t n_exp = static_cast<int32_t>(ctx.geom.n_expert);

    rf = {};
    rf.layer = ef.layer;
    rf.n_tokens = ef.n_tokens;
    rf.n_embd = ef.n_embd;
    rf.n_assignments_executed = 0;
    rf.status = static_cast<uint32_t>(ExpertWorkerErrorCode::OK);

    std::memset(out_mask, 0, ef.n_tokens);
    std::memset(out_sums, 0, static_cast<size_t>(ef.n_tokens) * N * sizeof(float));

    if (ef.layer < 0 || ef.layer >= ctx.total_layers || ef.n_embd != static_cast<uint32_t>(N) || ef.n_tokens == 0 ||
        ef.n_tokens > static_cast<uint32_t>(kMaxTokens) || ef.n_assignments > static_cast<uint32_t>(kMaxCap)) {
        rf.status = static_cast<uint32_t>(ExpertWorkerErrorCode::MALFORMED_PACKET);
        rf.total_us = us_since(t0);
        return true;
    }

    if (ef.n_assignments == 0) {
        rf.total_us = us_since(t0);
        return true;
    }

    // Validate residency and group assignments by expert_id (ascending order or arrival order)
    struct Entry {
        uint16_t tok;
        uint16_t slot_k;
        int32_t expert;
        int32_t cache_slot;
        float weight;
        int32_t compact_row;
    };
    std::set<std::pair<uint16_t, uint16_t>> seen_slots;
    std::vector<Entry> entries(ef.n_assignments);
    std::vector<int32_t> group_experts;
    std::vector<int32_t> group_cache_slots;
    group_experts.reserve(ef.n_assignments);
    group_cache_slots.reserve(ef.n_assignments);

    for (uint32_t i = 0; i < ef.n_assignments; ++i) {
        const auto &a = assigns[i];
        if (a.token_idx >= ef.n_tokens || a.expert_id < 0 || a.expert_id >= n_exp || a.slot_idx >= ctx.top_k ||
            !std::isfinite(a.weight) || !seen_slots.emplace(a.token_idx, a.slot_idx).second) {
            rf.status = static_cast<uint32_t>(ExpertWorkerErrorCode::MALFORMED_PACKET);
            rf.total_us = us_since(t0);
            return true;
        }
        const int32_t cslot =
            ctx.slot_of[static_cast<size_t>(ef.layer) * static_cast<size_t>(n_exp) + static_cast<size_t>(a.expert_id)];
        if (cslot < 0) {
            rf.status = static_cast<uint32_t>(ExpertWorkerErrorCode::NONRESIDENT_EXPERT);
            rf.total_us = us_since(t0);
            return true;
        }
        entries[i] = {a.token_idx, a.slot_idx, a.expert_id, cslot, a.weight, -1};
        bool found = false;
        for (int32_t ge : group_experts) {
            if (ge == a.expert_id) {
                found = true;
                break;
            }
        }
        if (!found) {
            group_experts.push_back(a.expert_id);
            group_cache_slots.push_back(cslot);
        }
    }

    WorkerGroupMeta *gm = ctx.h_meta;
    const int32_t n_groups = static_cast<int32_t>(group_experts.size());
    gm->count = n_groups;
    for (uint32_t t = 0; t < ef.n_tokens; ++t)
        gm->red_cnt[t] = 0;

    int32_t compact = 0;
    for (int32_t g = 0; g < n_groups; ++g) {
        const int32_t ge = group_experts[static_cast<size_t>(g)];
        const int32_t cslot = group_cache_slots[static_cast<size_t>(g)];
        gm->ptr[g] = reinterpret_cast<unsigned long long>(ctx.d_arena + ctx.slot_offset[static_cast<size_t>(cslot)]);
        gm->start[g] = compact;
        for (uint32_t i = 0; i < ef.n_assignments; ++i) {
            if (entries[i].expert == ge) {
                entries[i].compact_row = compact;
                gm->dst[compact] = compact;
                gm->tok[compact] = entries[i].tok;
                ++compact;
            }
        }
    }
    gm->start[n_groups] = compact;

    // Populate per-token reduction table in original assignment order (preserves router pick order!)
    for (uint32_t i = 0; i < ef.n_assignments; ++i) {
        const uint16_t t = entries[i].tok;
        const int32_t c = gm->red_cnt[t];
        if (c >= kMaxTopK) {
            rf.status = static_cast<uint32_t>(ExpertWorkerErrorCode::MALFORMED_PACKET);
            rf.total_us = us_since(t0);
            return true;
        }
        gm->red_row[t * kMaxTopK + c] = entries[i].compact_row;
        gm->red_wt[t * kMaxTopK + c] = entries[i].weight;
        gm->red_cnt[t] = c + 1;
        out_mask[t] = 1;
    }

    const size_t act_bytes = static_cast<size_t>(ef.n_tokens) * N * sizeof(float);
    std::memcpy(ctx.h_x, acts, act_bytes);

    if (!check_cuda(cudaEventRecord(ctx.timing[0], ctx.stream), "stage timing", err))
        return false;
    if (!check_cuda(cudaMemcpyAsync(ctx.d_x, ctx.h_x, act_bytes, cudaMemcpyHostToDevice, ctx.stream), "copy d_x",
                    err) ||
        !check_cuda(
            cudaMemcpyAsync(ctx.d_meta, ctx.h_meta, sizeof(WorkerGroupMeta), cudaMemcpyHostToDevice, ctx.stream),
            "copy d_meta", err)) {
        rf.status = static_cast<uint32_t>(ExpertWorkerErrorCode::KERNEL_FAILURE);
        return false;
    }
    if (!check_cuda(cudaEventRecord(ctx.timing[1], ctx.stream), "kernel timing", err))
        return false;
    const WorkerLayerSource &wl = ctx.layers[static_cast<size_t>(ef.layer)];
#ifdef STRATA_KIMI2_LIBRARY
    if(ctx.kimi2){
        for(int g=0;g<n_groups;++g){
            const int first=gm->start[g],rows=gm->start[g+1]-first;
            std::vector<float> input(static_cast<size_t>(rows)*N);
            for(int i=0;i<rows;++i)std::copy_n(acts+static_cast<size_t>(gm->tok[first+i])*N,N,input.data()+static_cast<size_t>(i)*N);
            const auto result=strata::kimi2::run_cached_expert(reinterpret_cast<const uint8_t*>(gm->ptr[g]),wl.layout.gu_type,wl.layout.d_type,input);
            if(!check_cuda(cudaMemcpyAsync(ctx.d_parts+static_cast<size_t>(first)*N,result.data(),result.size()*4,cudaMemcpyHostToDevice,ctx.stream),"K2 expert parts",err) ||
               !check_cuda(cudaStreamSynchronize(ctx.stream),"K2 expert completion",err))return false;
        }
    }else
#endif
    {
    K::quantize_q8_1_rows(ctx.d_x, ef.n_tokens, N, ctx.d_q8, ctx.stream);
    K::native_expert_grouped(wl.layout, ctx.d_meta->ptr, ctx.d_meta->start, &ctx.d_meta->count, ctx.d_meta->dst,
                             ctx.d_meta->tok, n_groups, compact, ctx.d_q8, ctx.d_scratch, ctx.d_parts, ctx.stream);

    }

    k_reduce_weighted_parts<<<dim3((N + 255) / 256, ef.n_tokens), 256, 0, ctx.stream>>>(ctx.d_parts, ctx.d_meta,
                                                                                        ctx.d_sum, N);

    if (!check_cuda(cudaGetLastError(), "worker kernel launch", err)) {
        rf.status = static_cast<uint32_t>(ExpertWorkerErrorCode::KERNEL_FAILURE);
        return false;
    }
    if (!check_cuda(cudaEventRecord(ctx.timing[2], ctx.stream), "copy timing", err))
        return false;
    if (!check_cuda(cudaMemcpyAsync(ctx.h_sum, ctx.d_sum, act_bytes, cudaMemcpyDeviceToHost, ctx.stream), "copy h_sum",
                    err)) {
        rf.status = static_cast<uint32_t>(ExpertWorkerErrorCode::KERNEL_FAILURE);
        return false;
    }
    if (!check_cuda(cudaEventRecord(ctx.timing[3], ctx.stream), "completion timing", err))
        return false;
    // A failed GPU must not keep this service alive indefinitely. The primary has
    // its own shorter network deadline and recomputes its missing rows locally.
    cudaError_t se;
    while ((se = cudaStreamQuery(ctx.stream)) == cudaErrorNotReady) {
        if (us_since(t0) > 30000000) {
            std::fprintf(stderr, "strata-expert-worker: GPU request exceeded 30 s; terminating worker\n");
            std::_Exit(1);
        }
        std::this_thread::yield();
    }
    if (!check_cuda(se, "cudaStreamQuery", err)) {
        rf.status = static_cast<uint32_t>(ExpertWorkerErrorCode::KERNEL_FAILURE);
        return false;
    }
    std::memcpy(out_sums, ctx.h_sum, act_bytes);
    float elapsed = 0;
    if (!check_cuda(cudaEventElapsedTime(&elapsed, ctx.timing[0], ctx.timing[1]), "stage elapsed", err))
        return false;
    rf.stage_us = static_cast<uint64_t>(elapsed * 1000);
    if (!check_cuda(cudaEventElapsedTime(&elapsed, ctx.timing[1], ctx.timing[2]), "kernel elapsed", err))
        return false;
    rf.kernel_us = static_cast<uint64_t>(elapsed * 1000);
    if (!check_cuda(cudaEventElapsedTime(&elapsed, ctx.timing[2], ctx.timing[3]), "copy elapsed", err))
        return false;
    rf.copy_out_us = static_cast<uint64_t>(elapsed * 1000);
    rf.n_assignments_executed = ef.n_assignments;
    rf.total_us = us_since(t0);
    return true;
}

#if defined(STRATA_WORKER_DSV41)
/// Run assigned DeepSeek experts through the stock GPU or CPU row functions.
/// Requests preserve original router order; no model weights traverse the wire.
bool execute_deepseek_request(WorkerContext &ctx, const ExpertWorkerExecuteFixed &ef,
                              const ExpertWorkerAssignment *assigns, const float *acts, ExpertWorkerResultFixed &rf,
                              uint8_t *mask, float *sums, std::string &err) {
    namespace G = strata::ds4::gguf;
    const auto start = Clock::now();
    const size_t N = ctx.geom.n_embd;
    rf = {}; rf.layer = ef.layer; rf.n_tokens = ef.n_tokens; rf.n_embd = ef.n_embd;
    if (ef.layer < 0 || ef.layer >= ctx.total_layers || ef.n_embd != N || !ef.n_tokens ||
        ef.n_tokens > kMaxTokens || ef.n_assignments > ef.n_tokens * ctx.top_k) {
        err = "invalid DeepSeek request geometry"; return false;
    }
    std::set<std::pair<uint16_t, uint16_t>> seen;
    bool has_gpu=false,has_cpu=false;
    for (uint32_t i = 0; i < ef.n_assignments; ++i) {
        const auto a = assigns[i];
        if (a.token_idx >= ef.n_tokens || a.slot_idx >= ctx.top_k || a.expert_id < 0 ||
            a.expert_id >= ctx.geom.n_expert || !std::isfinite(a.weight) ||
            !seen.emplace(a.token_idx, a.slot_idx).second ||
            ctx.slot_of[static_cast<size_t>(ef.layer) * ctx.geom.n_expert + a.expert_id] == -1) {
            err = "invalid or unowned DeepSeek assignment"; return false;
        }
        if (ctx.slot_of[static_cast<size_t>(ef.layer) * ctx.geom.n_expert + a.expert_id] >= 0) has_gpu=true;
        else has_cpu=true;
    }
    std::fill(mask, mask + ef.n_tokens, 0);
    std::fill(sums, sums + ef.n_tokens * N, 0.0f);
    if (has_gpu) {
        std::memcpy(ctx.h_x, acts, ef.n_tokens * N * sizeof(float));
        if (!check_cuda(cudaMemcpyAsync(ctx.d_x, ctx.h_x, ef.n_tokens * N * sizeof(float), cudaMemcpyHostToDevice, ctx.stream), "DeepSeek activation upload", err)) return false;
    }
    const auto &source=ctx.layers[ef.layer];const auto &layout=source.layout;
    const auto &cpu=source.deepseek_cpu;
    std::vector<uint8_t> quant,hidden_quant;
    std::vector<float> hidden;
    if (has_cpu) {
        if (!ctx.deepseek_cpu_pool || ctx.arithmetic != ExpertWorkerArithmetic::DEEPSEEK_IQ_Q8K) {
            err="DeepSeek CPU expert execution lacks its qualified native tier setup";return false;
        }
        quant.resize(static_cast<size_t>(ef.n_tokens)*cpu.act_bytes);
        hidden_quant.resize(cpu.h_bytes);hidden.resize(static_cast<size_t>(ctx.geom.n_ff));
        for (uint32_t t=0;t<ef.n_tokens;++t)
            G::iq_cpu_quant(cpu.gu_act,acts+static_cast<size_t>(t)*N,static_cast<int>(N),quant.data()+static_cast<size_t>(t)*cpu.act_bytes);
    }
    strata::dsv41k::exl3::MoeArgs args;
    args.k = 1; args.n_embd = static_cast<int>(N); args.n_ff = static_cast<int>(ctx.geom.n_ff);
    args.limit = ctx.swiglu_limit; args.lay.fmt = G::kFmtMXFP4;
    args.lay.K[0] = args.lay.K[1] = layout.gu_type; args.lay.K[2] = layout.d_type;
    args.lay.trellis[0] = 0; args.lay.trellis[1] = layout.up_off; args.lay.trellis[2] = layout.down_off;
    args.plan_ptr = ctx.d_meta->ptr; args.plan_w = ctx.d_meta->red_wt;
    if (has_gpu && !G::prepare(ctx.stream)) { err = "DeepSeek GPU scratch initialization failed"; return false; }
    const int errors_before = G::launch_errors();
    for (uint32_t i = 0; i < ef.n_assignments; ++i) {
        const auto a = assigns[i];
        const int slot = ctx.slot_of[static_cast<size_t>(ef.layer) * ctx.geom.n_expert + a.expert_id];
        if (us_since(start)>30000000) {std::fprintf(stderr,"DeepSeek worker service deadline exceeded\n");std::_Exit(1);}
        if (slot<0) {
            const size_t index=static_cast<size_t>(ef.layer)*ctx.geom.n_expert+a.expert_id;
            const auto *packed=ctx.host_blob_of[index];
            const auto *gate=packed ? packed:source.src_g+static_cast<size_t>(a.expert_id)*source.src_gsz;
            const auto *up=packed ? packed+layout.up_off:source.src_u+static_cast<size_t>(a.expert_id)*source.src_gsz;
            const auto *down=packed ? packed+layout.down_off:source.src_d+static_cast<size_t>(a.expert_id)*source.src_dsz;
            const std::span<const uint8_t> activation(quant.data()+static_cast<size_t>(a.token_idx)*cpu.act_bytes,cpu.act_bytes);
            if (!strata::ds4::worker::execute_cpu_rows(cpu,*ctx.deepseek_cpu_pool,{gate,up,down},activation,
                    hidden,hidden_quant,std::span<float>(ctx.h_sum,N),ctx.swiglu_limit,err)) return false;
            if (packed) ++ctx.ram_assignments;else ++ctx.mapped_assignments;
        } else {
            ctx.h_meta->ptr[0] = reinterpret_cast<unsigned long long>(ctx.d_arena + ctx.slot_offset[slot]);
            ctx.h_meta->red_wt[0] = 1.0f;
            if (!check_cuda(cudaMemcpyAsync(ctx.d_meta, ctx.h_meta, sizeof(WorkerGroupMeta), cudaMemcpyHostToDevice, ctx.stream), "DeepSeek plan upload", err)) return false;
            G::moe_gate_up(args, ctx.d_x + a.token_idx * N, static_cast<float *>(ctx.d_scratch), ctx.stream);
            G::moe_down(args, static_cast<float *>(ctx.d_scratch), nullptr, ctx.d_sum, ctx.stream);
            if (G::launch_errors() != errors_before) { err = "DeepSeek expert kernel refused execution"; return false; }
            if (!check_cuda(cudaGetLastError(), "DeepSeek expert launch", err) ||
                !check_cuda(cudaMemcpyAsync(ctx.h_sum, ctx.d_sum, N * sizeof(float), cudaMemcpyDeviceToHost, ctx.stream), "DeepSeek result copy", err)) return false;
            cudaError_t status;
            while ((status = cudaStreamQuery(ctx.stream)) == cudaErrorNotReady) {
                if (us_since(start) > 30000000) { std::fprintf(stderr, "DeepSeek worker GPU deadline exceeded\n"); std::_Exit(1); }
                std::this_thread::yield();
            }
            if (!check_cuda(status, "DeepSeek completion", err)) return false;
            ++ctx.gpu_assignments;
        }
        mask[a.token_idx] = 1;
        for (size_t c = 0; c < N; ++c)
            sums[a.token_idx * N + c] = std::fma(a.weight, ctx.h_sum[c], sums[a.token_idx * N + c]);
    }
    rf.n_assignments_executed = ef.n_assignments;
    rf.total_us = us_since(start);
    rf.kernel_us = rf.total_us; // end-to-end service time, including transfers/CPU rows; not pure GPU time
    return true;
}
#endif

bool execute_worker_request(WorkerContext &ctx, const ExpertWorkerExecuteFixed &ef,
                            const ExpertWorkerAssignment *assigns, const float *acts, ExpertWorkerResultFixed &rf,
                            uint8_t *out_mask, float *out_sums, std::string &err) {
#if defined(STRATA_WORKER_DSV41)
    if (ctx.deepseek) return execute_deepseek_request(ctx, ef, assigns, acts, rf, out_mask, out_sums, err);
#endif
    if (ctx.expert_vram_mib < 0) return execute_gpu_request(ctx, ef, assigns, acts, rf, out_mask, out_sums, err);
    const auto start = Clock::now();
    const size_t N = ctx.geom.n_embd;
    if (ef.layer < 0 || ef.layer >= ctx.total_layers || ef.n_tokens == 0 || ef.n_tokens > kMaxTokens ||
        ef.n_embd != N || ef.n_assignments > kMaxCap) { err = "invalid tiered request geometry"; return false; }
    std::set<std::pair<uint16_t, uint16_t>> seen;
    std::vector<ExpertWorkerAssignment> gpu, cpu;
    for (uint32_t i = 0; i < ef.n_assignments; ++i) {
        const auto a = assigns[i];
        if (a.token_idx >= ef.n_tokens || a.slot_idx >= ctx.top_k || a.expert_id < 0 ||
            a.expert_id >= ctx.geom.n_expert || !std::isfinite(a.weight) ||
            !seen.emplace(a.token_idx, a.slot_idx).second) { err = "invalid tiered assignment"; return false; }
        const auto slot = ctx.slot_of[static_cast<size_t>(ef.layer) * ctx.geom.n_expert + a.expert_id];
        if (slot == -1) { err = "tiered request for unowned expert"; return false; }
        (slot >= 0 ? gpu : cpu).push_back(a);
    }
    auto ge = ef;
    ge.n_assignments = static_cast<uint32_t>(gpu.size());
    if (!execute_gpu_request(ctx, ge, gpu.data(), acts, rf, out_mask, out_sums, err)) return false;
    if (rf.status != 0) return true;
    ctx.gpu_assignments += gpu.size();
    if (!cpu.empty()) {
        const auto cpu_start = Clock::now();
        const auto &wl = ctx.layers[ef.layer];
        K::cpu::NativeFmt fmt;
        if (!K::cpu::native_fmt(wl.layout.gu_type, wl.layout.d_type, N, ctx.geom.n_ff, fmt, err)) return false;
        std::vector<uint8_t> quant(static_cast<size_t>(ef.n_tokens) * K::cpu::kNativeActBytes);
        for (uint32_t t = 0; t < ef.n_tokens; ++t)
            K::cpu::native_quant_act(fmt, acts + t * N, quant.data() + t * K::cpu::kNativeActBytes);
        std::vector<float> parts(cpu.size() * N);
        std::vector<K::cpu::NativeFmt> formats(cpu.size(), fmt);
        std::vector<K::cpu::ExpertJobMulti> jobs(cpu.size());
        for (size_t i = 0; i < cpu.size(); ++i) {
            const auto a = cpu[i];
            const size_t idx = static_cast<size_t>(ef.layer) * ctx.geom.n_expert + a.expert_id;
            const uint8_t *blob = ctx.host_blob_of[idx];
            if (blob) ++ctx.ram_assignments;
            else {
                ++ctx.mapped_assignments;
                blob = wl.src_g + static_cast<size_t>(a.expert_id) * wl.src_gsz;
                formats[i].up_off = static_cast<size_t>(reinterpret_cast<uintptr_t>(wl.src_u + static_cast<size_t>(a.expert_id) * wl.src_gsz) - reinterpret_cast<uintptr_t>(blob));
                formats[i].down_off = static_cast<size_t>(reinterpret_cast<uintptr_t>(wl.src_d + static_cast<size_t>(a.expert_id) * wl.src_dsz) - reinterpret_cast<uintptr_t>(blob));
            }
            jobs[i].blob = blob;
            jobs[i].fmt = &formats[i];
            jobs[i].nt = 1;
            jobs[i].nact[0] = quant.data() + a.token_idx * K::cpu::kNativeActBytes;
            jobs[i].out[0] = parts.data() + i * N;
        }
        for (size_t i = 0; i < jobs.size(); i += K::cpu::ExpertPool::kMaxSplitMulti)
            ctx.cpu_pool->run_split_multi_native(fmt, jobs.data() + i,
                static_cast<int>(std::min(jobs.size() - i, static_cast<size_t>(K::cpu::ExpertPool::kMaxSplitMulti))));
        for (size_t i = 0; i < cpu.size(); ++i) {
            const auto a = cpu[i];
            out_mask[a.token_idx] = 1;
            for (size_t c = 0; c < N; ++c)
                out_sums[a.token_idx * N + c] = std::fma(a.weight, parts[i * N + c], out_sums[a.token_idx * N + c]);
        }
        rf.kernel_us += us_since(cpu_start); // includes host quantization and reduction
    }
    rf.n_assignments_executed = ef.n_assignments;
    rf.total_us = us_since(start);
    return true;
}

bool serve_session(int cfd, WorkerContext &ctx, int timeout_ms, int idle_timeout_ms, bool verbose) {
    std::string err;
    const uint16_t version = ctx.deepseek ? kExpertWorkerArithmeticVersion : kExpertWorkerVersion;
    if (!expert_worker_set_socket_opts(cfd, timeout_ms, &err)) {
        std::fprintf(stderr, "strata-expert-worker: %s\n", err.c_str());
        return false;
    }
    ctx.close_cache(); // Each connection must install and acknowledge its own immutable plan.
    bool hello_ok = false, installed = false;
    uint64_t last_id = 0;
    ExpertWorkerFrameHeader hdr{};
    std::vector<uint8_t> payload;
    auto reject = [&](const char *why) {
        expert_worker_send_error(cfd, hdr.request_id, ExpertWorkerErrorCode::MALFORMED_PACKET, why, version);
        std::fprintf(stderr, "strata-expert-worker: rejected request: %s\n", why);
        return false;
    };
    while (expert_worker_recv_frame(cfd, hdr, payload, &err, timeout_ms, idle_timeout_ms)) {
        if (hdr.version != version) return reject("protocol version differs from the model adapter");
        if (hdr.request_id <= last_id)
            return reject("request IDs must increase");
        last_id = hdr.request_id;
        const auto type = static_cast<ExpertWorkerMsgType>(hdr.msg_type);
        if (type == ExpertWorkerMsgType::PING) {
            if (!payload.empty())
                return reject("PING must be empty");
            if (!expert_worker_send_frame(cfd, ExpertWorkerMsgType::PONG, hdr.request_id, nullptr, 0, &err, timeout_ms, version))
                return false;
            continue;
        }
        if (type == ExpertWorkerMsgType::HELLO) {
            ExpertWorkerHelloPayload hp{};
            uint32_t requested_arithmetic = 0;
            bool decoded = false;
            if (version == kExpertWorkerVersion) decoded = expert_wire_decode(payload,hp);
            else {
                ExpertWorkerHelloV2 v2{};
                decoded = expert_wire_decode(payload,v2); hp = v2.model; requested_arithmetic = v2.arithmetic;
            }
            if (hello_ok || !decoded)
                return reject("invalid or repeated HELLO");
            ExpertWorkerCapsPayload cp{};
            cp.model_fingerprint = ctx.fingerprint;
            cp.model_sha256 = ctx.model_sha256;
            cp.n_embd = ctx.geom.n_embd;
            cp.n_ff = ctx.geom.n_ff;
            cp.n_layers = ctx.geom.n_layers;
            cp.n_experts = ctx.geom.n_expert;
            cp.top_k = ctx.top_k;
            cp.endianness = 0x01020304u;
#if defined(STRATA_USE_HIP)
            cp.backend_id = 2;
#else
            cp.backend_id = 1;
#endif
            cp.max_tokens = kMaxTokens;
            cp.max_slots = ctx.total_layers * ctx.geom.n_expert;
            size_t free_b = 0, total_b = 0;
            if (!check_cuda(cudaMemGetInfo(&free_b, &total_b), "cudaMemGetInfo", err))
                return reject(err.c_str());
            cp.vram_free_bytes = free_b;
            cp.vram_total_bytes = total_b;
            std::snprintf(cp.device_name, sizeof(cp.device_name), "%s", ctx.device_name.c_str());
            hello_ok = hp.model_sha256 == cp.model_sha256 && hp.model_fingerprint == cp.model_fingerprint &&
                       hp.n_embd == cp.n_embd && hp.n_ff == cp.n_ff && hp.n_layers == cp.n_layers &&
                       hp.n_experts == cp.n_experts && hp.top_k == cp.top_k && hp.dtype == 0 && hp.reserved == 0 &&
                       hp.endianness == cp.endianness && requested_arithmetic == static_cast<uint32_t>(ctx.arithmetic);
            cp.status =
                static_cast<uint32_t>(hello_ok ? ExpertWorkerErrorCode::OK : ExpertWorkerErrorCode::MODEL_MISMATCH);
            const bool sent = version == kExpertWorkerVersion
                ? expert_worker_send_message(cfd, ExpertWorkerMsgType::CAPS, hdr.request_id, cp, &err, timeout_ms)
                : expert_worker_send_message(cfd, ExpertWorkerMsgType::CAPS, hdr.request_id,
                      ExpertWorkerCapsV2{cp,static_cast<uint32_t>(ctx.arithmetic)}, &err, timeout_ms, version);
            if (!sent)
                return false;
            if (!hello_ok) {
                std::fprintf(stderr, "strata-expert-worker: model SHA-256, geometry or arithmetic mismatch\n");
                return false;
            }
            continue;
        }
        if (!hello_ok)
            return reject("HELLO required");
        if (type == ExpertWorkerMsgType::INSTALL_PLAN) {
            if (installed)
                return reject("placement is immutable for a session");
            ExpertWireReader r{payload};
            uint32_t n = 0;
            r.get(n);
            if (!r.ok || n > ctx.slot_of.size() || payload.size() != 4 + static_cast<size_t>(n) * 8)
                return reject("invalid INSTALL_PLAN length");
            std::vector<ExpertWorkerPair> pairs(n);
            std::set<std::pair<int32_t, int32_t>> unique;
            for (auto &p : pairs) {
                r.get(p);
                if (!unique.emplace(p.layer, p.expert).second)
                    return reject("duplicate plan entry");
            }
            ExpertWorkerReadyPayload ready{};
            installed = install_worker_plan(ctx, pairs.data(), n, ready, err);
            if (!installed) {
                ctx.close_cache();
                std::fprintf(stderr, "strata-expert-worker: plan failed: %s\n", err.c_str());
            }
            if (!expert_worker_send_message(cfd, ExpertWorkerMsgType::READY, hdr.request_id, ready, &err, timeout_ms, version))
                return false;
            if (!installed)
                return false;
            if (ctx.expert_vram_mib >= 0)
                std::fprintf(stderr, "strata-expert-worker: READY owned %u experts (%llu bytes VRAM); GPU slots verified, CPU layouts checked, %llu us\n",
                             ready.loaded_count, (unsigned long long)ready.vram_used_bytes, (unsigned long long)ready.load_time_us);
            else std::fprintf(
                stderr,
                "strata-expert-worker: READY loaded %u experts (%llu bytes VRAM), verified every slot, %llu us\n",
                ready.loaded_count, (unsigned long long)ready.vram_used_bytes, (unsigned long long)ready.load_time_us);
            continue;
        }
        if (type != ExpertWorkerMsgType::EXECUTE || !installed)
            return reject("EXECUTE requires READY");
        ExpertWireReader r{payload};
        ExpertWorkerExecuteFixed ef{};
        r.get(ef);
        if (!r.ok || ef.layer < 0 || ef.layer >= ctx.total_layers || ef.n_tokens == 0 || ef.n_tokens > kMaxTokens ||
            ef.n_embd != ctx.geom.n_embd || ef.n_UniqueTokens_or_FullRows != ef.n_tokens || ef.flags ||
            ef.n_assignments > ef.n_tokens * ctx.top_k)
            return reject("invalid EXECUTE geometry");
        const size_t count = static_cast<size_t>(ef.n_tokens) * ef.n_embd;
        if (payload.size() != 24 + static_cast<size_t>(ef.n_assignments) * 12 + count * 4)
            return reject("invalid EXECUTE length");
        std::vector<ExpertWorkerAssignment> assigns(ef.n_assignments);
        std::vector<float> acts(count), sums(count);
        std::vector<uint8_t> mask(ef.n_tokens);
        for (auto &a : assigns)
            r.get(a);
        for (auto &v : acts) {
            r.get(v);
            if (!std::isfinite(v))
                return reject("nonfinite activation");
        }
        ExpertWorkerResultFixed rf{};
        const auto gpu_before=ctx.gpu_assignments,ram_before=ctx.ram_assignments,mapped_before=ctx.mapped_assignments;
        if (!execute_worker_request(ctx, ef, assigns.data(), acts.data(), rf, mask.data(), sums.data(), err)) {
            std::fprintf(stderr, "strata-expert-worker: expert execution failed: %s\n", err.c_str());
            expert_worker_send_error(cfd, hdr.request_id, ExpertWorkerErrorCode::KERNEL_FAILURE, err, version);
            std::_Exit(1); // Do not reuse a possibly poisoned expert context.
        }
        if (verbose && ctx.deepseek && ctx.expert_vram_mib>=0)
            std::fprintf(stderr,"strata-worker-tier-request: layer=%d tokens=%u assignments=%u gpu=%llu ram=%llu mapped=%llu\n",
                ef.layer,ef.n_tokens,rf.n_assignments_executed,static_cast<unsigned long long>(ctx.gpu_assignments-gpu_before),
                static_cast<unsigned long long>(ctx.ram_assignments-ram_before),static_cast<unsigned long long>(ctx.mapped_assignments-mapped_before));
        ExpertWireWriter w;
        w.put(rf);
        for (auto v : mask)
            w.put(v);
        for (auto v : sums)
            w.put(v);
        if (!expert_worker_send_frame(cfd, ExpertWorkerMsgType::RESULT, hdr.request_id, w.bytes.data(),
                                      static_cast<uint32_t>(w.bytes.size()), &err, timeout_ms, version))
            return false;
    }
    if (ctx.expert_vram_mib >= 0)
        std::fprintf(stderr, "strata-worker-tier-execution: gpu=%llu ram=%llu mapped=%llu\n",
                     (unsigned long long)ctx.gpu_assignments, (unsigned long long)ctx.ram_assignments,
                     (unsigned long long)ctx.mapped_assignments);
    if (verbose)
        std::fprintf(stderr, "strata-expert-worker: session closed: %s\n", err.c_str());
    return true;
}

} // namespace

int main(int argc, char **argv) {
    std::string model_path;
#ifdef STRATA_KIMI2_LIBRARY
    bool kimi2=false,kimi2_full=false;std::string kimi2_identity;
#endif
    std::string bind_ep = "127.0.0.1:19450";
    int device = 0;
    int timeout_ms = 30000;
    int idle_timeout_ms = 0; // opt-in first-byte wait; active frame deadlines stay unchanged
    int max_sessions = 0; // 0 = run indefinitely accepting sequential sessions
    bool verbose = false;
    int64_t expert_vram_mib = -1;
    size_t resident_budget_mib = 0;

    for (int i = 1; i < argc; ++i) {
        const std::string a = argv[i];
        auto next = [&]() -> std::string {
            if (i + 1 >= argc) {
                std::fprintf(stderr, "strata-expert-worker: missing value for %s\n", a.c_str());
                std::exit(2);
            }
            return argv[++i];
        };
#ifdef STRATA_KIMI2_LIBRARY
        if (a == "--kimi2-fixture") { model_path=next();kimi2=true; }
        else if(a=="--kimi2-model"){model_path=next();kimi2=true;kimi2_full=true;}
        else if(a=="--kimi2-identity"){kimi2_identity=next();}
        else
#endif
        if (a == "--model")
            model_path = next();
        else if (a == "--bind")
            bind_ep = next();
        else if (a == "--device")
            device = std::atoi(next().c_str());
        else if (a == "--timeout-ms")
            timeout_ms = std::atoi(next().c_str());
        else if (a == "--idle-timeout-ms") {
            const auto value = next();
            if (value.empty() || value.size() > 7 || value.find_first_not_of("0123456789") != std::string::npos ||
                std::stoull(value) == 0 || std::stoull(value) > 7200000) {
                std::fprintf(stderr, "strata-expert-worker: --idle-timeout-ms requires 1..7200000 milliseconds\n");
                return 2;
            }
            idle_timeout_ms = static_cast<int>(std::stoull(value));
        }
        else if (a == "--max-sessions")
            max_sessions = std::atoi(next().c_str());
        else if (a == "--expert-vram-mib" || a == "--resident-budget-mib") {
            const auto value = next();
            if (value.empty() || value.find_first_not_of("0123456789") != std::string::npos || value.size() > 7 ||
                std::stoull(value) > 1048576) {
                std::fprintf(stderr, "strata-expert-worker: %s requires 0..1048576 MiB\n", a.c_str());
                return 2;
            }
            if (a == "--expert-vram-mib") expert_vram_mib = static_cast<int64_t>(std::stoull(value));
            else resident_budget_mib = static_cast<size_t>(std::stoull(value));
        }
        else if (a == "--verbose")
            verbose = true;
        else {
            std::fprintf(
                stderr,
                "usage: strata-expert-worker --model GGUF [--bind IP:PORT]\n"
                "                            [--device 0] [--timeout-ms 30000] [--max-sessions N] [--verbose]\n"
                "                            [--idle-timeout-ms N]\n"
                "                            [--expert-vram-mib N [--resident-budget-mib N]]\n");
            return 2;
        }
    }

    if (resident_budget_mib && expert_vram_mib < 0) {
        std::fprintf(stderr, "strata-expert-worker: --resident-budget-mib requires --expert-vram-mib\n");
        return 2;
    }
    int timeout_limit = 600000;
#ifdef STRATA_KIMI2_LIBRARY
    if (kimi2_full) timeout_limit = 3600000;
#endif
    if (timeout_ms <= 0 || timeout_ms > timeout_limit || max_sessions < 0) {
        std::fprintf(stderr, "strata-expert-worker: invalid timeout or session limit\n");
        return 2;
    }
    if (model_path.empty()) {
        std::fprintf(stderr, "strata-expert-worker: --model <path.gguf> is required\n");
        return 2;
    }

    std::string bind_ip;
    uint16_t bind_port = 0;
    if (!expert_worker_parse_host_port(bind_ep, bind_ip, bind_port)) {
        std::fprintf(stderr, "strata-expert-worker: invalid --bind %s (expected <ip>:<port>)\n", bind_ep.c_str());
        return 2;
    }
    if (bind_ip == "0.0.0.0" && std::getenv("STRATA_WORKER_ALLOW_ANY") == nullptr) {
        std::fprintf(stderr,
                     "strata-expert-worker: refusing to bind 0.0.0.0 by default; specify private interface IP\n");
        return 2;
    }

    WorkerContext ctx;
    ctx.expert_vram_mib = expert_vram_mib;
    ctx.resident_budget_mib = resident_budget_mib;
#ifdef STRATA_KIMI2_LIBRARY
    ctx.kimi2=kimi2;ctx.kimi2_full=kimi2_full;ctx.kimi2_identity=kimi2_identity;
#endif
    std::string err;
    try {
        if (!init_worker_model(model_path, device, ctx, err)) {
            std::fprintf(stderr, "strata-expert-worker: init failed: %s\n", err.c_str());
            return 1;
        }
    } catch (const std::exception &ex) {
        std::fprintf(stderr, "strata-expert-worker: init failed: %s\n", ex.what());
        return 1;
    }

#ifdef STRATA_KIMI2_LIBRARY
    if(ctx.kimi2 && ctx.kimi2_full)std::fprintf(stderr,"K2 FULL GGUF MAPPINGS: all61 layer/expert tables addressable; no full hot-copy allocation\n");
    if(ctx.kimi2 && !ctx.kimi2_full)std::fprintf(stderr,"K2 PARTIAL FIXTURE: original selected expert slices only; F32 decoded-weight arithmetic; not a full checkpoint load\n");
#endif
    size_t free_b = 0, total_b = 0;
    if (!check_cuda(cudaMemGetInfo(&free_b, &total_b), "cudaMemGetInfo", err)) {
        std::fprintf(stderr, "strata-expert-worker: %s\n", err.c_str());
        return 1;
    }
    std::fprintf(stderr,
                 "strata-expert-worker: device %d (%s), %.2f / %.2f GiB VRAM free, model %s (%lld layers, %lld "
                 "experts, N=%lld, FF=%lld)\n",
                 device, ctx.device_name.c_str(), static_cast<double>(free_b) / (1ULL << 30),
                 static_cast<double>(total_b) / (1ULL << 30), model_path.c_str(),
                 static_cast<long long>(ctx.geom.n_layers), static_cast<long long>(ctx.geom.n_expert),
                 static_cast<long long>(ctx.geom.n_embd), static_cast<long long>(ctx.geom.n_ff));

    uint16_t actual_port = 0;
    int lfd = expert_worker_listen(bind_ip, bind_port, 4, &actual_port, &err);
    if (lfd < 0) {
        std::fprintf(stderr, "strata-expert-worker: listen(%s:%u) failed: %s\n", bind_ip.c_str(), bind_port,
                     err.c_str());
        return 1;
    }

    std::printf("WORKER_LISTENING %s:%u\n", bind_ip.c_str(), actual_port);
    std::fflush(stdout);

    int sessions = 0;
    while (max_sessions <= 0 || sessions < max_sessions) {
        struct sockaddr_in caddr{};
        socklen_t clen = sizeof(caddr);
        int cfd = ::accept(lfd, reinterpret_cast<struct sockaddr *>(&caddr), &clen);
        if (cfd < 0) {
            if (errno == EINTR)
                continue;
            std::fprintf(stderr, "strata-expert-worker: accept failed: %s\n", std::strerror(errno));
            expert_worker_close(lfd);
            return 1;
        }
        ++sessions;
        serve_session(cfd, ctx, timeout_ms, idle_timeout_ms, verbose);
        expert_worker_close(cfd);
    }

    expert_worker_close(lfd);
    return 0;
}
