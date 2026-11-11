#pragma once
#include <string>
#include <cstdint>
#include <vector>
namespace strata::core { class NetworkExpertExecutor; }
namespace strata::kimi2 {
class Model;
/// Compressed K2 MLA history, owned by the executing layer.
struct MLAState {
    std::vector<float> latent;
    std::vector<float> rotary;
};
/// Execute a contiguous window of up to eight rows and append its compressed KV state.
std::vector<float> run_layer(int layer, const std::string& weights, const std::string& experts,
    const std::vector<float>& input, const std::string& capture_prefix, MLAState& state, core::NetworkExpertExecutor* remote=nullptr);
/// Execute a real layer directly from the shared model's existing packed GGUF mappings.
std::vector<float> run_layer(int layer, const Model& model, const std::vector<float>& input,
    const std::string& capture_prefix, MLAState& state, core::NetworkExpertExecutor* remote=nullptr);
/// Apply native final RMSNorm and the original output head, returning 163840 logits per row.
std::vector<float> logits(const Model& model, const std::vector<float>& hidden);
/// Execute a fresh-state K2 layer0/1 window of one to eight F32 hidden rows.
/// Layer0 is dense; layer1 uses the existing GPU cache and mapped CPU pool.
std::vector<float> run_layer(int layer, const std::string& weights, const std::string& experts,
                             const std::vector<float>& input, const std::string& capture_prefix);
/// Execute original packed K2 expert bytes already resident in an existing GPU slot.
/// Input/output rows use F32 decoded-weight arithmetic; no slot allocation or placement is performed.
std::vector<float> run_cached_expert(const uint8_t* device_blob, int gate_type, int down_type,
    const std::vector<float>& input);
/// Execute a K2 routed/shared MoE window of up to eight F32 rows with a fixed
/// expert0 GPU-cache plan and explicit DequantizedF32 CPU miss arithmetic.
std::vector<float> run_moe(const std::string& weights, const std::vector<float>& input,
                           const std::string& capture_prefix, core::NetworkExpertExecutor* remote=nullptr, int layer=1, const Model* model=nullptr);
}
