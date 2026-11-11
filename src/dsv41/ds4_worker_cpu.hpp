#pragma once

#include "ds4_gguf_cpu.hpp"
#include "ds4_pool.hpp"
#include <algorithm>
#include <array>
#include <cmath>
#include <span>

namespace strata::ds4::worker {

/// Dispatch one unweighted expert through the existing DeepSeek CPU row functions.
/// Matrix pointers borrow validated gate/up/down GGUF views or one packed RAM
/// blob. The caller owns all scratch, quantizes the input with format.gu_act,
/// preserves router reduction order and supervises stalled backing-file reads.
/// Pool bands keep the native row arithmetic unchanged. Invalid scratch/layout
/// arguments fail before mutating any output; this function owns no residency.
inline bool execute_cpu_rows(const gguf::IqCpu &format, Pool &pool,
    const std::array<const uint8_t*,3> &matrices, std::span<const uint8_t> activation,
    std::span<float> hidden, std::span<uint8_t> quantized_hidden,
    std::span<float> output, float limit, std::string &error) {
    if (format.n_embd<=0 || format.n_ff<=0 || !format.act_bytes || !format.h_bytes ||
        !std::all_of(matrices.begin(),matrices.end(),[](const auto *p){return p!=nullptr;}) ||
        activation.size()!=format.act_bytes || hidden.size()!=static_cast<size_t>(format.n_ff) ||
        quantized_hidden.size()!=format.h_bytes || output.size()!=static_cast<size_t>(format.n_embd) ||
        !std::isfinite(limit) || limit<0) {
        error="invalid DeepSeek CPU expert views or scratch";return false;
    }
    constexpr int band=64;
    const void *input[1]={activation.data()};float *intermediate[1]={hidden.data()};
    pool.run((format.n_ff+band-1)/band,[&](int job) {
        gguf::iq_cpu_gate_up(format,matrices[0],matrices[1],input,1,intermediate,
            job*band,std::min(format.n_ff,(job+1)*band),limit);
    });
    gguf::iq_cpu_quant(format.d_act,hidden.data(),format.n_ff,quantized_hidden.data());
    const void *down_input[1]={quantized_hidden.data()};float *result[1]={output.data()};
    pool.run((format.n_embd+band-1)/band,[&](int job) {
        gguf::iq_cpu_down(format,matrices[2],down_input,1,result,
            job*band,std::min(format.n_embd,(job+1)*band));
    });
    return true;
}

} // namespace strata::ds4::worker
