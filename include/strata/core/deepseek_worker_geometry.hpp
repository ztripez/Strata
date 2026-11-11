#pragma once

#include "strata/artifact/gguf_reader.hpp"
#include <cmath>
#include <stdexcept>

namespace strata::core {
/// Validated geometry for the first DeepSeek network adapter. Deliberately
/// restricted to the pinned Flash pack until other shapes are qualified.
struct DeepSeekWorkerGeometry {
    int layers, experts, hidden, intermediate, picks;
    float limit = 0;
    explicit DeepSeekWorkerGeometry(const strata::GgufModel &model) {
        const auto &meta = model.meta();
        const auto *arch = meta.get("general.architecture");
        if (!arch || arch->s != "deepseek41") throw std::runtime_error("expected deepseek41 GGUF");
        auto number = [&](const char *name) {
            const auto *v = meta.get(std::string("deepseek41.") + name);
            if (!v || !v->is_num() || !std::isfinite(v->num()) || v->num() < 1 || v->num() > 65536 ||
                std::floor(v->num()) != v->num()) throw std::runtime_error(std::string("invalid DeepSeek key: ") + name);
            return static_cast<int>(v->num());
        };
        layers = number("block_count"); experts = number("expert_count");
        hidden = number("embedding_length"); intermediate = number("expert_feed_forward_length");
        picks = number("expert_used_count");
        if (layers != 40 || experts != 384 || hidden != 5120 || intermediate != 2304 || picks != 6)
            throw std::runtime_error("DeepSeek network adapter only qualifies Flash 40/384/5120/2304/top-6");
        const auto *clamp = meta.get("deepseek41.swiglu_clamp_exp");
        if (!clamp || clamp->type != strata::MetaType::ARRAY || clamp->items.size() != static_cast<size_t>(layers))
            throw std::runtime_error("missing DeepSeek per-layer SwiGLU clamp");
        limit = static_cast<float>(clamp->items.front().num());
        for (const auto &v : clamp->items)
            if (!v.is_num() || !std::isfinite(v.num()) || v.num() < 0 || std::abs(v.num() - limit) > 1e-6)
                throw std::runtime_error("DeepSeek network adapter requires one finite uniform SwiGLU clamp");
    }
};
} // namespace strata::core
