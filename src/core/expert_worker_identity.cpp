#include "strata/core/expert_worker_identity.hpp"
#include <memory>
#include <cstdio>
#include <cstdlib>
#include <openssl/evp.h>
#include <stdexcept>

namespace strata::core {
std::array<uint8_t, 32> expert_worker_spans_sha256(
    const std::vector<std::pair<const uint8_t *, size_t>> &spans) {
    std::unique_ptr<EVP_MD_CTX, decltype(&EVP_MD_CTX_free)> ctx(EVP_MD_CTX_new(), EVP_MD_CTX_free);
    if (!ctx || EVP_DigestInit_ex(ctx.get(), EVP_sha256(), nullptr) != 1)
        throw std::runtime_error("expert worker SHA-256 initialization failed");
    for (const auto &[data, size] : spans) {
        if ((size && !data) || EVP_DigestUpdate(ctx.get(), data, size) != 1)
            throw std::runtime_error("expert worker SHA-256 update failed");
    }
    std::array<uint8_t, 32> result{};
    unsigned int n = 0;
    if (EVP_DigestFinal_ex(ctx.get(), result.data(), &n) != 1 || n != result.size())
        throw std::runtime_error("expert worker SHA-256 finalization failed");
    return result;
}
std::array<uint8_t, 32> expert_worker_model_sha256(const strata::GgufModel &model) {
    if (const char *trusted = std::getenv("STRATA_TRUST_VERIFIED_MODEL_SHA256")) {
        const std::string hex(trusted);
        if (hex.size() != 64) throw std::runtime_error("trusted model SHA-256 must contain 64 hex digits");
        const auto digit = [](char value) -> uint8_t {
            if (value >= '0' && value <= '9') return static_cast<uint8_t>(value - '0');
            if (value >= 'a' && value <= 'f') return static_cast<uint8_t>(value - 'a' + 10);
            if (value >= 'A' && value <= 'F') return static_cast<uint8_t>(value - 'A' + 10);
            throw std::runtime_error("trusted model SHA-256 contains a non-hex digit");
        };
        std::array<uint8_t, 32> result{};
        for (size_t i = 0; i < result.size(); ++i)
            result[i] = static_cast<uint8_t>((digit(hex[2 * i]) << 4) | digit(hex[2 * i + 1]));
        std::fprintf(stderr, "strata-artifact-identity: mode=trusted-prior-verification; mapped bytes NOT rehashed\n");
        return result;
    }
    std::vector<std::pair<const uint8_t *, size_t>> spans;
    for (size_t i = 0; i < model.size(); ++i) {
        const auto &shard = model.shard(i);
        spans.emplace_back(shard.file_data(), shard.file_size());
    }
    return expert_worker_spans_sha256(spans);
}
} // namespace strata::core
