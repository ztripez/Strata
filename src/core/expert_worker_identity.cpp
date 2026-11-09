#include "strata/core/expert_worker_identity.hpp"
#include <memory>
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
    std::vector<std::pair<const uint8_t *, size_t>> spans;
    for (size_t i = 0; i < model.size(); ++i) {
        const auto &shard = model.shard(i);
        spans.emplace_back(shard.file_data(), shard.file_size());
    }
    return expert_worker_spans_sha256(spans);
}
} // namespace strata::core
