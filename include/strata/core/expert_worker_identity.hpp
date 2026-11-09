#pragma once

#include "strata/artifact/gguf_reader.hpp"
#include <array>
#include <cstddef>
#include <utility>
#include <vector>

namespace strata::core {
/// SHA-256 of the supplied immutable memory spans in order. Null nonempty spans
/// are rejected. Used to bind native loaded mappings rather than reopen paths.
std::array<uint8_t, 32> expert_worker_spans_sha256(
    const std::vector<std::pair<const uint8_t *, size_t>> &spans);
// SHA-256 of the mapped GGUF bytes, in shard order. Hashes the exact mapping used
// for expert loads; no cached hashes, path names or geometry-only identities.
std::array<uint8_t, 32> expert_worker_model_sha256(const strata::GgufModel &model);
} // namespace strata::core
