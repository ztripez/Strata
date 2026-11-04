#pragma once

#include "strata/artifact/gguf_reader.hpp"
#include <array>

namespace strata::core {
// SHA-256 of the mapped GGUF bytes, in shard order. Hashes the exact mapping used
// for expert loads; no cached hashes, path names or geometry-only identities.
std::array<uint8_t, 32> expert_worker_model_sha256(const strata::GgufModel &model);
} // namespace strata::core
