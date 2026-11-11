#pragma once

#include "strata/core/deepseek_layer_boundary.hpp"

namespace strata::core {

/// Encode and send one DS4B capture under a single finite deadline. This is
/// capture transport only: it does not install a plan or commit engine state.
/// Close the socket on failure; never reuse a partially delivered stream.
bool send_deepseek_boundary(int fd, const DeepSeekBoundaryLayout &layout,
    const DeepSeekLayerBoundary &capture, int timeout_ms, std::string &error);

/// Receive one bounded DS4B capture into a fresh host value. Validate the header
/// before allocating the body, and validate full identity/layout before changing
/// `capture`. The caller must still check its local model's selection counts
/// before applying state. Close the socket after any failure.
bool receive_deepseek_boundary(int fd, const DeepSeekBoundaryLayout &layout,
    const DeepSeekBoundaryHeader &expected, DeepSeekLayerBoundary &capture,
    int timeout_ms, std::string &error);

} // namespace strata::core
