#pragma once

#include <chrono>
#include <cstddef>
#include <cstdint>
#include <string>

namespace strata::core::network_detail {

/// Transfer exactly `size` bytes on a POSIX stream socket by one absolute
/// deadline. Partial progress never renews the deadline. The caller owns and
/// closes the socket after failure; an incomplete frame cannot be retried.
bool transfer(int fd, uint8_t *data, size_t size, bool sending,
              std::chrono::steady_clock::time_point deadline, std::string *error);

} // namespace strata::core::network_detail
