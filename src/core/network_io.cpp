#include "strata/core/network_io.hpp"
#include <algorithm>
#include <cerrno>
#include <climits>
#include <cstring>
#if !defined(_WIN32)
#include <poll.h>
#include <sys/socket.h>
#endif

namespace strata::core::network_detail {
bool transfer(int fd, uint8_t *data, size_t size, bool sending,
              std::chrono::steady_clock::time_point deadline, std::string *error) {
#if !defined(_WIN32)
    if (fd < 0 || (size && !data)) {
        if (error) *error = "invalid network transfer buffer or socket";
        return false;
    }
    while (size) {
        const auto left = std::chrono::duration_cast<std::chrono::milliseconds>(
            deadline-std::chrono::steady_clock::now()).count();
        if (left <= 0) {
            if (error) *error = "network frame deadline exceeded";
            return false;
        }
        pollfd pfd{fd,static_cast<short>(sending ? POLLOUT : POLLIN),0};
        const int ready = ::poll(&pfd,1,static_cast<int>(std::min<int64_t>(left,INT_MAX)));
        if (ready < 0 && errno == EINTR) continue;
        if (ready <= 0) {
            if (error) *error = ready == 0 ? "network frame deadline exceeded" : std::strerror(errno);
            return false;
        }
        const ssize_t n = sending ? ::send(fd,data,size,MSG_NOSIGNAL|MSG_DONTWAIT)
                                  : ::recv(fd,data,size,MSG_DONTWAIT);
        if (n < 0 && (errno == EINTR || errno == EAGAIN || errno == EWOULDBLOCK)) continue;
        if (n <= 0) {
            if (error) *error = n == 0 ? "network peer disconnected" : std::strerror(errno);
            return false;
        }
        data += n; size -= static_cast<size_t>(n);
    }
    return true;
#else
    if (error) *error = "network transfers require POSIX sockets";
    return false;
#endif
}
} // namespace strata::core::network_detail
