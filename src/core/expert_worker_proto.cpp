#include "strata/core/expert_worker_wire.hpp"
#include <charconv>
#include <chrono>

#include <algorithm>
#include <cctype>
#include <cerrno>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <fstream>
#include <set>
#include <sstream>

#if !defined(_WIN32)
#include <arpa/inet.h>
#include <fcntl.h>
#include <netdb.h>
#include <netinet/in.h>
#include <netinet/tcp.h>
#include <poll.h>
#include <sys/socket.h>
#include <sys/time.h>
#include <sys/types.h>
#include <sys/uio.h>
#include <unistd.h>
#endif

namespace strata::core {

uint64_t compute_model_fingerprint(uint32_t n_layers, uint32_t n_experts, uint32_t n_embd, uint32_t n_ff) {
    uint64_t h = 14695981039346656037ULL;
    auto mix = [&](uint32_t v) {
        for (int i = 0; i < 4; ++i) {
            h ^= static_cast<uint64_t>((v >> (i * 8)) & 0xFFu);
            h *= 1099511628211ULL;
        }
    };
    mix(kExpertWorkerMagic);
    mix(n_layers);
    mix(n_experts);
    mix(n_embd);
    mix(n_ff);
    return h;
}

bool expert_worker_parse_host_port(const std::string &endpoint, std::string &host, uint16_t &port) {
    const auto pos = endpoint.rfind(':');
    if (pos == std::string::npos || pos == 0 || pos + 1 >= endpoint.size()) {
        return false;
    }
    host = endpoint.substr(0, pos);
    if (host.size() >= 2 && host.front() == '[' && host.back() == ']') {
        host = host.substr(1, host.size() - 2);
    }
    char *endp = nullptr;
    const std::string port_text = endpoint.substr(pos + 1);
    const long p = std::strtol(port_text.c_str(), &endp, 10);
    if (endp == port_text.c_str() || *endp != '\0' || p <= 0 || p > 65535) {
        return false;
    }
    port = static_cast<uint16_t>(p);
    return true;
}

void expert_worker_close(int &fd) {
#if !defined(_WIN32)
    if (fd >= 0) {
        ::close(fd);
        fd = -1;
    }
#else
    fd = -1;
#endif
}

void expert_worker_shutdown(int fd) {
#if !defined(_WIN32)
    if (fd >= 0)
        ::shutdown(fd, SHUT_RDWR);
#endif
}

bool expert_worker_set_socket_opts(int fd, int timeout_ms, std::string *err) {
#if !defined(_WIN32)
    if (fd < 0) {
        if (err)
            *err = "invalid socket fd";
        return false;
    }
    int one = 1;
    if (::setsockopt(fd, IPPROTO_TCP, TCP_NODELAY, &one, sizeof(one)) != 0) {
        if (err)
            *err = std::string("setsockopt(TCP_NODELAY) failed: ") + std::strerror(errno);
        return false;
    }
#ifdef TCP_QUICKACK
    (void)::setsockopt(fd, IPPROTO_TCP, TCP_QUICKACK, &one, sizeof(one));
#endif
    if (timeout_ms > 0) {
        struct timeval tv{};
        tv.tv_sec = timeout_ms / 1000;
        tv.tv_usec = (timeout_ms % 1000) * 1000;
        if (::setsockopt(fd, SOL_SOCKET, SO_RCVTIMEO, &tv, sizeof(tv)) != 0) {
            if (err)
                *err = std::string("setsockopt(SO_RCVTIMEO) failed: ") + std::strerror(errno);
            return false;
        }
        if (::setsockopt(fd, SOL_SOCKET, SO_SNDTIMEO, &tv, sizeof(tv)) != 0) {
            if (err)
                *err = std::string("setsockopt(SO_SNDTIMEO) failed: ") + std::strerror(errno);
            return false;
        }
    }
    return true;
#else
    if (err)
        *err = "expert worker sockets not supported on Windows in v1";
    return false;
#endif
}

int expert_worker_connect(const std::string &host, uint16_t port, int timeout_ms, std::string *err) {
#if !defined(_WIN32)
    struct addrinfo hints{};
    hints.ai_flags = AI_NUMERICHOST; // DNS resolution cannot exceed the connection deadline.
    hints.ai_family = AF_INET;
    hints.ai_socktype = SOCK_STREAM;
    hints.ai_protocol = IPPROTO_TCP;

    char port_str[16];
    std::snprintf(port_str, sizeof(port_str), "%u", static_cast<unsigned>(port));

    struct addrinfo *res = nullptr;
    const int gai = ::getaddrinfo(host.c_str(), port_str, &hints, &res);
    if (gai != 0 || !res) {
        if (err)
            *err = std::string("getaddrinfo(") + host + ":" + port_str + ") failed: " + ::gai_strerror(gai);
        return -1;
    }

    int fd = -1;
    std::string last_err;
    for (struct addrinfo *p = res; p != nullptr; p = p->ai_next) {
        fd = ::socket(p->ai_family, p->ai_socktype, p->ai_protocol);
        if (fd < 0) {
            last_err = std::string("socket() failed: ") + std::strerror(errno);
            continue;
        }
        const int flags = ::fcntl(fd, F_GETFL, 0);
        if (flags < 0 || ::fcntl(fd, F_SETFL, flags | O_NONBLOCK) != 0) {
            last_err = "cannot set nonblocking socket";
            expert_worker_close(fd);
            continue;
        }
        int rc = ::connect(fd, p->ai_addr, p->ai_addrlen);
        if (rc != 0 && errno == EINPROGRESS) {
            struct pollfd pfd{};
            pfd.fd = fd;
            pfd.events = POLLOUT;
            rc = ::poll(&pfd, 1, timeout_ms > 0 ? timeout_ms : 5000);
            if (rc <= 0) {
                last_err = (rc == 0) ? "connect timed out" : (std::string("poll() failed: ") + std::strerror(errno));
                expert_worker_close(fd);
                continue;
            }
            int so_err = 0;
            socklen_t slen = sizeof(so_err);
            if (::getsockopt(fd, SOL_SOCKET, SO_ERROR, &so_err, &slen) != 0 || so_err != 0) {
                last_err = std::string("connect failed: ") + std::strerror(so_err ? so_err : errno);
                expert_worker_close(fd);
                continue;
            }
            rc = 0;
        }
        if (flags >= 0 && timeout_ms > 0) {
            (void)::fcntl(fd, F_SETFL, flags);
        }
        if (rc != 0) {
            last_err = std::string("connect() failed: ") + std::strerror(errno);
            expert_worker_close(fd);
            continue;
        }
        if (!expert_worker_set_socket_opts(fd, timeout_ms, &last_err)) {
            expert_worker_close(fd);
            continue;
        }
        break;
    }
    ::freeaddrinfo(res);
    if (fd < 0 && err)
        *err = last_err;
    return fd;
#else
    if (err)
        *err = "not supported on Windows";
    return -1;
#endif
}

int expert_worker_listen(const std::string &bind_ip, uint16_t port, int backlog, uint16_t *out_port, std::string *err) {
#if !defined(_WIN32)
    int fd = ::socket(AF_INET, SOCK_STREAM, IPPROTO_TCP);
    if (fd < 0) {
        if (err)
            *err = std::string("socket() failed: ") + std::strerror(errno);
        return -1;
    }
    int one = 1;
    (void)::setsockopt(fd, SOL_SOCKET, SO_REUSEADDR, &one, sizeof(one));

    struct sockaddr_in addr{};
    addr.sin_family = AF_INET;
    addr.sin_port = htons(port);
    if (::inet_pton(AF_INET, bind_ip.c_str(), &addr.sin_addr) != 1) {
        if (err)
            *err = "inet_pton failed for bind_ip: " + bind_ip;
        expert_worker_close(fd);
        return -1;
    }
    if (::bind(fd, reinterpret_cast<struct sockaddr *>(&addr), sizeof(addr)) != 0) {
        if (err)
            *err = std::string("bind(") + bind_ip + ") failed: " + std::strerror(errno);
        expert_worker_close(fd);
        return -1;
    }
    if (::listen(fd, backlog > 0 ? backlog : 4) != 0) {
        if (err)
            *err = std::string("listen() failed: ") + std::strerror(errno);
        expert_worker_close(fd);
        return -1;
    }
    if (out_port) {
        struct sockaddr_in bound{};
        socklen_t len = sizeof(bound);
        if (::getsockname(fd, reinterpret_cast<struct sockaddr *>(&bound), &len) == 0) {
            *out_port = ntohs(bound.sin_port);
        } else {
            *out_port = port;
        }
    }
    return fd;
#else
    if (err)
        *err = "not supported on Windows";
    return -1;
#endif
}

namespace {
using Deadline = std::chrono::steady_clock::time_point;
#if !defined(_WIN32)
bool transfer(int fd, uint8_t *data, size_t size, bool send, Deadline deadline, std::string *err) {
    while (size) {
        const auto left =
            std::chrono::duration_cast<std::chrono::milliseconds>(deadline - std::chrono::steady_clock::now()).count();
        if (left <= 0) {
            if (err)
                *err = "expert worker frame deadline exceeded";
            return false;
        }
        pollfd pfd{fd, static_cast<short>(send ? POLLOUT : POLLIN), 0};
        const int rc = ::poll(&pfd, 1, static_cast<int>(left));
        if (rc < 0 && errno == EINTR)
            continue;
        if (rc <= 0) {
            if (err)
                *err = rc == 0 ? "expert worker frame deadline exceeded" : std::strerror(errno);
            return false;
        }
        const ssize_t n =
            send ? ::send(fd, data, size, MSG_NOSIGNAL | MSG_DONTWAIT) : ::recv(fd, data, size, MSG_DONTWAIT);
        if (n < 0 && (errno == EINTR || errno == EAGAIN || errno == EWOULDBLOCK))
            continue;
        if (n <= 0) {
            if (err)
                *err = n == 0 ? "expert worker disconnected" : std::strerror(errno);
            return false;
        }
        data += n;
        size -= static_cast<size_t>(n);
    }
    return true;
}
#endif
} // namespace

bool expert_worker_send_frame(int fd, ExpertWorkerMsgType type, uint64_t req_id, const void *payload,
                              uint32_t payload_len, std::string *err, int timeout_ms, uint16_t version) {
#if !defined(_WIN32)
    if (payload_len > kExpertWorkerMaxPayloadBytes || (payload_len && !payload) || timeout_ms <= 0 ||
        (version != kExpertWorkerVersion && version != kExpertWorkerArithmeticVersion)) {
        if (err)
            *err = "invalid expert worker frame length or deadline";
        return false;
    }
    const ExpertWorkerFrameHeader hdr{
        kExpertWorkerMagic, version, static_cast<uint16_t>(type), req_id, payload_len, 0};
    auto bytes = expert_wire_encode(hdr);
    if (payload_len) {
        const auto *p = static_cast<const uint8_t *>(payload);
        bytes.insert(bytes.end(), p, p + payload_len);
    }
    return transfer(fd, bytes.data(), bytes.size(), true,
                    std::chrono::steady_clock::now() + std::chrono::milliseconds(timeout_ms), err);
#else
    if (err)
        *err = "network experts require POSIX sockets";
    return false;
#endif
}

bool expert_worker_recv_frame(int fd, ExpertWorkerFrameHeader &hdr, std::vector<uint8_t> &payload, std::string *err,
                              int timeout_ms) {
#if !defined(_WIN32)
    const auto deadline = std::chrono::steady_clock::now() + std::chrono::milliseconds(timeout_ms);
    std::vector<uint8_t> bytes(24);
    if (!transfer(fd, bytes.data(), bytes.size(), false, deadline, err))
        return false;
    if (!expert_wire_decode(bytes, hdr) || hdr.magic != kExpertWorkerMagic ||
        (hdr.version != kExpertWorkerVersion && hdr.version != kExpertWorkerArithmeticVersion) ||
        hdr.flags != 0 || hdr.payload_len > kExpertWorkerMaxPayloadBytes || hdr.msg_type < 1 || hdr.msg_type > 9) {
        if (err)
            *err = "invalid expert worker frame magic, version, type, flags or length";
        return false;
    }
    payload.resize(hdr.payload_len);
    return transfer(fd, payload.data(), payload.size(), false, deadline, err);
#else
    if (err)
        *err = "network experts require POSIX sockets";
    return false;
#endif
}

bool expert_worker_send_error(int fd, uint64_t req_id, ExpertWorkerErrorCode code, const std::string &msg, uint16_t version) {
    ExpertWorkerErrorPayload ep{};
    ep.code = static_cast<uint32_t>(code);
    const size_t n = std::min(msg.size(), sizeof(ep.message) - 1);
    ep.detail_len = static_cast<uint32_t>(n);
    std::memcpy(ep.message, msg.data(), n);
    return expert_worker_send_message(fd, ExpertWorkerMsgType::ERROR_MSG, req_id, ep, nullptr, 5000, version);
}

bool load_expert_placement_plan(const std::string &path, int n_layers, int n_experts,
                                std::vector<ExpertWorkerPair> &out_pairs, std::string *err) {
    std::ifstream in(path);
    if (!in.is_open()) {
        if (err)
            *err = "failed to open placement plan file: " + path;
        return false;
    }
    out_pairs.clear();
    std::set<std::pair<int32_t, int32_t>> seen;

    std::string line;
    int line_no = 0;
    while (std::getline(in, line)) {
        ++line_no;
        const auto hash_pos = line.find('#');
        if (hash_pos != std::string::npos) {
            line.resize(hash_pos);
        }
        // Replace ':' and ',' with spaces, except keep '-' for ranges
        for (char &c : line) {
            if (c == ':' || c == ',')
                c = ' ';
        }
        std::istringstream iss(line);
        iss >> std::ws;
        if (iss.eof())
            continue;
        int layer = -1;
        if (!(iss >> layer)) {
            if (err)
                *err = "placement plan line " + std::to_string(line_no) + ": invalid layer";
            return false;
        }
        if (n_layers > 0 && (layer < 0 || layer >= n_layers)) {
            if (err)
                *err = "placement plan line " + std::to_string(line_no) + ": layer " + std::to_string(layer) +
                       " out of range";
            return false;
        }
        std::string tok;
        int added_on_line = 0;
        while (iss >> tok) {
            if (tok == "*" || tok == "all") {
                if (n_experts <= 0) {
                    if (err)
                        *err = "placement plan line " + std::to_string(line_no) +
                               ": wildcard '*' requires known n_experts";
                    return false;
                }
                for (int e = 0; e < n_experts; ++e) {
                    if (seen.insert({layer, e}).second) {
                        out_pairs.push_back({layer, e});
                    }
                    ++added_on_line;
                }
                continue;
            }
            const auto dash = tok.find('-');
            if (dash != std::string::npos && dash > 0 && dash + 1 < tok.size()) {
                char *e1 = nullptr;
                char *e2 = nullptr;
                const std::string first = tok.substr(0, dash), last = tok.substr(dash + 1);
                const long a = std::strtol(first.c_str(), &e1, 10);
                const long b = std::strtol(last.c_str(), &e2, 10);
                if (!e1 || *e1 != '\0' || !e2 || *e2 != '\0' || a < 0 || b < a || (n_experts > 0 && b >= n_experts)) {
                    if (err)
                        *err =
                            "placement plan line " + std::to_string(line_no) + ": invalid expert range '" + tok + "'";
                    return false;
                }
                for (long e = a; e <= b; ++e) {
                    if (seen.insert({layer, static_cast<int32_t>(e)}).second) {
                        out_pairs.push_back({layer, static_cast<int32_t>(e)});
                    }
                    ++added_on_line;
                }
            } else {
                char *endp = nullptr;
                const long e = std::strtol(tok.c_str(), &endp, 10);
                if (!endp || *endp != '\0' || e < 0 || (n_experts > 0 && e >= n_experts)) {
                    if (err)
                        *err = "placement plan line " + std::to_string(line_no) + ": invalid expert id '" + tok + "'";
                    return false;
                }
                if (seen.insert({layer, static_cast<int32_t>(e)}).second) {
                    out_pairs.push_back({layer, static_cast<int32_t>(e)});
                }
                ++added_on_line;
            }
        }
        if (added_on_line == 0) {
            if (err)
                *err = "placement plan line " + std::to_string(line_no) + ": missing expert id(s) for layer " +
                       std::to_string(layer);
            return false;
        }
    }
    if (in.bad()) {
        if (err)
            *err = "failed reading placement plan: " + path;
        return false;
    }
    return true;
}

bool save_expert_placement_plan(const std::string &path, const std::vector<ExpertWorkerPair> &pairs, std::string *err) {
    std::ofstream out(path);
    if (!out.is_open()) {
        if (err)
            *err = "failed to write placement plan file: " + path;
        return false;
    }
    out << "# Strata distributed expert placement plan (" << pairs.size() << " layer:expert pairs)\n";
    for (const auto &p : pairs) {
        out << p.layer << " " << p.expert << "\n";
    }
    return out.good();
}

} // namespace strata::core
