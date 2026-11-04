#pragma once

#include "strata/core/expert_worker_proto.hpp"

#include <cstring>
#include <limits>
#include <type_traits>

namespace strata::core {

// V1 wire format is explicitly little endian, with IEEE-754 binary32 rows.
// Native object layout, padding and alignment never enter the byte stream.
static_assert(sizeof(float) == 4 && std::numeric_limits<float>::is_iec559);

template <class A, class T> void expert_wire_fields(A &a, T &v) {
    if constexpr (std::is_same_v<T, ExpertWorkerFrameHeader>)
        a(v.magic, v.version, v.msg_type, v.request_id, v.payload_len, v.flags);
    else if constexpr (std::is_same_v<T, ExpertWorkerHelloPayload>)
        a(v.model_fingerprint, v.model_sha256, v.n_embd, v.n_ff, v.n_layers, v.n_experts, v.top_k, v.dtype,
          v.endianness, v.reserved);
    else if constexpr (std::is_same_v<T, ExpertWorkerCapsPayload>)
        a(v.model_fingerprint, v.model_sha256, v.n_embd, v.n_ff, v.n_layers, v.n_experts, v.top_k, v.dtype,
          v.endianness, v.backend_id, v.max_tokens, v.max_slots, v.status, v.reserved, v.vram_free_bytes,
          v.vram_total_bytes, v.device_name);
    else if constexpr (std::is_same_v<T, ExpertWorkerPair>)
        a(v.layer, v.expert);
    else if constexpr (std::is_same_v<T, ExpertWorkerReadyPayload>)
        a(v.status, v.loaded_count, v.vram_used_bytes, v.vram_free_bytes, v.load_time_us);
    else if constexpr (std::is_same_v<T, ExpertWorkerAssignment>)
        a(v.token_idx, v.slot_idx, v.expert_id, v.weight);
    else if constexpr (std::is_same_v<T, ExpertWorkerExecuteFixed>)
        a(v.layer, v.n_tokens, v.n_embd, v.n_UniqueTokens_or_FullRows, v.n_assignments, v.flags);
    else if constexpr (std::is_same_v<T, ExpertWorkerResultFixed>)
        a(v.layer, v.n_tokens, v.n_embd, v.n_assignments_executed, v.status, v.reserved, v.stage_us, v.kernel_us,
          v.copy_out_us, v.total_us);
    else if constexpr (std::is_same_v<T, ExpertWorkerErrorPayload>)
        a(v.code, v.detail_len, v.message);
    else
        static_assert(!sizeof(T), "missing expert wire schema");
}

struct ExpertWireWriter {
    std::vector<uint8_t> bytes;
    template <class... T> void operator()(const T &...v) { (put(v), ...); }
    template <class T> void put(T v) {
        if constexpr (std::is_integral_v<T>) {
            auto u = static_cast<std::make_unsigned_t<T>>(v);
            for (size_t i = 0; i < sizeof(T); ++i)
                bytes.push_back(static_cast<uint8_t>(u >> (8 * i)));
        } else if constexpr (std::is_same_v<T, float>) {
            uint32_t u;
            std::memcpy(&u, &v, 4);
            put(u);
        } else
            expert_wire_fields(*this, v);
    }
    template <size_t N> void put(const char (&v)[N]) {
        for (char c : v)
            put(static_cast<uint8_t>(c));
    }
    template <size_t N> void put(const std::array<uint8_t, N> &v) {
        for (auto c : v)
            put(c);
    }
};

struct ExpertWireReader {
    const std::vector<uint8_t> &bytes;
    size_t pos = 0;
    bool ok = true;
    bool done() const { return ok && pos == bytes.size(); }
    template <class... T> void operator()(T &...v) { (get(v), ...); }
    template <class T> void get(T &v) {
        if constexpr (std::is_integral_v<T>) {
            if (!ok || bytes.size() - pos < sizeof(T)) {
                ok = false;
                return;
            }
            std::make_unsigned_t<T> u = 0;
            for (size_t i = 0; i < sizeof(T); ++i)
                u |= static_cast<std::make_unsigned_t<T>>(bytes[pos++]) << (8 * i);
            std::memcpy(&v, &u, sizeof(T));
        } else if constexpr (std::is_same_v<T, float>) {
            uint32_t u = 0;
            get(u);
            std::memcpy(&v, &u, 4);
        } else
            expert_wire_fields(*this, v);
    }
    template <size_t N> void get(char (&v)[N]) {
        for (char &c : v)
            get(c);
    }
    template <size_t N> void get(std::array<uint8_t, N> &v) {
        for (auto &c : v)
            get(c);
    }
};

template <class T> std::vector<uint8_t> expert_wire_encode(const T &v) {
    ExpertWireWriter w;
    w.put(v);
    return std::move(w.bytes);
}
template <class T> bool expert_wire_decode(const std::vector<uint8_t> &bytes, T &v) {
    ExpertWireReader r{bytes};
    r.get(v);
    return r.done();
}
template <class T>
bool expert_worker_send_message(int fd, ExpertWorkerMsgType type, uint64_t id, const T &v, std::string *err,
                                int timeout_ms = 5000) {
    auto bytes = expert_wire_encode(v);
    return expert_worker_send_frame(fd, type, id, bytes.data(), static_cast<uint32_t>(bytes.size()), err, timeout_ms);
}

} // namespace strata::core
