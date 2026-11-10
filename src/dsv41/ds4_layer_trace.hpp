#pragma once

#include "ds4_layer_process.hpp"
#include "strata/core/deepseek_layer_rpc.hpp"
#include <cstdio>

namespace strata::ds4::network {

/// Caller-side, nonoverlapping Model::feed spans. RPC already includes the
/// suffix's native computation; these are not independent costs to add again.
struct PrefixWindowTiming {
    uint64_t feed_us=0,before_rpc_us=0,rpc_us=0,after_rpc_us=0;
    core::DeepSeekLayerTiming transaction;
};

/// Keep a complete multi-part record together on the shared stderr stream,
/// including during exceptions. This does not serialize native computation.
class WindowTraceLock {
public:
    explicit WindowTraceLock(FILE *out):out_(out) {require(out_!=nullptr,"missing window trace stream");::flockfile(out_);}
    ~WindowTraceLock() {::funlockfile(out_);}
    WindowTraceLock(const WindowTraceLock&)=delete;
    WindowTraceLock &operator=(const WindowTraceLock&)=delete;
private:
    FILE *out_;
};

/// Emit the shared actual identity of one observed native window. Only numeric
/// fields, fixed role labels and hexadecimal digests are written as JSON.
inline void window_trace_header(FILE *out,const char *role,const ProcessConfig &c,
    const Digest &binary,const core::DeepSeekBoundaryHeader &h,uint64_t incarnation) {
    require(out!=nullptr,"missing window trace stream");
    require(std::fprintf(out,
        "FABRIC_WINDOW {\"schema\":\"strata-native-window-cost/1\",\"role\":\"%s\",\"model_sha256\":\"%s\",\"pack_sha256\":\"%s\",\"plan_sha256\":\"%s\",\"binary_sha256\":\"%s\",\"incarnation\":%llu,\"epoch\":%llu,\"session\":%llu,\"sequence\":%llu,\"position\":%llu,\"rows\":%u,\"cut\":%u,\"cpu_threads_limit\":%d,\"expert_vram_mib\":%lld",
        role,hex(h.model).c_str(),hex(h.pack).c_str(),hex(h.plan).c_str(),hex(binary).c_str(),
        static_cast<unsigned long long>(incarnation),static_cast<unsigned long long>(h.epoch),static_cast<unsigned long long>(h.session),
        static_cast<unsigned long long>(h.sequence),static_cast<unsigned long long>(h.position),h.rows,h.next_layer,
        c.cpu_threads,static_cast<long long>(c.expert_vram_mib))>=0,"window trace write failed");
}

/// Write one fully acknowledged prefix feed. Microsecond rounding can leave
/// at most two microseconds between the whole feed and its three summed spans.
/// Print time is outside these spans. A trace I/O error fails the caller.
inline void prefix_window_trace(FILE *out,const ProcessConfig &c,const Digest &binary,
    const core::DeepSeekBoundaryHeader &h,uint64_t incarnation,const PrefixWindowTiming &t) {
    WindowTraceLock lock(out);
    window_trace_header(out,"prefix",c,binary,h,incarnation);
    require(std::fprintf(out,
        ",\"feed_us\":%llu,\"before_rpc_us\":%llu,\"rpc_us\":%llu,\"after_rpc_us\":%llu,\"transaction\":{\"total_us\":%llu,\"execute_us\":%llu,\"ack_us\":%llu,\"tx_bytes\":%llu,\"rx_bytes\":%llu}}\n",
        static_cast<unsigned long long>(t.feed_us),static_cast<unsigned long long>(t.before_rpc_us),
        static_cast<unsigned long long>(t.rpc_us),static_cast<unsigned long long>(t.after_rpc_us),
        static_cast<unsigned long long>(t.transaction.total_us),static_cast<unsigned long long>(t.transaction.execute_us),
        static_cast<unsigned long long>(t.transaction.ack_us),static_cast<unsigned long long>(t.transaction.tx_bytes),
        static_cast<unsigned long long>(t.transaction.rx_bytes))>=0 && std::fflush(out)==0,"window trace write/flush failed");
}

/// Write the suffix's replay plus host-logit-readback span. This work is already
/// inside the prefix RPC duration. It excludes result encoding and socket I/O.
inline void suffix_window_trace(FILE *out,const ProcessConfig &c,const Digest &binary,
    const core::DeepSeekBoundaryHeader &h,uint64_t incarnation,uint64_t native_us) {
    WindowTraceLock lock(out);
    window_trace_header(out,"suffix",c,binary,h,incarnation);
    require(std::fprintf(out,",\"native_us\":%llu}\n",static_cast<unsigned long long>(native_us))>=0 &&
            std::fflush(out)==0,"window trace write/flush failed");
}
} // namespace strata::ds4::network
