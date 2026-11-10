#pragma once

#include "strata/core/deepseek_layer_session.hpp"
#include <chrono>
#include <mutex>

namespace strata::core {

/// Immutable contract for one suffix service. The identity has zero session,
/// sequence, position and rows; next_layer names the cut. Layout and arithmetic
/// come from the loaded native model, not from the peer's request.
struct DeepSeekLayerContract {
    DeepSeekBoundaryHeader identity;
    DeepSeekBoundaryLayout layout;
    uint32_t effective_experts = 0;
    uint32_t arithmetic = 0; // ExpertWorkerArithmetic; initially Q8_K only
};
/// Actual worker capability receipt. Incarnation changes on process restart;
/// last_session fences earlier sessions within this worker incarnation.
struct DeepSeekLayerCapabilities {
    DeepSeekLayerContract contract;
    std::array<uint8_t,32> binary_sha256{};
    uint32_t backend = 0; // 1 CUDA, 2 HIP
    uint64_t incarnation = 0, last_session = 0;
};

/// Separately versioned DS4L framing. These internal helpers are exposed for
/// adapters and fault tests; expert-v1/v2 fields are never reinterpreted.
namespace deepseek_layer_rpc_detail {
enum class Type : uint16_t { discover=1, capabilities, begin, ready, execute, result, acknowledge, acknowledged, close, closed, error };
struct Frame { Type type{}; uint64_t request_id=0; std::vector<uint8_t> payload; };
bool send(int fd,const Frame &frame,int timeout_ms,std::string &error);
bool receive(int fd,Frame &frame,int timeout_ms,std::string &error);
std::vector<uint8_t> contract_bytes(const DeepSeekLayerContract &contract);
std::vector<uint8_t> capability_bytes(const DeepSeekLayerCapabilities &capabilities);
DeepSeekLayerCapabilities capabilities(const std::vector<uint8_t> &bytes);
}

/// Serial suffix service: one connection/session and one unacknowledged window.
/// Preparation must reset/fence native Model and verify effective settings before
/// returning true. The executor must independently bound native computation;
/// socket deadlines cannot interrupt a GPU kernel or a filesystem read.
class DeepSeekLayerService {
public:
    using Prepare = std::function<bool(const DeepSeekBoundaryHeader&,std::string&)>;
    DeepSeekLayerService(DeepSeekLayerCapabilities capabilities,Prepare prepare,DeepSeekLayerSession::Executor executor);
    DeepSeekLayerService(const DeepSeekLayerService&) = delete;
    DeepSeekLayerService &operator=(const DeepSeekLayerService&) = delete;
    /// Serve one connection through an explicit CLOSE, or return a visible
    /// protocol/transport/native failure. Caller owns/closes fd and must abandon
    /// the native session after failure. Never transparently retry a window.
    bool serve(int fd,int timeout_ms,std::string &error);
    const DeepSeekLayerCapabilities &capabilities() const { return capabilities_; }
private:
    DeepSeekLayerCapabilities capabilities_;
    Prepare prepare_;
    DeepSeekLayerSession::Executor executor_;
};

/// Persistent numeric-IP TCP client. A failed transaction closes this channel;
/// there is no reconnect/retry into possibly committed layer state.
class DeepSeekLayerClient {
public:
    DeepSeekLayerClient() = default;
    ~DeepSeekLayerClient();
    DeepSeekLayerClient(const DeepSeekLayerClient&) = delete;
    DeepSeekLayerClient &operator=(const DeepSeekLayerClient&) = delete;
    /// Discover and verify the worker contract and pinned executable before BEGIN.
    bool connect(const std::string &endpoint,const DeepSeekLayerContract &expected,
                 const std::array<uint8_t,32> &worker_binary,int timeout_ms,std::string &error);
    /// Establish a fresh monotonically fenced native session for this incarnation.
    bool begin(uint64_t session,std::string &error);
    /// Exchange one capture/result/ACK under one transaction deadline. Publish
    /// result only after ACKED; on any failure the result is unchanged.
    bool execute(const DeepSeekLayerBoundary &capture,uint32_t logits_rows,DeepSeekLayerResult &result,std::string &error);
    /// End a fully acknowledged session. Failure closes the channel too.
    bool close(std::string &error);
    /// Interrupt an established channel from a cancellation thread. The request
    /// is latched; pending transactions fail without publishing output or retry.
    /// TCP connection setup still has its declared connect deadline. Only this
    /// method may run concurrently with other operations. Join cancellation
    /// callers before destruction or reusing the client for another connect().
    /// Interruption does not prove the remote executor has stopped; its session
    /// must reset.
    void interrupt();
    const DeepSeekLayerCapabilities &capabilities() const { return capabilities_; }
    /// Whether the owned channel is open and has not been interrupted.
    bool healthy() const { std::lock_guard lock(socket_mutex_);return fd_>=0 && !interrupted_; }
private:
    bool fail(const std::string &reason,std::string &error);
    bool exchange(deepseek_layer_rpc_detail::Type request,const std::vector<uint8_t> &payload,
                  deepseek_layer_rpc_detail::Type expected,std::vector<uint8_t> &reply,
                  std::chrono::steady_clock::time_point deadline,std::string &error);
    int fd_=-1, timeout_ms_=0;
    mutable std::mutex socket_mutex_;
    bool interrupted_=false;
    uint64_t next_id_=1;
    bool begun_=false;
    DeepSeekLayerCapabilities capabilities_;
    DeepSeekBoundaryHeader session_;
};
} // namespace strata::core
