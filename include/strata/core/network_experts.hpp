// include/strata/core/network_experts.hpp - Opt-in network-distributed MoE expert executor.
//
// Connects to a remote `strata-expert-worker` over persistent TCP, negotiates HELLO/CAPS and INSTALL_PLAN/READY,
// and executes remote-owned MoE expert assignments asynchronously on a dedicated background I/O thread so the
// primary GPU and local CPU expert pool overlap with network round-trips.

#pragma once

#include "strata/core/expert_worker_proto.hpp"

#include <atomic>
#include <chrono>
#include <condition_variable>
#include <cstddef>
#include <cstdint>
#include <cstdio>
#include <mutex>
#include <string>
#include <thread>
#include <vector>

namespace strata::core {

struct NetworkExpertStats {
    uint64_t layers_dispatched = 0;
    uint64_t assignments_executed = 0;
    uint64_t tokens_with_remote = 0;
    uint64_t tx_bytes = 0;
    uint64_t rx_bytes = 0;
    uint64_t fallbacks = 0;
    uint64_t resident_experts = 0;
    uint64_t load_time_us = 0;
    uint64_t worker_vram_used_bytes = 0;
    // Aligned with the timing rings below; summaries cover the last 4096 successful requests.
    std::vector<uint64_t> layer_ids;
    std::vector<uint64_t> rtt_us;
    std::vector<uint64_t> wait_us;
    std::vector<uint64_t> worker_stage_us;
    std::vector<uint64_t> worker_kernel_us;
    std::vector<uint64_t> worker_copy_us;
    std::vector<uint64_t> worker_total_us;
};

class NetworkExpertExecutor {
  public:
    NetworkExpertExecutor() = default;
    ~NetworkExpertExecutor();

    NetworkExpertExecutor(const NetworkExpertExecutor &) = delete;
    NetworkExpertExecutor &operator=(const NetworkExpertExecutor &) = delete;

    // Connect to worker and verify model geometry (HELLO / CAPS).
    bool connect_and_handshake(const std::string &endpoint, uint32_t n_layers, uint32_t total_layers,
                               uint32_t n_experts, uint32_t n_embd, uint32_t n_ff, uint32_t top_k,
                               const std::array<uint8_t, 32> &model_sha256, int timeout_ms, std::string &err);

    // Send INSTALL_PLAN and wait for READY. Marks acknowledged (layer, expert) pairs as remote-owned.
    bool install_plan(const std::vector<ExpertWorkerPair> &pairs, std::string &err);

    // Close connection and stop background worker thread.
    void close();

    bool active() const { return active_.load(std::memory_order_relaxed); }
    bool healthy() const { return healthy_.load(std::memory_order_relaxed); }
    const ExpertWorkerCapsPayload &caps() const { return caps_; }
    const ExpertWorkerReadyPayload &ready_info() const { return ready_; }

    // Query whether (layer, expert) is owned by the remote worker and the worker connection is healthy.
    bool holds(int32_t layer, int32_t expert) const {
        if (!healthy_.load(std::memory_order_relaxed))
            return false;
        if (layer < 0 || layer >= static_cast<int32_t>(total_layers_) || expert < 0 ||
            expert >= static_cast<int32_t>(n_experts_)) {
            return false;
        }
        return owned_[static_cast<size_t>(layer) * n_experts_ + static_cast<size_t>(expert)] != 0;
    }

    // Begin an asynchronous EXECUTE request for `layer` with `n_tokens` activation rows (`acts`, n_tokens * n_embd
    // floats) and `assigns`. Returns immediately after queueing the request to the background I/O thread.
    bool begin_async(int32_t layer, uint32_t n_tokens, const float *acts,
                     const std::vector<ExpertWorkerAssignment> &assigns);

    // Wait for the in-flight request started by begin_async() to complete.
    // On success: copies `n_tokens * n_embd` weighted partial sums into `out_sums` and `n_tokens` coverage flags into
    // `out_mask`. On failure/timeout/disconnect: marks executor unhealthy, increments `stats_.fallbacks`, and returns
    // false so the caller can recompute affected assignments locally on the CPU fallback pool before releasing the
    // layer!
    bool wait_result(float *out_sums, uint8_t *out_mask, std::string *err = nullptr);

    // Synchronous convenience wrapper (begin_async + wait_result).
    bool execute_sync(int32_t layer, uint32_t n_tokens, const float *acts,
                      const std::vector<ExpertWorkerAssignment> &assigns, float *out_sums, uint8_t *out_mask,
                      std::string *err = nullptr);

    const NetworkExpertStats &stats() const { return stats_; }
    void reset_stats();
    void print_summary(FILE *out = stderr) const;

  private:
    void io_loop();

    int fd_ = -1;
    std::string endpoint_;
    int timeout_ms_ = 5000;
    uint32_t n_layers_ = 0;
    uint32_t total_layers_ = 0;
    uint32_t n_experts_ = 0;
    uint32_t n_embd_ = 0;
    uint32_t n_ff_ = 0;
    uint32_t top_k_ = 0;
    uint64_t next_req_id_ = 1;

    ExpertWorkerCapsPayload caps_{};
    ExpertWorkerReadyPayload ready_{};
    std::vector<uint8_t> owned_; // [total_layers * n_experts]

    std::atomic<bool> active_{false};
    std::atomic<bool> healthy_{false};

    // Background I/O thread state
    std::thread io_thread_;
    std::mutex mu_;
    std::condition_variable cv_req_;
    std::condition_variable cv_done_;
    bool quit_ = false;
    bool installed_ = false;
    bool in_flight_ = false;
    std::vector<uint8_t> expected_mask_;
    bool req_pending_ = false;
    bool req_done_ = false;
    bool req_ok_ = false;
    std::string req_err_;

    int32_t cur_layer_ = 0;
    uint32_t cur_n_tokens_ = 0;
    uint32_t cur_n_assigns_ = 0;
    std::vector<uint8_t> tx_buf_;
    std::vector<uint8_t> rx_buf_;
    ExpertWorkerFrameHeader rx_hdr_{};
    uint64_t last_rtt_us_ = 0;
    std::chrono::steady_clock::time_point dispatch_t0_{};

    NetworkExpertStats stats_;
};

} // namespace strata::core
