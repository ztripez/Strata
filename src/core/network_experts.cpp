#include "strata/core/network_experts.hpp"
#include "strata/core/expert_worker_wire.hpp"

#include <algorithm>
#include <cmath>
#include <cstring>
#include <set>
#include <stdexcept>

namespace strata::core {
namespace {
using Clock = std::chrono::steady_clock;
uint64_t us_since(Clock::time_point t0) {
    return static_cast<uint64_t>(std::chrono::duration_cast<std::chrono::microseconds>(Clock::now() - t0).count());
}
uint64_t percentile(std::vector<uint64_t> v, int pct) {
    if (v.empty())
        return 0;
    std::sort(v.begin(), v.end());
    return v[std::min(v.size() - 1, v.size() * static_cast<size_t>(pct) / 100)];
}
// Keep the most recent 4096 samples, bounding memory during long-running serving.
void sample(std::vector<uint64_t> &v, uint64_t n, uint64_t value) {
    if (v.size() < 4096)
        v.push_back(value);
    else
        v[(n - 1) % 4096] = value;
}
} // namespace

NetworkExpertExecutor::~NetworkExpertExecutor() { close(); }
void NetworkExpertExecutor::close() {
    {
        std::lock_guard<std::mutex> lk(mu_);
        quit_ = true;
        expert_worker_shutdown(fd_);
        cv_req_.notify_all();
    }
    if (io_thread_.joinable())
        io_thread_.join();
    expert_worker_close(fd_);
    active_.store(false);
    healthy_.store(false);
    installed_ = false;
    in_flight_ = false;
    keepalive_interval_ms_ = 0;
}

bool NetworkExpertExecutor::connect_and_handshake(const std::string &endpoint, uint32_t n_layers, uint32_t total_layers,
                                                  uint32_t n_experts, uint32_t n_embd, uint32_t n_ff, uint32_t top_k,
                                                  const std::array<uint8_t, 32> &model_sha256, int timeout_ms,
                                                  std::string &err, ExpertWorkerArithmetic arithmetic) {
    close();
    if (!n_layers || total_layers < n_layers || !n_experts || !n_embd || !n_ff || !top_k || top_k > 16 ||
        timeout_ms <= 0 || timeout_ms > 600000 || static_cast<uint64_t>(total_layers) * n_experts > 1000000 ||
        static_cast<uint32_t>(arithmetic) > static_cast<uint32_t>(ExpertWorkerArithmetic::DEEPSEEK_IQ_Q8K)) {
        err = "invalid expert worker geometry or timeout (1..600000 ms)";
        return false;
    }
    std::string host;
    uint16_t port = 0;
    if (!expert_worker_parse_host_port(endpoint, host, port)) {
        err = "invalid --expert-worker endpoint; expected numeric IPv4:port";
        return false;
    }
    endpoint_ = endpoint;
    timeout_ms_ = timeout_ms;
    protocol_version_ = arithmetic == ExpertWorkerArithmetic::NATIVE_V1 ? kExpertWorkerVersion : kExpertWorkerArithmeticVersion;
    n_layers_ = n_layers;
    total_layers_ = total_layers;
    n_experts_ = n_experts;
    n_embd_ = n_embd;
    n_ff_ = n_ff;
    top_k_ = top_k;
    owned_.assign(static_cast<size_t>(total_layers) * n_experts, 0);
    layer_owned_.assign(static_cast<size_t>(total_layers), 0);
    ready_ = {};
    stats_ = {};
    fd_ = expert_worker_connect(host, port, timeout_ms, &err);
    if (fd_ < 0)
        return false;
    ExpertWorkerHelloPayload hp{};
    hp.model_fingerprint = compute_model_fingerprint(n_layers, n_experts, n_embd, n_ff);
    hp.model_sha256 = model_sha256;
    hp.n_embd = n_embd;
    hp.n_ff = n_ff;
    hp.n_layers = n_layers;
    hp.n_experts = n_experts;
    hp.top_k = top_k;
    hp.endianness = 0x01020304u;
    const uint64_t rid = next_req_id_++;
    ExpertWorkerFrameHeader hdr{};
    std::vector<uint8_t> payload;
    const bool sent = protocol_version_ == kExpertWorkerVersion
        ? expert_worker_send_message(fd_, ExpertWorkerMsgType::HELLO, rid, hp, &err, timeout_ms)
        : expert_worker_send_message(fd_, ExpertWorkerMsgType::HELLO, rid,
              ExpertWorkerHelloV2{hp,static_cast<uint32_t>(arithmetic)}, &err, timeout_ms, protocol_version_);
    if (!sent ||
        !expert_worker_recv_frame(fd_, hdr, payload, &err, timeout_ms)) {
        close();
        return false;
    }
    bool decoded = false;
    uint32_t peer_arithmetic = 0;
    if (protocol_version_ == kExpertWorkerVersion) decoded = expert_wire_decode(payload, caps_);
    else {
        ExpertWorkerCapsV2 v2{};
        decoded = expert_wire_decode(payload,v2); caps_ = v2.model; peer_arithmetic = v2.arithmetic;
    }
    if (hdr.version != protocol_version_ || hdr.request_id != rid || hdr.msg_type != static_cast<uint16_t>(ExpertWorkerMsgType::CAPS) ||
        !decoded || peer_arithmetic != static_cast<uint32_t>(arithmetic) || caps_.status != 0 || caps_.model_sha256 != model_sha256 ||
        caps_.model_fingerprint != hp.model_fingerprint || caps_.n_embd != n_embd || caps_.n_ff != n_ff ||
        caps_.n_layers != n_layers || caps_.n_experts != n_experts || caps_.top_k != top_k || caps_.dtype != 0 ||
        caps_.endianness != hp.endianness || caps_.reserved != 0 || !caps_.max_tokens || caps_.backend_id > 2 ||
        !std::memchr(caps_.device_name, 0, sizeof(caps_.device_name))) {
        auto hex = [](const std::array<uint8_t, 32> &digest) {
            std::string text;
            constexpr char digits[] = "0123456789abcdef";
            for (auto b : digest) {
                text += digits[b >> 4];
                text += digits[b & 15];
            }
            return text;
        };
        err = "worker CAPS rejected: local SHA-256=" + hex(model_sha256) +
              ", worker SHA-256=" + hex(caps_.model_sha256) +
              "; check model bytes, geometry, arithmetic policy, protocol version, request ID and capabilities";
        close();
        return false;
    }
    quit_ = false;
    req_pending_ = false;
    req_done_ = false;
    active_.store(true);
    healthy_.store(true);
    io_thread_ = std::thread(&NetworkExpertExecutor::io_loop, this);
    return true;
}

bool NetworkExpertExecutor::install_plan(const std::vector<ExpertWorkerPair> &pairs, std::string &err) {
    std::unique_lock<std::mutex> lk(mu_);
    if (!healthy() || installed_ || in_flight_) {
        err = "worker plan must be installed once before execution";
        return false;
    }
    if (pairs.size() > caps_.max_slots) {
        err = "worker plan exceeds advertised slot limit";
        return false;
    }
    std::vector<uint8_t> proposed(owned_.size(), 0);
    std::vector<uint8_t> proposed_layers(layer_owned_.size(), 0);
    for (const auto &p : pairs) {
        if (p.layer < 0 || static_cast<uint32_t>(p.layer) >= total_layers_ || p.expert < 0 ||
            static_cast<uint32_t>(p.expert) >= n_experts_) {
            err = "worker plan pair out of bounds";
            return false;
        }
        auto &bit = proposed[static_cast<size_t>(p.layer) * n_experts_ + p.expert];
        if (bit) {
            err = "duplicate worker plan pair";
            return false;
        }
        bit = 1;
        proposed_layers[static_cast<size_t>(p.layer)] = 1;
    }
    ExpertWireWriter w;
    w.put(static_cast<uint32_t>(pairs.size()));
    for (auto p : pairs)
        w.put(p);
    const uint64_t rid = next_req_id_++;
    ExpertWorkerFrameHeader hdr{};
    std::vector<uint8_t> payload;
    if (!expert_worker_send_frame(fd_, ExpertWorkerMsgType::INSTALL_PLAN, rid, w.bytes.data(),
                                  static_cast<uint32_t>(w.bytes.size()), &err, timeout_ms_, protocol_version_) ||
        !expert_worker_recv_frame(fd_, hdr, payload, &err, timeout_ms_)) {
        healthy_.store(false);
        return false;
    }
    if (hdr.version != protocol_version_ || hdr.request_id != rid || hdr.msg_type != static_cast<uint16_t>(ExpertWorkerMsgType::READY) ||
        !expert_wire_decode(payload, ready_) || ready_.status != 0 || ready_.loaded_count != pairs.size()) {
        err = "worker did not acknowledge the complete requested placement";
        healthy_.store(false);
        return false;
    }
    owned_ = std::move(proposed); // Publish ownership only after the exact READY acknowledgement.
    layer_owned_ = std::move(proposed_layers);
    installed_ = true;
    stats_.resident_experts = ready_.loaded_count;
    stats_.load_time_us = ready_.load_time_us;
    stats_.worker_vram_used_bytes = ready_.vram_used_bytes;
    return true;
}

bool NetworkExpertExecutor::begin_async(int32_t layer, uint32_t n_tokens, const float *acts,
                                        const std::vector<ExpertWorkerAssignment> &assigns) {
    std::lock_guard<std::mutex> lk(mu_);
    if (!healthy() || !installed_ || in_flight_ || !acts || assigns.empty() || !n_tokens ||
        n_tokens > caps_.max_tokens || n_tokens > 65535 || assigns.size() > static_cast<size_t>(n_tokens) * top_k_)
        return false;
    expected_mask_.assign(n_tokens, 0);
    std::set<std::pair<uint16_t, uint16_t>> slots;
    for (const auto &a : assigns) {
        if (a.token_idx >= n_tokens || a.slot_idx >= top_k_ || !holds(layer, a.expert_id) || !std::isfinite(a.weight) ||
            !slots.emplace(a.token_idx, a.slot_idx).second)
            return false;
        expected_mask_[a.token_idx] = 1;
    }
    const size_t count = static_cast<size_t>(n_tokens) * n_embd_;
    if (24 + assigns.size() * 12 + count * 4 > kExpertWorkerMaxPayloadBytes)
        return false;
    for (size_t i = 0; i < count; ++i)
        if (!std::isfinite(acts[i]))
            return false;
    cur_layer_ = layer;
    cur_n_tokens_ = n_tokens;
    cur_n_assigns_ = static_cast<uint32_t>(assigns.size());
    ExpertWireWriter w;
    w.put(ExpertWorkerExecuteFixed{layer, n_tokens, n_embd_, n_tokens, cur_n_assigns_, 0});
    for (const auto &a : assigns)
        w.put(a);
    for (size_t i = 0; i < count; ++i)
        w.put(acts[i]);
    tx_buf_ = std::move(w.bytes);
    dispatch_t0_ = Clock::now();
    req_done_ = false;
    req_pending_ = true;
    in_flight_ = true;
    cv_req_.notify_one();
    return true;
}

void NetworkExpertExecutor::set_keepalive_interval(int interval_ms) {
    if (interval_ms < 0)
        throw std::invalid_argument("worker keepalive interval must be nonnegative");
    std::lock_guard<std::mutex> lk(mu_);
    keepalive_interval_ms_ = interval_ms;
    cv_req_.notify_one();
}

void NetworkExpertExecutor::io_loop() {
    std::unique_lock<std::mutex> lk(mu_);
    while (true) {
        if (keepalive_interval_ms_ == 0) {
            cv_req_.wait(lk, [&] { return quit_ || req_pending_ || keepalive_interval_ms_ > 0; });
        } else if (!cv_req_.wait_for(lk, std::chrono::milliseconds(keepalive_interval_ms_),
                                    [&] { return quit_ || req_pending_ || keepalive_interval_ms_ == 0; }) &&
                   installed_ && !in_flight_ && healthy()) {
            const uint64_t rid = next_req_id_++;
            lk.unlock();
            ExpertWorkerFrameHeader pong{};
            std::vector<uint8_t> payload;
            std::string err;
            const bool ok = expert_worker_send_frame(fd_, ExpertWorkerMsgType::PING, rid, nullptr, 0,
                                                     &err, timeout_ms_) &&
                            expert_worker_recv_frame(fd_, pong, payload, &err, timeout_ms_) &&
                            pong.request_id == rid &&
                            pong.msg_type == static_cast<uint16_t>(ExpertWorkerMsgType::PONG) && payload.empty();
            lk.lock();
            if (quit_)
                break;
            if (!ok) {
                healthy_.store(false);
                expert_worker_shutdown(fd_);
                std::fprintf(stderr, "strata-network: worker keepalive failed: %s\n",
                             err.empty() ? "unexpected worker PONG" : err.c_str());
            } else {
                std::fprintf(stderr, "strata-network: worker PONG request=%llu\n", (unsigned long long)rid);
            }
            continue;
        }
        if (quit_)
            break;
        if (!req_pending_)
            continue;
        req_pending_ = false;
        if (!healthy()) {
            req_ok_ = false;
            req_err_ = "expert worker connection is unhealthy";
            req_done_ = true;
            cv_done_.notify_all();
            continue;
        }
        const uint64_t rid = next_req_id_++;
        const auto deadline = dispatch_t0_ + std::chrono::milliseconds(timeout_ms_);
        lk.unlock();
        const auto t0 = Clock::now();
        std::string err;
        auto remaining = [&] {
            return static_cast<int>(
                std::chrono::duration_cast<std::chrono::milliseconds>(deadline - Clock::now()).count());
        };
        bool ok = expert_worker_send_frame(fd_, ExpertWorkerMsgType::EXECUTE, rid, tx_buf_.data(),
                                           static_cast<uint32_t>(tx_buf_.size()), &err, remaining(), protocol_version_) &&
                  expert_worker_recv_frame(fd_, rx_hdr_, rx_buf_, &err, remaining());
        if (ok &&
            (rx_hdr_.version != protocol_version_ || rx_hdr_.request_id != rid || rx_hdr_.msg_type != static_cast<uint16_t>(ExpertWorkerMsgType::RESULT))) {
            ok = false;
            err = "unexpected worker response or stale request ID";
        }
        lk.lock();
        last_rtt_us_ = us_since(t0);
        req_ok_ = ok;
        req_err_ = err;
        if (!ok)
            healthy_.store(false);
        req_done_ = true;
        cv_done_.notify_all();
    }
}

bool NetworkExpertExecutor::wait_result(float *out_sums, uint8_t *out_mask, std::string *err) {
    const auto t0 = Clock::now();
    std::unique_lock<std::mutex> lk(mu_);
    if (!in_flight_) {
        if (err)
            *err = "no worker request in flight";
        return false;
    }
    if (!cv_done_.wait_until(lk, dispatch_t0_ + std::chrono::milliseconds(timeout_ms_), [&] { return req_done_; })) {
        expert_worker_shutdown(fd_); // Wakes the I/O thread; do not reuse buffers until it finishes.
        cv_done_.wait(lk, [&] { return req_done_; });
        req_ok_ = false;
        req_err_ = "expert worker request deadline exceeded";
    }
    in_flight_ = false;
    auto fail = [&](const std::string &why) {
        if (err)
            *err = why;
        healthy_.store(false);
        expert_worker_shutdown(fd_);
        ++stats_.fallbacks;
        return false;
    };
    if (!req_ok_)
        return fail(req_err_);
    ExpertWireReader r{rx_buf_};
    ExpertWorkerResultFixed rf{};
    r.get(rf);
    if (!r.ok || rf.status != 0 || rf.reserved || rf.layer != cur_layer_ || rf.n_tokens != cur_n_tokens_ ||
        rf.n_embd != n_embd_ || rf.n_assignments_executed != cur_n_assigns_)
        return fail("worker RESULT geometry or assignment coverage mismatch");
    std::vector<uint8_t> mask(cur_n_tokens_);
    for (auto &v : mask)
        r.get(v);
    if (mask != expected_mask_)
        return fail("worker RESULT token coverage mismatch");
    const size_t count = static_cast<size_t>(cur_n_tokens_) * n_embd_;
    std::vector<float> sums(count);
    for (auto &v : sums)
        r.get(v);
    if (!r.done())
        return fail("worker RESULT payload length mismatch");
    for (size_t i = 0; i < count; ++i)
        if (!std::isfinite(sums[i]) || (!mask[i / n_embd_] && sums[i] != 0))
            return fail("invalid worker partial sum");
    if (out_sums)
        std::memcpy(out_sums, sums.data(), count * sizeof(float));
    if (out_mask)
        std::memcpy(out_mask, mask.data(), mask.size());
    ++stats_.layers_dispatched;
    stats_.assignments_executed += rf.n_assignments_executed;
    for (auto m : mask)
        stats_.tokens_with_remote += m;
    stats_.tx_bytes += 24 + tx_buf_.size();
    stats_.rx_bytes += 24 + rx_buf_.size();
    const uint64_t n = stats_.layers_dispatched;
    sample(stats_.layer_ids, n, static_cast<uint64_t>(cur_layer_));
    sample(stats_.rtt_us, n, last_rtt_us_);
    sample(stats_.wait_us, n, us_since(t0));
    sample(stats_.worker_stage_us, n, rf.stage_us);
    sample(stats_.worker_kernel_us, n, rf.kernel_us);
    sample(stats_.worker_copy_us, n, rf.copy_out_us);
    sample(stats_.worker_total_us, n, rf.total_us);
    return true;
}

bool NetworkExpertExecutor::execute_sync(int32_t layer, uint32_t n_tokens, const float *acts,
                                         const std::vector<ExpertWorkerAssignment> &assigns, float *sums, uint8_t *mask,
                                         std::string *err) {
    if (!begin_async(layer, n_tokens, acts, assigns)) {
        if (err)
            *err = "invalid or overlapping worker request";
        return false;
    }
    return wait_result(sums, mask, err);
}
void NetworkExpertExecutor::reset_stats() {
    std::lock_guard<std::mutex> lk(mu_);
    stats_ = {};
    stats_.resident_experts = ready_.loaded_count;
    stats_.load_time_us = ready_.load_time_us;
    stats_.worker_vram_used_bytes = ready_.vram_used_bytes;
}
void NetworkExpertExecutor::print_summary(FILE *out) const {
    if (!active())
        return;
    std::fprintf(out,
                 "strata-network-worker: endpoint=%s device='%s' backend=%u resident_experts=%llu "
                 "layers_dispatched=%llu remote_entries=%llu tx_bytes=%llu rx_bytes=%llu fallbacks=%llu",
                 endpoint_.c_str(), caps_.device_name, caps_.backend_id, (unsigned long long)stats_.resident_experts,
                 (unsigned long long)stats_.layers_dispatched, (unsigned long long)stats_.assignments_executed,
                 (unsigned long long)stats_.tx_bytes, (unsigned long long)stats_.rx_bytes,
                 (unsigned long long)stats_.fallbacks);
    auto report = [&](const char *name, const std::vector<uint64_t> &v) {
        std::fprintf(out, " %s_us(p50=%llu,p95=%llu,p99=%llu)", name, (unsigned long long)percentile(v, 50),
                     (unsigned long long)percentile(v, 95), (unsigned long long)percentile(v, 99));
    };
    report("rtt", stats_.rtt_us);
    report("wait", stats_.wait_us);
    report("worker_stage", stats_.worker_stage_us);
    report("worker_kernel", stats_.worker_kernel_us);
    report("worker_copy", stats_.worker_copy_us);
    report("worker_total", stats_.worker_total_us);
    std::fprintf(out, "\n");
    const std::set<uint64_t> layers(stats_.layer_ids.begin(), stats_.layer_ids.end());
    for (const auto layer : layers) {
        std::vector<uint64_t> wait, rtt, stage, kernel, copy, total;
        for (size_t i = 0; i < stats_.layer_ids.size(); ++i) {
            if (stats_.layer_ids[i] != layer) continue;
            wait.push_back(stats_.wait_us[i]);
            rtt.push_back(stats_.rtt_us[i]);
            stage.push_back(stats_.worker_stage_us[i]);
            kernel.push_back(stats_.worker_kernel_us[i]);
            copy.push_back(stats_.worker_copy_us[i]);
            total.push_back(stats_.worker_total_us[i]);
        }
        std::fprintf(out, "strata-network-layer: layer=%llu samples=%zu",
                     (unsigned long long)layer, wait.size());
        report("wait", wait);
        report("rtt", rtt);
        report("worker_stage", stage);
        report("worker_kernel", kernel);
        report("worker_copy", copy);
        report("worker_total", total);
        std::fprintf(out, "\n");
    }
}
} // namespace strata::core
