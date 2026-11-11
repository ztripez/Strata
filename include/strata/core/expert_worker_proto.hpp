#pragma once

#include <array>
#include <cstddef>
#include <cstdint>
#include <string>
#include <vector>

namespace strata::core {

// Magic: "STEX" in little-endian (0x58455453u)
inline constexpr uint32_t kExpertWorkerMagic = 0x58455453u;
inline constexpr uint16_t kExpertWorkerVersion = 1;
/// Expert v2 adds an explicit arithmetic policy to HELLO/CAPS. Qwen keeps v1.
inline constexpr uint16_t kExpertWorkerArithmeticVersion = 2;

/// Resolved execution arithmetic, independent of immutable model-weight bytes.
enum class ExpertWorkerArithmetic : uint32_t {
    NATIVE_V1 = 0,
    DEEPSEEK_IQ_PER32 = 1,
    DEEPSEEK_IQ_Q8K = 2,
};

enum class ExpertWorkerMsgType : uint16_t {
    HELLO = 1,
    CAPS = 2,
    INSTALL_PLAN = 3,
    READY = 4,
    EXECUTE = 5,
    RESULT = 6,
    PING = 7,
    PONG = 8,
    ERROR_MSG = 9,
};

enum class ExpertWorkerErrorCode : uint32_t {
    OK = 0,
    BAD_MAGIC_OR_VERSION = 1,
    MODEL_MISMATCH = 2,
    UNSUPPORTED_CONFIG = 3,
    PLAN_LOAD_FAILED = 4,
    NONRESIDENT_EXPERT = 5,
    KERNEL_FAILURE = 6,
    MALFORMED_PACKET = 7,
    INTERNAL_ERROR = 8,
};

struct ExpertWorkerFrameHeader {
    uint32_t magic;       // kExpertWorkerMagic
    uint16_t version;     // v1 or explicit-arithmetic v2; fixed for a session
    uint16_t msg_type;    // ExpertWorkerMsgType
    uint64_t request_id;  // monotonically increasing request id
    uint32_t payload_len; // byte length following header
    uint32_t flags;       // reserved (0)
};

struct ExpertWorkerHelloPayload {
    uint64_t model_fingerprint; // geometry discriminator, not model identity
    std::array<uint8_t, 32> model_sha256{};
    uint32_t n_embd;
    uint32_t n_ff;
    uint32_t n_layers;
    uint32_t n_experts;
    uint32_t top_k;
    uint32_t dtype;      // 0 = FP32 activation/output rows
    uint32_t endianness; // 0x01020304u
    uint32_t reserved;
};

struct ExpertWorkerCapsPayload {
    uint64_t model_fingerprint; // geometry discriminator, not model identity
    std::array<uint8_t, 32> model_sha256{};
    uint32_t n_embd;
    uint32_t n_ff;
    uint32_t n_layers;
    uint32_t n_experts;
    uint32_t top_k;
    uint32_t dtype;
    uint32_t endianness;
    uint32_t backend_id; // 0 = CPU_MOCK, 1 = CUDA, 2 = HIP
    uint32_t max_tokens;
    uint32_t max_slots;
    uint32_t status; // ExpertWorkerErrorCode
    uint32_t reserved;
    uint64_t vram_free_bytes;
    uint64_t vram_total_bytes;
    char device_name[64];
};

struct ExpertWorkerPair {
    int32_t layer;
    int32_t expert;
};

/// V2 envelope; the v1 schema and all its reserved fields remain unchanged.
struct ExpertWorkerHelloV2 {
    ExpertWorkerHelloPayload model{};
    uint32_t arithmetic = 0; // ExpertWorkerArithmetic, mandatory in v2
};
/// V2 acknowledgement reports the worker's effective arithmetic, not a request
/// echo. A mismatch must fail before placement is installed.
struct ExpertWorkerCapsV2 {
    ExpertWorkerCapsPayload model{};
    uint32_t arithmetic = 0; // ExpertWorkerArithmetic
};

struct ExpertWorkerReadyPayload {
    uint32_t status; // ExpertWorkerErrorCode
    uint32_t loaded_count;
    uint64_t vram_used_bytes;
    uint64_t vram_free_bytes;
    uint64_t load_time_us;
};

struct ExpertWorkerAssignment {
    uint16_t token_idx;
    uint16_t slot_idx;
    int32_t expert_id;
    float weight;
};

struct ExpertWorkerExecuteFixed {
    int32_t layer;
    uint32_t n_tokens;
    uint32_t n_embd;
    uint32_t n_UniqueTokens_or_FullRows; // number of float32 rows of length n_embd following assignments (always
                                         // n_tokens in v1)
    uint32_t n_assignments;
    uint32_t flags;
};

struct ExpertWorkerResultFixed {
    int32_t layer;
    uint32_t n_tokens;
    uint32_t n_embd;
    uint32_t n_assignments_executed;
    uint32_t status; // ExpertWorkerErrorCode
    uint32_t reserved;
    uint64_t stage_us;    // host->device upload + group staging time on worker
    uint64_t kernel_us;   // GPU kernel + reduction time on worker
    uint64_t copy_out_us; // device->host copy time on worker
    uint64_t total_us;    // total worker request handling time
};

struct ExpertWorkerErrorPayload {
    uint32_t code; // ExpertWorkerErrorCode
    uint32_t detail_len;
    char message[120];
};

// Maximum frame payload size (64 MiB safety cap to prevent OOM on malformed frames)
inline constexpr uint32_t kExpertWorkerMaxPayloadBytes = 64u * 1024u * 1024u;

// Compute a deterministic 64-bit FNV-1a fingerprint from model geometry.
uint64_t compute_model_fingerprint(uint32_t n_layers, uint32_t n_experts, uint32_t n_embd, uint32_t n_ff);

// Socket helpers: numeric IPv4 endpoints, POSIX TCP, finite frame deadlines.
bool expert_worker_set_socket_opts(int fd, int timeout_ms, std::string *err = nullptr);
int expert_worker_connect(const std::string &host, uint16_t port, int timeout_ms, std::string *err = nullptr);
int expert_worker_listen(const std::string &bind_ip, uint16_t port, int backlog, uint16_t *out_port = nullptr,
                         std::string *err = nullptr);
bool expert_worker_parse_host_port(const std::string &endpoint, std::string &host, uint16_t &port);
void expert_worker_close(int &fd);
// Interrupt an in-flight read/write without closing or reusing its descriptor.
void expert_worker_shutdown(int fd);

// Framing I/O helpers.
bool expert_worker_send_frame(int fd, ExpertWorkerMsgType type, uint64_t req_id, const void *payload,
                              uint32_t payload_len, std::string *err = nullptr, int timeout_ms = 5000,
                              uint16_t version = kExpertWorkerVersion);
/// Receive one frame. An opt-in positive idle timeout bounds the first byte;
/// after that byte arrives, the entire remaining frame shares timeout_ms.
/// Zero retains the original single deadline, including idle time.
bool expert_worker_recv_frame(int fd, ExpertWorkerFrameHeader &hdr, std::vector<uint8_t> &payload,
                              std::string *err = nullptr, int timeout_ms = 5000, int idle_timeout_ms = 0);
bool expert_worker_send_error(int fd, uint64_t req_id, ExpertWorkerErrorCode code, const std::string &msg,
                              uint16_t version = kExpertWorkerVersion);

// Placement plan file helpers.
// Format: text file with one "layer expert" or "layer:expert" or "layer e0,e1,e2..eN" or "layer start-end" per line,
// '#' comments allowed.
bool load_expert_placement_plan(const std::string &path, int n_layers, int n_experts,
                                std::vector<ExpertWorkerPair> &out_pairs, std::string *err = nullptr);
bool save_expert_placement_plan(const std::string &path, const std::vector<ExpertWorkerPair> &pairs,
                                std::string *err = nullptr);

} // namespace strata::core
