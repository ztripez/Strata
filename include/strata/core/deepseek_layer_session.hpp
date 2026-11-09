#pragma once

#include "strata/core/deepseek_layer_boundary.hpp"
#include <functional>

namespace strata::core {

/// A suffix executor's committed output. `request` identifies the input window;
/// committed_position must attest its actual native position after execution.
struct DeepSeekLayerResult {
    DeepSeekBoundaryHeader request;
    uint64_t committed_position = 0;
    uint32_t logits_rows = 0;
    std::vector<float> logits;
};

/// Encode a finite result with a separate DS4R magic/version and 16-MiB limit.
/// Geometry comes from the trusted local layout, never from received dimensions.
std::vector<uint8_t> encode_deepseek_layer_result(const DeepSeekBoundaryLayout &layout,
                                                const DeepSeekLayerResult &result);
/// Decode against the exact outstanding request and requested logit-row count.
/// This creates a host value only; it does not publish coordinator/session state.
DeepSeekLayerResult decode_deepseek_layer_result(const DeepSeekBoundaryLayout &layout,
    const DeepSeekBoundaryHeader &expected, uint32_t logits_rows, const std::vector<uint8_t> &bytes);

/// One in-flight layer window and one retained result. Native Model owns all
/// attention/KV state; this object owns delivery/acknowledgement state only.
/// An identical duplicate before ACK returns the retained result without calling
/// the executor. Conflicts, late duplicates and any execution failure abort this
/// session. A new session also requires native reset and a process-level fence
/// against reused session IDs; constructing this object alone does not provide it.
class DeepSeekLayerSession {
public:
    /// The callback invokes the authoritative native suffix executor. It must
    /// report actual committed position and logits, or a visible failure. A false
    /// return or exception ends this session even if the engine rejected before
    /// mutation; no implicit layer retry or local fallback is performed.
    using Executor = std::function<bool(const DeepSeekLayerBoundary&, uint32_t,
                                       DeepSeekLayerResult&, std::string&)>;
    /// Start at position zero/window one with immutable model/pack/plan identity.
    /// `cut` and layout are locally trusted; other window fields in identity are
    /// replaced with the initial position/sequence during configuration.
    DeepSeekLayerSession(DeepSeekBoundaryLayout layout, DeepSeekBoundaryHeader identity, uint32_t cut);
    /// Validate a complete capture, execute once and retain the checked result.
    /// On failure, `result` is unchanged and the session is aborted.
    bool execute(const DeepSeekLayerBoundary &capture, uint32_t logits_rows,
                 const Executor &executor, DeepSeekLayerResult &result, std::string &error);
    /// Release the retained result only for its exact request identity. Late or
    /// conflicting acknowledgements abort; the next window cannot precede ACK.
    bool acknowledge(const DeepSeekBoundaryHeader &request, std::string &error);
    /// True after a protocol or execution failure. Only a new native session can
    /// recover; this object deliberately offers no reset of delivery state alone.
    bool aborted() const { return aborted_; }
    /// Whether a completed response is still retained pending acknowledgement.
    bool awaiting_acknowledgement() const { return retained_; }
private:
    bool fail(const std::string &reason, std::string &error);
    DeepSeekBoundaryLayout layout_;
    DeepSeekBoundaryHeader next_;
    bool aborted_ = false, retained_ = false;
    uint32_t retained_logits_rows_ = 0;
    std::vector<uint8_t> retained_request_;
    DeepSeekLayerResult retained_result_;
};

} // namespace strata::core
