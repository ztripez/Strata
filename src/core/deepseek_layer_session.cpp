#include "strata/core/deepseek_layer_session.hpp"
#include <limits>

namespace strata::core {
namespace {
namespace D = deepseek_boundary_detail;
constexpr uint32_t result_magic = 0x52345344; // "DS4R", not STEX or DS4B
constexpr uint16_t result_version = 1;
constexpr size_t result_max_bytes = 16u << 20;
size_t result_values(const DeepSeekBoundaryLayout &layout, const DeepSeekBoundaryHeader &h, uint32_t rows) {
    D::layout(layout,h);
#ifdef STRATA_KIMI2_LIBRARY
    const uint64_t width=layout.result_width?layout.result_width:layout.vocab;
#else
    const uint64_t width=layout.vocab;
#endif
    const uint64_t values=static_cast<uint64_t>(rows)*width;
    D::require(rows && rows<=h.rows && values<=(result_max_bytes-160)/4,"invalid layer result row count or byte limit");
    return static_cast<size_t>(values);
}
void validate_result(const DeepSeekBoundaryLayout &layout, const DeepSeekLayerResult &result) {
    const size_t n=result_values(layout,result.request,result.logits_rows);
    D::require(result.committed_position==result.request.position+result.request.rows && result.logits.size()==n,
               "layer result did not commit the requested window or has wrong logits shape");
    for (const auto x : result.logits) D::require(std::isfinite(x),"nonfinite layer result");
}
}

std::vector<uint8_t> encode_deepseek_layer_result(const DeepSeekBoundaryLayout &layout,
                                                const DeepSeekLayerResult &result) {
    validate_result(layout,result);
    ExpertWireWriter body;
    D::fields(body,result.request);
    body(result.committed_position,result.logits_rows);
    for (const auto x : result.logits) body.put(x);
    ExpertWireWriter frame;
    frame(result_magic,result_version,uint16_t{0},static_cast<uint32_t>(body.bytes.size()));
    frame.bytes.insert(frame.bytes.end(),body.bytes.begin(),body.bytes.end());
    D::require(frame.bytes.size()<=result_max_bytes,"layer result exceeds byte limit");
    return std::move(frame.bytes);
}

DeepSeekLayerResult decode_deepseek_layer_result(const DeepSeekBoundaryLayout &layout,
    const DeepSeekBoundaryHeader &expected, uint32_t logits_rows, const std::vector<uint8_t> &bytes) {
    const size_t n=result_values(layout,expected,logits_rows);
    // Prefix 12 + request 136 + committed position 8 + row count 4.
    D::require(bytes.size()==160+n*4 && bytes.size()<=result_max_bytes,"invalid layer result frame length");
    ExpertWireReader r{bytes};
    uint32_t magic=0,length=0; uint16_t version=0,flags=0;
    r(magic,version,flags,length);
    D::require(r.ok && magic==result_magic && version==result_version && !flags && length==bytes.size()-12,
               "invalid layer result magic, version, flags or length");
    DeepSeekLayerResult result;
    D::fields(r,result.request); r(result.committed_position,result.logits_rows);
    D::require(r.ok && D::equal(expected,result.request) && result.logits_rows==logits_rows,"unexpected layer result identity or rows");
    result.logits.resize(n);
    for (auto &x : result.logits) r.get(x);
    D::require(r.done(),"truncated layer result or trailing bytes");
    validate_result(layout,result);
    return result;
}

DeepSeekLayerSession::DeepSeekLayerSession(DeepSeekBoundaryLayout layout, DeepSeekBoundaryHeader identity, uint32_t cut)
    : layout_(std::move(layout)),next_(identity) {
    next_.position=0; next_.sequence=1; next_.rows=1; next_.next_layer=cut;
    D::layout(layout_,next_);
}
bool DeepSeekLayerSession::fail(const std::string &reason, std::string &error) {
    aborted_=true; error=reason;
    retained_=false; retained_result_=DeepSeekLayerResult{};
    std::vector<uint8_t>{}.swap(retained_request_);
    return false;
}
bool DeepSeekLayerSession::execute(const DeepSeekLayerBoundary &capture, uint32_t logits_rows,
    const Executor &executor, DeepSeekLayerResult &result, std::string &error) {
    if (aborted_) { error="layer delivery session is aborted"; return false; }
    try {
        D::validate(layout_,capture);
        (void)result_values(layout_,capture.header,logits_rows);
        const auto request=encode_deepseek_boundary(layout_,capture);
        if (retained_) {
            if (!D::equal(capture.header,retained_result_.request) || logits_rows!=retained_logits_rows_ || request!=retained_request_)
                return fail("conflicting duplicate or next layer window before acknowledgement",error);
            auto delivered=retained_result_;
            result=std::move(delivered); return true;
        }
        auto expected=next_; expected.rows=capture.header.rows;
        if (!D::equal(expected,capture.header) || capture.header.sequence==std::numeric_limits<uint64_t>::max())
            return fail("stale, duplicate or out-of-order layer window",error);
        if (!executor) return fail("missing native layer executor",error);
        DeepSeekLayerResult completed;
        std::string execution_error;
        if (!executor(capture,logits_rows,completed,execution_error))
            return fail(execution_error.empty() ? "native layer executor failed without a reason" : execution_error,error);
        validate_result(layout_,completed);
        if (!D::equal(completed.request,capture.header) || completed.logits_rows!=logits_rows)
            return fail("native layer executor returned a different window",error);
        retained_request_=request; retained_result_=std::move(completed);
        retained_logits_rows_=logits_rows; retained_=true;
        next_.position=capture.header.position+capture.header.rows; ++next_.sequence;
        auto delivered=retained_result_;
        result=std::move(delivered); return true;
    } catch (const std::exception &e) { return fail(e.what(),error); }
}
bool DeepSeekLayerSession::acknowledge(const DeepSeekBoundaryHeader &request, std::string &error) {
    if (aborted_) { error="layer delivery session is aborted"; return false; }
    if (!retained_ || !D::equal(request,retained_result_.request)) return fail("stale or conflicting layer acknowledgement",error);
    retained_=false; retained_result_=DeepSeekLayerResult{};
    std::vector<uint8_t>{}.swap(retained_request_);
    return true;
}
} // namespace strata::core
