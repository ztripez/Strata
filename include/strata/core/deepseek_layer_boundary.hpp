#pragma once

#include "strata/core/expert_worker_wire.hpp"
#include <algorithm>
#include <cmath>
#include <stdexcept>

namespace strata::core {

/// Portable capture record for E2 local replay. This is not an expert-v1 message
/// or a complete layer RPC/session protocol. Every size comes from a trusted
/// local model layout; received lengths cannot override that layout.
struct DeepSeekBoundaryHeader {
    std::array<uint8_t, 32> model{}, pack{}, plan{};
    uint64_t epoch = 0, session = 0, sequence = 0, position = 0;
    uint32_t next_layer = 0, rows = 0;
};

/// Expected cross-cut cache update, derived from the source-layer dependency
/// audit. Both interval endpoints are completed compression groups, not tokens.
struct DeepSeekBoundarySource {
    uint32_t layer = 0, ratio = 0, kv_row_bytes = 0, key_row_bytes = 0;
};

/// Trusted layout for a single sequence's at-most-16-row capture. `pre` uses the
/// authoritative engine's padded eight-float row, irrespective of stream count.
struct DeepSeekBoundaryLayout {
    uint32_t hidden = 0, streams = 0, topk_stride = 0, cand_stride = 0, vocab = 0;
    uint64_t max_context = 0;
    std::vector<DeepSeekBoundarySource> sources;
};

/// Append-only rows produced at an audited source layer for this window. Empty
/// intervals still carry their descriptor so a missing update cannot be hidden.
struct DeepSeekBoundaryCache {
    uint32_t source_layer = 0, rows = 0;
    uint64_t first_row = 0;
    std::vector<uint8_t> kv, keys;
};

/// Host-owned boundary fields. Device pointers, local events and queue flags
/// never enter this representation. Capture/restore must synchronize the local
/// engine before reading or applying it.
struct DeepSeekLayerBoundary {
    DeepSeekBoundaryHeader header;
    std::vector<int32_t> tokens, topk, cand;
    std::vector<uint32_t> topk_valid, cand_valid;
    std::vector<float> h, pre;
    std::vector<DeepSeekBoundaryCache> caches;
};

namespace deepseek_boundary_detail {
constexpr uint32_t magic = 0x42345344; // bytes "DS4B"; distinct from expert RPC
constexpr uint16_t version = 1;
constexpr size_t max_bytes = 4u << 20;

template<class Archive, class Header> void fields(Archive &a, Header &h) {
    a(h.model, h.pack, h.plan, h.epoch, h.session, h.sequence, h.position, h.next_layer, h.rows);
}
inline bool equal(const DeepSeekBoundaryHeader &a, const DeepSeekBoundaryHeader &b) {
    return a.model == b.model && a.pack == b.pack && a.plan == b.plan && a.epoch == b.epoch &&
        a.session == b.session && a.sequence == b.sequence && a.position == b.position &&
        a.next_layer == b.next_layer && a.rows == b.rows;
}
inline void require(bool valid, const char *message) {
    if (!valid) throw std::runtime_error(message);
}
inline uint64_t group_count(const DeepSeekBoundaryHeader &h, uint32_t ratio) {
    return (h.position + h.rows) / ratio - h.position / ratio;
}
inline void layout(const DeepSeekBoundaryLayout &l, const DeepSeekBoundaryHeader &h) {
    require(l.hidden && l.hidden <= 65536 && l.streams && l.streams <= 8 &&
        l.topk_stride && l.topk_stride <= 65536 && l.cand_stride && l.cand_stride <= 65536 &&
        l.vocab && l.max_context && h.rows && h.rows <= 16 && h.next_layer && h.next_layer < 40 &&
        h.epoch && h.session && h.sequence && h.position < l.max_context &&
        h.rows <= l.max_context - h.position, "invalid DeepSeek boundary layout or position");
    for (const auto *id : {&h.model, &h.pack, &h.plan})
        require(std::any_of(id->begin(), id->end(), [](uint8_t b) { return b != 0; }), "missing boundary identity");
    require(l.sources.size() <= 40, "too many boundary cache sources");
    uint32_t previous = 0;
    bool first = true;
    uint64_t bytes = 4ull * h.rows * (l.streams * l.hidden + 8ull + l.topk_stride + l.cand_stride + 3ull) + 160;
    for (const auto &s : l.sources) {
        require(s.layer < h.next_layer && s.ratio && s.ratio <= 65536 && s.kv_row_bytes &&
            s.kv_row_bytes <= 65536 && s.key_row_bytes <= 65536 && (first || s.layer > previous),
            "invalid or unordered boundary cache source");
        bytes += 16 + group_count(h, s.ratio) * (s.kv_row_bytes + static_cast<uint64_t>(s.key_row_bytes));
        previous = s.layer; first = false;
    }
    require(bytes <= max_bytes, "boundary layout exceeds byte limit");
}
inline void validate(const DeepSeekBoundaryLayout &l, const DeepSeekLayerBoundary &b) {
    const auto &h = b.header;
    layout(l, h);
    const size_t rows = h.rows;
    require(b.tokens.size() == rows && b.topk_valid.size() == rows && b.cand_valid.size() == rows &&
        b.h.size() == rows * l.streams * l.hidden && b.pre.size() == rows * 8 &&
        b.topk.size() == rows * l.topk_stride && b.cand.size() == rows * l.cand_stride &&
        b.caches.size() == l.sources.size(), "boundary field size mismatch");
    for (size_t t = 0; t < rows; ++t) {
        require(b.tokens[t] >= 0 && static_cast<uint32_t>(b.tokens[t]) < l.vocab &&
            b.topk_valid[t] <= l.topk_stride && b.cand_valid[t] <= l.cand_stride, "invalid boundary token or selection count");
        for (uint32_t j = 0; j < b.topk_valid[t]; ++j)
            require(b.topk[t * l.topk_stride + j] >= 0 &&
                static_cast<uint64_t>(b.topk[t * l.topk_stride + j]) < l.max_context, "invalid boundary index selection");
        for (uint32_t j = 0; j < b.cand_valid[t]; ++j)
            require(b.cand[t * l.cand_stride + j] >= 0 &&
                static_cast<uint64_t>(b.cand[t * l.cand_stride + j]) < l.max_context, "invalid boundary candidate selection");
    }
    for (const auto *v : {&b.h, &b.pre})
        for (const auto x : *v) require(std::isfinite(x), "nonfinite boundary activation");
    for (size_t i = 0; i < l.sources.size(); ++i) {
        const auto &s = l.sources[i]; const auto &c = b.caches[i];
        require(c.source_layer == s.layer && c.first_row == h.position / s.ratio && c.rows == group_count(h, s.ratio) &&
            c.kv.size() == static_cast<size_t>(c.rows) * s.kv_row_bytes &&
            c.keys.size() == static_cast<size_t>(c.rows) * s.key_row_bytes, "boundary cache interval or byte count mismatch");
    }
}
} // namespace deepseek_boundary_detail

/// Encode a validated capture in explicit little-endian scalar order. The
/// separate magic/version protect expert-v1 and native session-slot records.
inline std::vector<uint8_t> encode_deepseek_boundary(const DeepSeekBoundaryLayout &layout,
                                                    const DeepSeekLayerBoundary &boundary) {
    namespace D = deepseek_boundary_detail;
    D::validate(layout, boundary);
    ExpertWireWriter body;
    D::fields(body, boundary.header);
    auto write = [&](const auto &v) { for (auto x : v) body.put(x); };
    write(boundary.tokens); write(boundary.topk_valid); write(boundary.cand_valid);
    write(boundary.h); write(boundary.pre); write(boundary.topk); write(boundary.cand);
    body.put(static_cast<uint32_t>(boundary.caches.size()));
    for (const auto &c : boundary.caches) {
        body(c.source_layer, c.first_row, c.rows);
        write(c.kv); write(c.keys);
    }
    D::require(body.bytes.size() + 12 <= D::max_bytes, "boundary capture exceeds byte limit");
    ExpertWireWriter frame;
    frame.put(D::magic); frame.put(D::version); frame.put(uint16_t{0}); frame.put(static_cast<uint32_t>(body.bytes.size()));
    frame.bytes.insert(frame.bytes.end(), body.bytes.begin(), body.bytes.end());
    return std::move(frame.bytes);
}

/// Decode into a fresh host record, checking the complete capture against the
/// expected model/plan/session/window and trusted layout before returning it.
/// Callers must not mutate the executor until this function succeeds, and must
/// additionally compare resolved selection counts to their authoritative plan.
inline DeepSeekLayerBoundary decode_deepseek_boundary(const DeepSeekBoundaryLayout &layout,
    const DeepSeekBoundaryHeader &expected, const std::vector<uint8_t> &bytes) {
    namespace D = deepseek_boundary_detail;
    D::layout(layout, expected);
    D::require(bytes.size() >= 12 && bytes.size() <= D::max_bytes, "invalid boundary frame length");
    ExpertWireReader r{bytes};
    uint32_t magic = 0, length = 0; uint16_t version = 0, reserved = 0;
    r(magic, version, reserved, length);
    D::require(magic == D::magic && version == D::version && !reserved && length == bytes.size()-12,
               "boundary magic, version or length mismatch");
    DeepSeekLayerBoundary b;
    D::fields(r, b.header);
    D::require(r.ok && D::equal(expected, b.header), "unexpected boundary identity, window or position");
    auto read = [&](auto &v, size_t n) { v.resize(n); for (auto &x : v) r.get(x); };
    const size_t rows = expected.rows;
    read(b.tokens, rows); read(b.topk_valid, rows); read(b.cand_valid, rows);
    read(b.h, rows * layout.streams * layout.hidden); read(b.pre, rows * 8);
    read(b.topk, rows * layout.topk_stride); read(b.cand, rows * layout.cand_stride);
    uint32_t count = 0; r.get(count);
    D::require(r.ok && count == layout.sources.size(), "boundary cache source count mismatch");
    b.caches.resize(count);
    for (size_t i = 0; i < count; ++i) {
        auto &c = b.caches[i]; const auto &s = layout.sources[i];
        r(c.source_layer, c.first_row, c.rows);
        D::require(r.ok && c.source_layer == s.layer && c.first_row == expected.position / s.ratio &&
            c.rows == D::group_count(expected, s.ratio), "unexpected boundary cache interval");
        read(c.kv, static_cast<size_t>(c.rows) * s.kv_row_bytes);
        read(c.keys, static_cast<size_t>(c.rows) * s.key_row_bytes);
    }
    D::require(r.done(), "truncated boundary fields or trailing bytes");
    D::validate(layout, b);
    return b;
}
} // namespace strata::core
