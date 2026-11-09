// GPU-free capture-record tests; these do not establish executor replay parity.
#include "strata/core/deepseek_layer_boundary.hpp"
#include <cstdio>
#include <limits>

using namespace strata::core;
namespace {
void check(bool ok, const char *why) { if (!ok) throw std::runtime_error(why); }
void put32(std::vector<uint8_t> &bytes, size_t at, uint32_t value) {
    for (size_t i=0; i<4; ++i) bytes.at(at+i)=static_cast<uint8_t>(value>>(8*i));
}
template<class F> void reject(F operation, const char *why) {
    bool refused = false;
    try { operation(); } catch (const std::runtime_error &) { refused = true; }
    check(refused, why);
}
DeepSeekLayerBoundary record(const DeepSeekBoundaryLayout &l, uint32_t rows, uint64_t position) {
    DeepSeekLayerBoundary b;
    auto &h = b.header;
    h.model[0]=1; h.pack[0]=2; h.plan[0]=3;
    h.epoch=1; h.session=2; h.sequence=3; h.position=position; h.rows=rows; h.next_layer=24;
    b.tokens.assign(rows,42); b.topk_valid.assign(rows,1); b.cand_valid.assign(rows,1);
    b.h.resize(static_cast<size_t>(rows)*l.streams*l.hidden);
    for (size_t i=0; i<b.h.size(); ++i) b.h[i]=static_cast<float>(i%97)/128;
    b.h[0]=-0.0f;
    b.pre.assign(static_cast<size_t>(rows)*8,.25f);
    b.topk.assign(static_cast<size_t>(rows)*l.topk_stride,0);
    b.cand.assign(static_cast<size_t>(rows)*l.cand_stride,0);
    for (const auto &s : l.sources) {
        DeepSeekBoundaryCache c;
        c.source_layer=s.layer; c.first_row=position/s.ratio;
        c.rows=static_cast<uint32_t>((position+rows)/s.ratio-c.first_row);
        c.kv.assign(static_cast<size_t>(c.rows)*s.kv_row_bytes,0xa5);
        c.keys.assign(static_cast<size_t>(c.rows)*s.key_row_bytes,0x5a);
        b.caches.push_back(std::move(c));
    }
    return b;
}
}

int main() {
    try {
        // Real residual/selection geometry; synthetic cache row widths so these
        // tests exercise framing without claiming the pack's cache ABI.
        DeepSeekBoundaryLayout l{5120,4,512,2048,129280,2048,{{2,2,32,16},{20,1,32,16}}};
        for (const auto rows : {1u,16u}) {
            const auto b=record(l,rows,127);
            const auto bytes=encode_deepseek_boundary(l,b);
            check(std::vector<uint8_t>(bytes.begin(),bytes.begin()+8)==
                std::vector<uint8_t>({0x44,0x53,0x34,0x42,1,0,0,0}),"golden magic/version/flags");
            check(bytes[12]==1 && bytes[44]==2 && bytes[76]==3 && bytes[108]==1 &&
                bytes[116]==2 && bytes[124]==3 && bytes[132]==127 && bytes[140]==24 && bytes[144]==rows,
                "golden identity/scalar offsets");
            const auto restored=decode_deepseek_boundary(l,b.header,bytes);
            check(encode_deepseek_boundary(l,restored)==bytes,"capture roundtrip lost bits");
            check(restored.caches[0].first_row==63 && restored.caches[0].rows==(rows==1?1:8),
                "partial ratio-two group interval");
            for (const auto size : {size_t{0},size_t{7},size_t{12},size_t{147},bytes.size()-1}) {
                auto bad=bytes; bad.resize(size);
                reject([&]{ decode_deepseek_boundary(l,b.header,bad); },"accepted truncated capture");
            }
            for (const auto offset : {size_t{0},size_t{4},size_t{6},size_t{8},size_t{12},size_t{108},size_t{116},size_t{124},size_t{132},size_t{140},size_t{144}}) {
                auto bad=bytes; bad[offset]^=0x10;
                reject([&]{ decode_deepseek_boundary(l,b.header,bad); },"accepted identity/frame mismatch");
            }
            auto trailing=bytes; trailing.push_back(0);
            put32(trailing,8,static_cast<uint32_t>(trailing.size()-12));
            reject([&]{ decode_deepseek_boundary(l,b.header,trailing); },"accepted trailing bytes");
            auto short_body=bytes; short_body.pop_back();
            put32(short_body,8,static_cast<uint32_t>(short_body.size()-12));
            reject([&]{ decode_deepseek_boundary(l,b.header,short_body); },"accepted internally truncated payload");
            auto bad_float=bytes; put32(bad_float,148+rows*12,0x7fc00000u);
            reject([&]{ decode_deepseek_boundary(l,b.header,bad_float); },"decoded nonfinite residual");
            auto bad_count=bytes; put32(bad_count,148+rows*4,l.topk_stride+1);
            reject([&]{ decode_deepseek_boundary(l,b.header,bad_count); },"decoded invalid selection count");
            const size_t cache_at=148+rows*12+(static_cast<size_t>(rows)*l.streams*l.hidden+rows*8+
                static_cast<size_t>(rows)*l.topk_stride+static_cast<size_t>(rows)*l.cand_stride)*4+4;
            auto bad_interval=bytes; bad_interval.at(cache_at+4)^=1;
            reject([&]{ decode_deepseek_boundary(l,b.header,bad_interval); },"decoded wrong source interval");
            auto bad=b; bad.caches.pop_back();
            reject([&]{ encode_deepseek_boundary(l,bad); },"accepted missing source update");
            bad=b; ++bad.caches[0].first_row;
            reject([&]{ encode_deepseek_boundary(l,bad); },"accepted wrong source interval");
            bad=b; bad.caches[0].kv.pop_back();
            reject([&]{ encode_deepseek_boundary(l,bad); },"accepted short source payload");
            bad=b; bad.topk_valid[0]=l.topk_stride+1;
            reject([&]{ encode_deepseek_boundary(l,bad); },"accepted invalid selection count");
            bad=b; bad.tokens[0]=static_cast<int32_t>(l.vocab);
            reject([&]{ encode_deepseek_boundary(l,bad); },"accepted invalid token");
            bad=b; bad.h[0]=std::numeric_limits<float>::quiet_NaN();
            reject([&]{ encode_deepseek_boundary(l,bad); },"accepted nonfinite residual");
            bad=b; bad.header.position=l.max_context;
            reject([&]{ encode_deepseek_boundary(l,bad); },"accepted context overflow");
            std::printf("boundary rows=%u: bit-preserving roundtrip and invalid-record checks PASS\n",rows);
        }
        auto empty_groups=record(l,1,126);
        const auto bytes=encode_deepseek_boundary(l,empty_groups);
        check(decode_deepseek_boundary(l,empty_groups.header,bytes).caches[0].rows==0,"empty group update lost");
        auto bad_layout=l; bad_layout.sources[1]=bad_layout.sources[0];
        reject([&]{ decode_deepseek_boundary(bad_layout,empty_groups.header,bytes); },"accepted duplicate cache source");
        bad_layout=l; bad_layout.topk_stride=65536;
        reject([&]{ encode_deepseek_boundary(bad_layout,record(bad_layout,16,0)); },"accepted oversized layout");
        std::puts("DeepSeek boundary capture codec: PASS (executor replay NOT RUN)");
        return 0;
    } catch (const std::exception &e) { std::fprintf(stderr,"%s\n",e.what()); return 1; }
}
