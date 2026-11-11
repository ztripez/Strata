#pragma once
#include "strata/core/deepseek_layer_session.hpp"
#include <atomic>
#include <stdexcept>

namespace strata::core::test {
inline void check(bool ok, const char *why) { if (!ok) throw std::runtime_error(why); }
inline const DeepSeekBoundaryLayout layout{8,4,4,4,32,256,{{20,1,32,16}}};
inline DeepSeekLayerBoundary capture(uint64_t sequence=1,uint64_t position=0,uint32_t rows=3) {
    DeepSeekLayerBoundary b;
    b.header.model[0]=1; b.header.pack[0]=2; b.header.plan[0]=3;
    b.header.epoch=2; b.header.session=7; b.header.sequence=sequence;
    b.header.position=position; b.header.rows=rows; b.header.next_layer=24;
    b.tokens.assign(rows,5); b.topk_valid.assign(rows,1); b.cand_valid.assign(rows,1);
    b.h.assign(static_cast<size_t>(rows)*layout.hidden*layout.streams,.25f);
    b.pre.assign(static_cast<size_t>(rows)*8,.5f);
    b.topk.assign(static_cast<size_t>(rows)*layout.topk_stride,0);
    b.cand.assign(static_cast<size_t>(rows)*layout.cand_stride,0);
    b.caches.push_back({20,rows,position,std::vector<uint8_t>(rows*32,0xa5),std::vector<uint8_t>(rows*16,0x5a)});
    return b;
}
struct NativeMock {
    std::atomic<int> calls{0}; std::atomic<uint64_t> position{0};
    bool operator()(const DeepSeekLayerBoundary &b,uint32_t rows,DeepSeekLayerResult &r,std::string &error) {
        ++calls;
        if (b.header.position!=position) {error="native position mismatch";return false;}
        position+=b.header.rows;
        r.request=b.header; r.committed_position=position; r.logits_rows=rows;
        r.logits.resize(static_cast<size_t>(rows)*layout.vocab);
        for (size_t i=0;i<r.logits.size();++i) r.logits[i]=b.h[0]+static_cast<float>(i)/128;
        return true;
    }
};
} // namespace strata::core::test
