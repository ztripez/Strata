// Delivery-state and output-record tests. Mock callbacks do not prove model math.
#include "strata/core/deepseek_layer_session.hpp"
#include <cstdio>
#include <limits>
#include <stdexcept>
using namespace strata::core;
namespace {
void check(bool ok, const char *why) { if (!ok) throw std::runtime_error(why); }
const DeepSeekBoundaryLayout layout{8,4,4,4,32,256,{{20,1,32,16}}};
DeepSeekLayerBoundary capture(uint64_t sequence=1,uint64_t position=0,uint32_t rows=3) {
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
    int calls=0; uint64_t position=0;
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
template<class F> void refuse(F f,const char *why) {
    bool rejected=false; try {f();} catch(const std::runtime_error &) {rejected=true;}
    check(rejected,why);
}
void put32(std::vector<uint8_t> &bytes,size_t at,uint32_t n) {
    for(size_t i=0;i<4;++i) bytes.at(at+i)=static_cast<uint8_t>(n>>(i*8));
}
}
int main() {
    try {
        const auto first=capture(); std::string error;
        {
            NativeMock native; DeepSeekLayerSession session(layout,first.header,24); DeepSeekLayerResult r,duplicate;
            check(session.execute(first,1,std::ref(native),r,error),error.c_str());
            check(native.calls==1 && native.position==3 && session.awaiting_acknowledgement(),"initial delivery did not commit once");
            const auto bytes=encode_deepseek_layer_result(layout,r);
            check(std::vector<uint8_t>(bytes.begin(),bytes.begin()+8)==std::vector<uint8_t>({0x44,0x53,0x34,0x52,1,0,0,0}),"DS4R golden prefix");
            check(bytes.size()==160+32*4 && bytes[148]==3 && bytes[156]==1,"DS4R golden position/rows/length");
            check(encode_deepseek_layer_result(layout,decode_deepseek_layer_result(layout,first.header,1,bytes))==bytes,"result roundtrip lost bits");
            check(session.execute(first,1,std::ref(native),duplicate,error) && native.calls==1 && native.position==3,
                  "duplicate re-executed native state");
            check(encode_deepseek_layer_result(layout,duplicate)==bytes,"duplicate did not return retained result");
            check(session.acknowledge(first.header,error) && !session.awaiting_acknowledgement(),"ACK did not release result");
            const auto second=capture(2,3,16);
            check(session.execute(second,16,std::ref(native),r,error) && native.calls==2 && native.position==19,"next acknowledged window failed");
            const auto full=encode_deepseek_layer_result(layout,r);
            check(decode_deepseek_layer_result(layout,second.header,16,full).logits.size()==16*32,"sixteen-row result failed");
            for (size_t at : {size_t{0},size_t{4},size_t{6},size_t{8},size_t{12},size_t{108},size_t{116},size_t{124},size_t{132},size_t{140},size_t{148},size_t{156}}) {
                auto bad=bytes; bad[at]^=0x10;
                refuse([&]{decode_deepseek_layer_result(layout,first.header,1,bad);},"accepted altered result identity/descriptor");
            }
            auto bad=bytes; put32(bad,160,0x7fc00000);
            refuse([&]{decode_deepseek_layer_result(layout,first.header,1,bad);},"accepted NaN result");
            for (size_t size : {size_t{0},size_t{12},size_t{159},bytes.size()-1,bytes.size()+1}) {
                auto short_frame=bytes;short_frame.resize(size);
                refuse([&]{decode_deepseek_layer_result(layout,first.header,1,short_frame);},"accepted truncated/trailing result");
            }
            refuse([&]{decode_deepseek_layer_result(layout,first.header,2,bytes);},"accepted wrong requested result rows");
            auto large=layout; large.vocab=0xffffffffu;
            refuse([&]{decode_deepseek_layer_result(large,first.header,1,bytes);},"untrusted shape escaped result byte limit");
        }
        // Every conflict gets a fresh session because failure is terminal.
        for(int fault=0;fault<5;++fault) {
            NativeMock native;DeepSeekLayerSession session(layout,first.header,24);DeepSeekLayerResult r;
            check(session.execute(first,1,std::ref(native),r,error),error.c_str());
            auto next=first;uint32_t rows=1;
            if(fault==0) next.h[0]+=1;
            if(fault==1) next=capture(2,3,1); // next before ACK
            if(fault==2) rows=2; // conflicting output request for identical capture
            if(fault==3) {check(session.acknowledge(first.header,error),error.c_str());} // late duplicate
            if(fault==4) ++next.header.epoch;
            check(!session.execute(next,rows,std::ref(native),r,error) && session.aborted() && native.calls==1 && native.position==3,
                  "delivery conflict mutated native state or remained healthy");
            check(!session.acknowledge(first.header,error),"aborted session accepted ACK");
        }
        for(int fault=0;fault<9;++fault) {
            NativeMock native;DeepSeekLayerSession session(layout,first.header,24);auto bad=first;
            DeepSeekLayerResult r;r.logits={42};r.committed_position=99;uint32_t rows=1;
            if(fault==0) ++bad.header.session;
            if(fault==1) ++bad.header.position;
            if(fault==2) ++bad.header.sequence;
            if(fault==3) bad.caches[0].kv.pop_back();
            if(fault==4) bad.h[0]=std::numeric_limits<float>::infinity();
            if(fault==5) bad.header.plan[0]^=1;
            if(fault==6) bad.topk_valid[0]=99;
            if(fault==7) rows=0;
            if(fault==8) rows=4;
            check(!session.execute(bad,rows,std::ref(native),r,error) && session.aborted() && native.calls==0 && native.position==0,
                  "invalid capture reached native executor");
            check(r.logits==std::vector<float>({42}) && r.committed_position==99,"rejected delivery changed output");
        }
        for(int fault=0;fault<6;++fault) {
            DeepSeekLayerSession session(layout,first.header,24);DeepSeekLayerResult r;int calls=0;
            auto broken=[&](const DeepSeekLayerBoundary &b,uint32_t rows,DeepSeekLayerResult &out,std::string &why) {
                ++calls;out.request=b.header;out.committed_position=b.header.position+b.header.rows;out.logits_rows=rows;
                out.logits.assign(rows*layout.vocab,1);
                if(fault==0) {why="native partial cancellation";return false;}
                if(fault==1) throw std::runtime_error("native exception");
                if(fault==2) ++out.committed_position;
                if(fault==3) out.logits[0]=std::numeric_limits<float>::quiet_NaN();
                if(fault==4) ++out.request.sequence;
                if(fault==5) {out.logits_rows=2;out.logits.resize(2*layout.vocab);}
                return true;
            };
            check(!session.execute(first,1,broken,r,error) && calls==1 && session.aborted(),"native failure published a healthy session");
            check(!session.execute(first,1,broken,r,error) && calls==1,"native failure was retried");
        }
        {
            DeepSeekLayerSession session(layout,first.header,24);DeepSeekLayerResult result;
            check(!session.execute(first,1,{},result,error) && session.aborted(),"accepted a missing native executor");
            auto invalid=layout;invalid.max_context=0;
            refuse([&]{DeepSeekLayerSession bad(invalid,first.header,24);},"accepted invalid session geometry");
        }
        for(int fault=0;fault<3;++fault) {
            DeepSeekLayerSession session(layout,first.header,24);NativeMock native;DeepSeekLayerResult r;auto ack=first.header;
            if(fault) check(session.execute(first,1,std::ref(native),r,error),error.c_str());
            if(fault==1) ++ack.session;
            if(fault==2) check(session.acknowledge(first.header,error),error.c_str());
            check(!session.acknowledge(ack,error) && session.aborted(),"invalid or duplicate ACK accepted");
        }
        std::puts("DeepSeek layer delivery/result contracts PASS; native execution and TCP integration remain separate");
        return 0;
    } catch(const std::exception &e) {std::fprintf(stderr,"%s\n",e.what());return 1;}
}
