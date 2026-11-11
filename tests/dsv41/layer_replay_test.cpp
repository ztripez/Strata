// Actual same-backend E2 probe: full executor versus a second suffix executor.
// Both invoke Strata's existing layers, CPU experts and packed attention state.
#include "layer_probe.hpp"
#include "ds4_gguf.cuh"
#include "strata/core/expert_worker_identity.hpp"
#include <openssl/evp.h>
#include <algorithm>
#include <array>
#include <chrono>
#include <cmath>
#include <charconv>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <fstream>
#include <filesystem>
#include <iterator>
#include <stdexcept>

namespace {
using strata::ds4::Model;
using namespace strata::core;
using namespace strata::ds4::test;
}

int main(int argc, char **argv) {
    if (argc < 3 || argc > 5) {
        std::fprintf(stderr,"usage: deepseek_layer_replay_test PACK FIRST_GGUF [DECODE_STEPS=128] [CUT=0|20|24]\n"); return 2;
    }
    try {
        const int steps = argc >= 4 ? number(argv[3],1024) : 128;
        const int only_cut = argc >= 5 ? number(argv[4],24) : 0;
        check(steps>0 && (only_cut==0 || only_cut==20 || only_cut==24),"steps must be positive and cut 0/20/24");
        const auto evidence=std::filesystem::temp_directory_path()/
            ("strata-layer-replay-"+std::to_string(std::chrono::steady_clock::now().time_since_epoch().count()));
        std::filesystem::create_directories(evidence);
        std::fprintf(stderr,"E2 mismatch evidence directory: %s\n",evidence.string().c_str());
        check(setenv("DS4_SYNC","1",1)==0,"could not select synchronous local replay");
        const auto mapped = strata::GgufModel::open(argv[2]);
        std::fprintf(stderr,"E2: hashing exact source mappings before two-executor load\n");
        const auto model_hash=expert_worker_model_sha256(mapped);
        const std::string pack=argv[1];
        const auto pack_hash=digest(read(pack+"/ds4.txt")+read(pack+"/tensors.txt")+read(pack+"/engram.bin"));
        identity_receipt("E2",model_hash,pack_hash);
        const auto o=options(pack);
        Model full, suffix; std::string error;
        if (!full.load(o,error)) throw std::runtime_error("full load: "+error);
        if (!suffix.load(o,error)) throw std::runtime_error("suffix load: "+error);
        const bool q8k=strata::ds4::gguf::iq_q8k_activations();
        const int prompt[]={0,128803,3085,344,270,6102,294,8760,33,9361,295,834,3107,10175,16,128804,128822};
        uint64_t next_session=1;
        for (int cut : {20,24}) {
            if (only_cut && cut!=only_cut) continue;
            full.reset(); suffix.reset();
            DeepSeekBoundaryHeader identity;
            identity.model=model_hash; identity.pack=pack_hash;
            identity.plan=digest("E2 sync CPU-expert replay iq_q8k="+std::to_string(q8k)+" cut="+std::to_string(cut));
            identity.epoch=1; identity.session=next_session++;
            check(suffix.configure_boundary_probe(cut,identity,{},error),error);
            const auto layout=suffix.boundary_layout();
            check(layout.sources.size()==(cut==20?0u:1u),"unexpected audited source count");
            int windows=0; uint64_t source_rows=0;
            unsigned future_checks=0;
            check(full.configure_boundary_probe(cut,identity,[&](const DeepSeekLayerBoundary &capture,std::string &err) {
                const auto bytes=encode_deepseek_boundary(layout,capture);
                const auto restored=decode_deepseek_boundary(layout,capture.header,bytes);
                future_checks=reject_future_selections(suffix,layout,restored,future_checks);
                if (!windows) {
                    std::vector<uint8_t> before,after; state(suffix,before);
                    auto wrong=restored; ++wrong.header.epoch;
                    check(!suffix.replay_boundary(wrong,1,err),"accepted stale epoch");
                    state(suffix,after); check(before==after,"invalid capture mutated state");
                    wrong=restored; wrong.h.pop_back();
                    check(!suffix.replay_boundary(wrong,1,err),"accepted a short residual field");
                    state(suffix,after); check(before==after,"invalid shape mutated state");
                    if (!restored.caches.empty()) {
                        wrong=restored; wrong.caches.clear();
                        check(!suffix.replay_boundary(wrong,1,err),"accepted a missing cache update");
                        state(suffix,after); check(before==after,"missing cache update mutated state");
                    }
                    wrong=restored; ++wrong.topk_valid[0];
                    std::fill(wrong.topk.begin(),wrong.topk.begin()+wrong.topk_valid[0],0);
                    check(!suffix.replay_boundary(wrong,1,err) && err.find("selection counts")!=std::string::npos,
                          "accepted counts inconsistent with the model plan");
                    state(suffix,after); check(before==after,"invalid count mutated state");
                    suffix.set_cancel([]{return true;});
                    check(!suffix.replay_boundary(restored,1,err),"ignored pre-mutation cancellation");
                    suffix.set_cancel({}); state(suffix,after); check(before==after,"cancel mutated state");
                    err.clear();
                }
                if (!suffix.replay_boundary(restored,1,err)) return false;
                std::vector<uint8_t> before,after; state(suffix,before);
                check(!suffix.replay_boundary(restored,1,err),"executed duplicate capture");
                state(suffix,after); check(before==after,"duplicate mutated state"); err.clear();
                for (const auto &cache : restored.caches) source_rows+=cache.rows;
                ++windows; return true;
            },error),error);
            // Both configure calls have checked every effective layer count.
            std::printf("E2_EXECUTORS_LOADED cut=%d effective_experts=6 synchronous=1 gpu_expert_cache=0 ram_store=0 iq_q8k=%d\n",cut,q8k);
            std::fflush(stdout);
            auto compare=[&] {
                std::vector<float> a,b;
                check(full.logits(a) && suffix.logits(b),"could not read logits");
                const auto finite=[](const auto &v) { return std::all_of(v.begin(),v.end(),[](float x){return std::isfinite(x);}); };
                if (a.empty() || a.size()!=b.size() || !finite(a) || !finite(b) ||
                    std::memcmp(a.data(),b.data(),a.size()*sizeof(float))!=0) {
                    std::fprintf(stderr,"E2 logits mismatch: cut=%d position=%lld rows=%zu/%zu\n",cut,
                        static_cast<long long>(full.pos()),a.size(),b.size());
                    preserve(evidence/("logits-full-cut"+std::to_string(cut)+".f32"),a);
                    preserve(evidence/("logits-suffix-cut"+std::to_string(cut)+".f32"),b);
                    throw std::runtime_error("replayed logits are nonfinite or not bit-identical; raw pair preserved");
                }
                std::vector<uint8_t> sa,sb; state(full,sa); state(suffix,sb);
                if (sa!=sb) {
                    std::fprintf(stderr,"E2 state mismatch: cut=%d position=%lld bytes=%zu/%zu\n",cut,
                        static_cast<long long>(full.pos()),sa.size(),sb.size());
                    preserve(evidence/("state-full-cut"+std::to_string(cut)+".bin"),sa);
                    preserve(evidence/("state-suffix-cut"+std::to_string(cut)+".bin"),sb);
                    throw std::runtime_error("replayed persistent state is not bit-identical; raw pair preserved");
                }
                check(full.tokens()==suffix.tokens() && full.pos()==suffix.pos(),"replayed token history or position differs");
            };
            check(full.feed(prompt,1,1,error),error); compare();
            check(full.feed(prompt+1,16,1,error),error); compare();
            for (int i=0; i<steps; ++i) {
                const int token=full.sample(0,1,0,0,1,i);
                check(full.step(token,true,error),error); compare();
                std::printf("E2 cut=%d step=%d position=%lld exact_logits=1 exact_state=1\n",cut,i+1,static_cast<long long>(full.pos()));
                std::fflush(stdout);
            }
            check(cut==20 ? source_rows==0 : source_rows==static_cast<uint64_t>(full.pos()),"source update row coverage differs");
            check(future_checks==(cut==20 ? 1u : 3u),"missing future-selection rejection coverage");
            std::printf("E2 cut=%d windows=%d decode_steps=%d source_rows=%llu PASS\n",cut,windows,steps,static_cast<unsigned long long>(source_rows));
            // Fail after the first suffix layer has mutated its ring. Both
            // executors must then reset into a new session; no partial retry.
            full.reset(); suffix.reset();
            check(!suffix.configure_boundary_probe(cut,identity,{},error),"reused an old session identity");
            identity.session=next_session++;
            check(suffix.configure_boundary_probe(cut,identity,{},error),error);
            DeepSeekLayerBoundary failed_capture;
            check(full.configure_boundary_probe(cut,identity,[&](const DeepSeekLayerBoundary &capture,std::string &err) {
                failed_capture=capture;
                return suffix.replay_boundary(capture,1,err);
            },error),error);
            int cancel_checks=0;
            suffix.set_cancel([&]{ return ++cancel_checks>=3; });
            check(!full.step(prompt[0],true,error),"ignored cancellation after layer-state mutation");
            suffix.set_cancel({});
            check(!suffix.replay_boundary(failed_capture,1,error),"reused a poisoned suffix session");
            std::vector<uint8_t> invalid_state;
            check(!suffix.boundary_state(invalid_state,error),"exposed an incomplete suffix state");
            full.reset(); suffix.reset();
            check(!suffix.replay_boundary(failed_capture,1,error),"reused a reset session");
            identity.session=next_session++;
            check(suffix.configure_boundary_probe(cut,identity,{},error),error);
            check(full.configure_boundary_probe(cut,identity,[&](const DeepSeekLayerBoundary &capture,std::string &err) {
                return suffix.replay_boundary(capture,1,err);
            },error),error);
            check(!suffix.replay_boundary(failed_capture,1,error),"accepted a previous session's capture");
            check(full.step(prompt[0],true,error),error); compare();
            std::printf("E2 cut=%d partial-cancel/reset/stale-session checks PASS\n",cut);
        }
        std::puts(steps>=128 && !only_cut ? "E2 long-replay arithmetic PASS; protocol/fault/hardware gates remain separate" :
                  "E2 bounded replay probe PASS; full two-cut 128-step gate NOT RUN");
        return 0;
    } catch (const std::exception &e) { std::fprintf(stderr,"E2 replay: %s\n",e.what()); return 1; }
}
