// Full native reference versus independently loaded native prefix and suffix.
// This is an assigned-range oracle, not a TCP or mixed-backend inference gate.
#include "layer_probe.hpp"
#include "ds4_gguf.cuh"
#include "strata/core/expert_worker_identity.hpp"
#include <algorithm>
#include <chrono>
#include <cmath>
#include <cstdio>
#include <cstdlib>

using namespace strata::core;
using namespace strata::ds4::test;
using strata::ds4::Model;

int main(int argc, char **argv) {
    if (argc<4 || argc>5) {
        std::fprintf(stderr,"usage: deepseek_layer_partition_test PACK FIRST_GGUF CUT=20|24 [DECODE_STEPS=128]\n");
        return 2;
    }
    try {
        const int cut=number(argv[3],24), steps=argc==5 ? number(argv[4],1024) : 128;
        check((cut==20 || cut==24) && steps>0,"invalid partition cut or step count");
        check(setenv("DS4_SYNC","1",1)==0 && setenv("DS4_STREAM","0",1)==0,"cannot select native synchronous partition probe");
        const auto evidence=std::filesystem::temp_directory_path()/
            ("strata-layer-partition-"+std::to_string(std::chrono::steady_clock::now().time_since_epoch().count()));
        std::filesystem::create_directories(evidence);
        std::fprintf(stderr,"E3 partition evidence directory: %s\n",evidence.string().c_str());
        const auto mapped=strata::GgufModel::open(argv[2]);
        std::fprintf(stderr,"E3: hashing exact source mappings before native partition oracle\n");
        const auto model_hash=expert_worker_model_sha256(mapped);
        const std::string pack=argv[1];
        const auto pack_hash=digest(read(pack+"/ds4.txt")+read(pack+"/tensors.txt")+read(pack+"/engram.bin"));
        identity_receipt("E3_PARTITION",model_hash,pack_hash);
        const auto base=options(pack);
        auto a=base,b=base; a.layer_begin=0; a.layer_end=cut; b.layer_begin=cut; b.layer_end=40;
        Model full,prefix,suffix; std::string error;
        if (!full.load(base,error)) throw std::runtime_error("reference load: "+error);
        if (!prefix.load(a,error)) throw std::runtime_error("prefix load: "+error);
        if (!suffix.load(b,error)) throw std::runtime_error("suffix load: "+error);
        const bool q8k=strata::ds4::gguf::iq_q8k_activations();
        const int prompt[]={0,128803,3085,344,270,6102,294,8760,33,9361,295,834,3107,10175,16,128804,128822};
        uint64_t session=0, windows=0, traffic=0, source_rows=0;
        DeepSeekLayerBoundary reference;
        DeepSeekBoundaryHeader identity;
        identity.model=model_hash; identity.pack=pack_hash;
        identity.plan=digest("assigned native partition iq_q8k="+std::to_string(q8k)+" cut="+std::to_string(cut));
        identity.epoch=1;
        DeepSeekBoundaryLayout layout;
        bool have_reference=false;
        unsigned future_checks=0;
        auto start=[&] {
            full.reset(); prefix.reset(); suffix.reset();
            identity.session=++session; have_reference=false;
            check(suffix.configure_boundary_probe(cut,identity,{},error),error);
            layout=suffix.boundary_layout();
            check(full.configure_boundary_probe(cut,identity,[&](const DeepSeekLayerBoundary &capture,std::string &) {
                reference=capture; have_reference=true; return true;
            },error),error);
            check(prefix.configure_boundary_probe(cut,identity,[&](const DeepSeekLayerBoundary &capture,std::string &err) {
                check(have_reference,"prefix advanced without its matching native reference");
                const auto want=encode_deepseek_boundary(layout,reference);
                const auto got=encode_deepseek_boundary(layout,capture);
                if (want!=got) {
                    preserve(evidence/"boundary-reference.bin",want); preserve(evidence/"boundary-prefix.bin",got);
                    throw std::runtime_error("assigned prefix boundary differs; raw pair preserved");
                }
                const auto restored=decode_deepseek_boundary(layout,capture.header,got);
                future_checks=reject_future_selections(suffix,layout,restored,future_checks);
                if (!suffix.replay_boundary(restored,1,err)) return false;
                have_reference=false; ++windows; traffic+=got.size();
                for (const auto &cache : capture.caches) source_rows+=cache.rows;
                return true;
            },error),error);
        };
        auto advance=[&](const int *tokens, int count) {
            check(count>0 && count<=16 && !have_reference,"invalid or unfinished partition window");
            if (!full.feed(tokens,count,1,error)) return false;
            return prefix.feed(tokens,count,0,error);
        };
        auto exact_state=[&](Model &assigned, bool prefix_view) {
            std::vector<uint8_t> want,got;
            state(full,want,prefix_view); state(assigned,got,prefix_view);
            if (want!=got) {
                const std::string role=prefix_view ? "prefix" : "suffix";
                preserve(evidence/("state-reference-"+role+".bin"),want);
                preserve(evidence/("state-assigned-"+role+".bin"),got);
                throw std::runtime_error("assigned "+role+" persistent state differs; raw pair preserved");
            }
        };
        auto compare=[&] {
            std::vector<float> want,got;
            check(full.logits(want) && suffix.logits(got),"could not read partition logits");
            const auto finite=[](const auto &v) { return std::all_of(v.begin(),v.end(),[](float x){return std::isfinite(x);}); };
            if (want.empty() || want.size()!=got.size() || !finite(want) || !finite(got) ||
                std::memcmp(want.data(),got.data(),want.size()*sizeof(float))!=0) {
                preserve(evidence/"logits-reference.f32",want); preserve(evidence/"logits-partition.f32",got);
                throw std::runtime_error("assigned partition logits differ or are nonfinite; raw pair preserved");
            }
            exact_state(prefix,true); exact_state(suffix,false);
            check(full.tokens()==prefix.tokens() && full.tokens()==suffix.tokens() &&
                  full.pos()==prefix.pos() && full.pos()==suffix.pos(),"assigned token history or position differs");
        };
        start();
        // All three configure calls have checked effective per-layer counts.
        std::printf("E3_PARTITION_EXECUTORS_LOADED cut=%d effective_experts=6 iq_q8k=%d transport=in_process\n",cut,q8k);
        std::fflush(stdout);
        std::vector<uint8_t> untouched{1,2,3};
        check(!prefix.boundary_state(untouched,error) && untouched==std::vector<uint8_t>({1,2,3}),"prefix exposed suffix-owned state");
        check(!suffix.boundary_state(untouched,error,true) && untouched==std::vector<uint8_t>({1,2,3}),"suffix exposed prefix-owned state");
        check(advance(prompt,1),error); compare();
        check(advance(prompt+1,16),error); compare();
        for (int i=0;i<steps;++i) {
            const int token=full.sample(0,1,0,0,1,i);
            check(advance(&token,1),error); compare();
            std::printf("E3_PARTITION cut=%d step=%d position=%lld exact_boundary=1 exact_logits=1 exact_prefix_state=1 exact_suffix_state=1\n",
                        cut,i+1,static_cast<long long>(full.pos()));
            std::fflush(stdout);
        }
        check(cut==20 ? source_rows==0 : source_rows==static_cast<uint64_t>(full.pos()),"cross-cut cache row coverage differs");
        check(traffic>0,"no assigned boundary data exchanged");
        check(future_checks==(cut==20 ? 1u : 3u),"missing assigned future-selection rejection coverage");
        // Cancel after a suffix layer mutates state. The failed prefix cannot
        // reuse that session; all three reset with a fresh monotonically fenced ID.
        start();
        int checks=0; suffix.set_cancel([&]{return ++checks>=3;});
        check(!advance(prompt,1),"partial suffix cancellation was ignored");
        suffix.set_cancel({});
        check(!prefix.feed(prompt,1,0,error),"poisoned prefix session was reused");
        check(!suffix.boundary_state(untouched,error),"incomplete suffix state was exposed");
        start(); check(advance(prompt,1),error); compare();
        std::printf("E3_PARTITION cut=%d decode_steps=%d windows=%llu capture_bytes=%llu PASS; transport=in_process; TCP/quality gates NOT RUN\n",
                    cut,steps,static_cast<unsigned long long>(windows),static_cast<unsigned long long>(traffic));
        return 0;
    } catch (const std::exception &e) { std::fprintf(stderr,"E3 partition: %s\n",e.what()); return 1; }
}
