// Real allocation probe for E3. A successful load is not inference acceptance.
#include "ds4_model.hpp"
#include "ds4_gguf.cuh"
#include <charconv>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <fstream>
#include <stdexcept>

namespace {
void check(bool ok, const std::string &why) { if (!ok) throw std::runtime_error(why); }
int number(const char *text) {
    int n=0; const auto end=text+std::strlen(text); const auto result=std::from_chars(text,end,n);
    check(result.ec==std::errc() && result.ptr==end,"invalid layer endpoint"); return n;
}
void memory(const char *phase) {
    size_t free=0,total=0;
    check(cudaMemGetInfo(&free,&total)==cudaSuccess,"device memory query failed");
    std::printf("E3_LOAD_MEMORY phase=%s free_bytes=%zu total_bytes=%zu\n",phase,free,total);
    std::ifstream in("/proc/self/status"); check(static_cast<bool>(in),"cannot read process memory status");
    std::string line;
    while (std::getline(in,line))
        if (line.starts_with("VmRSS:") || line.starts_with("VmHWM:") || line.starts_with("RssAnon:") ||
            line.starts_with("RssFile:") || line.starts_with("VmLck:"))
            std::printf("E3_LOAD_PROCESS phase=%s %s\n",phase,line.c_str());
    std::ifstream smaps("/proc/self/smaps_rollup"); check(static_cast<bool>(smaps),"cannot read process PSS");
    while (std::getline(smaps,line)) if (line.starts_with("Pss:"))
        std::printf("E3_LOAD_PROCESS phase=%s %s\n",phase,line.c_str());
    std::fflush(stdout);
}
}
int main(int argc, char **argv) {
    if (argc!=4) { std::fprintf(stderr,"usage: deepseek_layer_range_load_test PACK BEGIN END\n"); return 2; }
    try {
        check(setenv("DS4_SYNC","1",1)==0 && setenv("DS4_STREAM","0",1)==0,"could not select synchronous probe");
        strata::ds4::Options o;
        o.pack=argv[1]; o.layer_begin=number(argv[2]); o.layer_end=number(argv[3]);
        o.gpus={0}; o.tp=1; o.threads=false; o.max_context=2048;
        o.experts=6; o.dspark=false; o.dspark_auto=false; o.vision=false; o.prefill_chunk=0;
        o.vram_experts=0; o.vram_experts_auto=false; o.expert_cache=false; o.ssd_cache_gb=0;
        o.mmap_experts=1; o.profile="none"; o.profile_readonly=true; o.cpu_threads=2; o.verbose=true;
        check(cudaSetDevice(0)==cudaSuccess,"device selection failed");
        memory("before");
        for (int mode=0;mode<4;++mode) {
            auto bad=o;
            if (mode==0) bad.layer_begin=1;
            if (mode==1) bad.layer_end=0;
            if (mode==2) bad.experts=5;
            if (mode==3) bad.dspark=true;
            strata::ds4::Model rejected; std::string error;
            check(!rejected.load(bad,error) && !error.empty(),"unsupported range/configuration was loaded");
        }
        {
            strata::ds4::Model model; std::string error;
            check(model.load(o,error),error); memory("loaded");
            // A range cannot accidentally execute the complete graph or migrate
            // a native all-layer checkpoint before a boundary session exists.
            check(!model.step(0,false,error) && model.pos()==0,"unconfigured range advanced its state");
            bool refused=false;
            try { model.checkpoint(); } catch (const std::logic_error &) { refused=true; }
            check(refused,"range accepted a native full-model checkpoint");
            std::vector<uint8_t> slot{1,2,3};
            check(model.slot_save(slot,error)==0 && slot==std::vector<uint8_t>({1,2,3}),"range emitted a native slot");
            check(!model.slot_load(slot.data(),slot.size(),0,error),"range restored a native slot");
            const int token=0;
            check(!model.feed_mark(&token,1,0,0,error) && model.pos()==0,"checkpointed feed mutated a range");
            if (o.layer_begin==0) {
                std::vector<float> logits;
                check(!model.logits(logits),"prefix exposed an unowned output head");
            }
            model.reset();
            std::printf("E3_ASSIGNED_LOAD layers=[%d,%d) effective_experts=6 iq_q8k=%d PASS; inference NOT RUN\n",
                        o.layer_begin,o.layer_end,strata::ds4::gguf::iq_q8k_activations());
        }
        memory("released");
        return 0;
    } catch (const std::exception &e) { std::fprintf(stderr,"E3 assigned load: %s\n",e.what()); return 1; }
}
