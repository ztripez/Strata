// CPU dispatch/control oracle. No GPU, model files, sockets or model-quality claim.
#include "ds4_worker_cpu.hpp"
#include "ggml-cpu.h"
#include "ggml.h"
#include <cstdio>
#include <cstring>
#include <limits>
#include <stdexcept>

namespace {
namespace G=strata::ds4::gguf;
void check(bool pass,const std::string &error) {if(!pass) throw std::runtime_error(error);}
std::vector<uint8_t> weights(ggml_type type,int rows,int columns,int phase) {
    std::vector<float> values(static_cast<size_t>(rows)*columns);
    for(size_t i=0;i<values.size();++i) values[i]=.04f*std::sin(static_cast<float>(i+phase)*.13f);
    std::vector<uint8_t> encoded(ggml_row_size(type,columns)*static_cast<size_t>(rows));
    check(ggml_quantize_chunk(type,values.data(),encoded.data(),0,rows,columns,nullptr)==encoded.size(),"fixture quantization failed");
    return encoded;
}
}
int main() {
    try {
        ggml_cpu_init();unsigned comparisons=0,rejections=0;
        for(int pair=0;pair<2;++pair) {
            const int width=512,ff=pair ? 512:288;
            const auto gu=pair ? GGML_TYPE_IQ4_NL:GGML_TYPE_Q4_K;
            const auto down=pair ? GGML_TYPE_Q4_K:GGML_TYPE_IQ4_NL;
            G::IqCpu format;std::string error;
            check(G::iq_cpu_fmt(gu,down,width,ff,format,error),error);
            auto gate=weights(gu,ff,width,3),up=weights(gu,ff,width,11),dn=weights(down,width,ff,31);
            check(gate.size()==format.gu_row*ff && dn.size()==format.d_row*width,"fixture row sizes differ");
            std::vector<float> x(width),gold_hidden(ff),gold(width);
            for(int i=0;i<width;++i) x[i]=.4f*std::cos(static_cast<float>(i)*.19f);
            std::vector<uint8_t> activation(format.act_bytes),gold_q(format.h_bytes);
            G::iq_cpu_quant(format.gu_act,x.data(),width,activation.data());
            const void *input[1]={activation.data()};float *intermediate[1]={gold_hidden.data()};
            G::iq_cpu_gate_up(format,gate.data(),up.data(),input,1,intermediate,0,ff,1.3f);
            G::iq_cpu_quant(format.d_act,gold_hidden.data(),ff,gold_q.data());
            const void *hidden_input[1]={gold_q.data()};float *reference[1]={gold.data()};
            G::iq_cpu_down(format,dn.data(),hidden_input,1,reference,0,width);
            check(std::all_of(gold.begin(),gold.end(),[](float value){return std::isfinite(value);}),"native full-row reference is not finite");
            std::vector<uint8_t> packed;
            for(const auto *part:{&gate,&up,&dn}) packed.insert(packed.end(),part->begin(),part->end());
            for(int workers=0;workers<2;++workers) {
                strata::ds4::Pool pool(workers);
                for(bool ram:{false,true}) {
                    const std::array<const uint8_t*,3> views=ram
                        ? std::array<const uint8_t*,3>{packed.data(),packed.data()+gate.size(),packed.data()+gate.size()+up.size()}
                        : std::array<const uint8_t*,3>{gate.data(),up.data(),dn.data()};
                    std::vector<float> hidden(ff,-12),actual(width,-13);
                    std::vector<uint8_t> hq(format.h_bytes,0xa5);
                    check(strata::ds4::worker::execute_cpu_rows(format,pool,views,activation,hidden,hq,actual,1.3f,error),error);
                    check(std::memcmp(actual.data(),gold.data(),actual.size()*sizeof(float))==0 &&
                          std::memcmp(hidden.data(),gold_hidden.data(),hidden.size()*sizeof(float))==0 && hq==gold_q,
                          "pooled CPU dispatch differs from native full-row execution");
                    ++comparisons;
                }
                std::vector<float> hidden(ff,-12),actual(width,-13);
                std::vector<uint8_t> hq(format.h_bytes,0xa5);
                const auto unchanged_hidden=hidden,unchanged_output=actual;const auto unchanged_hq=hq;
                for(int bad=0;bad<7;++bad) {
                    auto shape=format;std::array<const uint8_t*,3> views{gate.data(),up.data(),dn.data()};
                    std::span<const uint8_t> a=activation;std::span<float> h=hidden,y=actual;std::span<uint8_t> q=hq;
                    float limit=1.3f;
                    if(bad==0) views[1]=nullptr;
                    if(bad==1) a=a.first(a.size()-1);
                    if(bad==2) h=h.first(h.size()-1);
                    if(bad==3) q=q.first(q.size()-1);
                    if(bad==4) y=y.first(y.size()-1);
                    if(bad==5) limit=std::numeric_limits<float>::quiet_NaN();
                    if(bad==6) shape={};
                    error.clear();
                    check(!strata::ds4::worker::execute_cpu_rows(shape,pool,views,a,h,q,y,limit,error) && !error.empty(),"invalid CPU dispatch accepted");
                    check(hidden==unchanged_hidden && actual==unchanged_output && hq==unchanged_hq,"invalid CPU dispatch mutated scratch/output");
                    ++rejections;
                }
            }
        }
        check(comparisons==8 && rejections==28,"CPU dispatch fixture coverage changed");
        std::printf("DeepSeek native CPU dispatch: %u exact serial/parallel separate/packed comparisons; %u nonmutating refusals PASS; no GPU/model inference\n",comparisons,rejections);
        return 0;
    } catch(const std::exception &error) {std::fprintf(stderr,"DeepSeek CPU dispatch: %s\n",error.what());return 1;}
}
