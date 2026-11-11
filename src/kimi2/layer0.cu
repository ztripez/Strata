// Experimental Kimi K2 layer adapter. Reuses Strata's F32 dense, RMSNorm and SwiGLU kernels.
#include "ds4_kernels.cuh"
#include "strata/kimi2/engine.hpp"
#include "strata/kimi2/model.hpp"
#include "ds4_iq.cuh"
#include "strata/kernels/quantize_act.hpp"
#include <cuda_runtime.h>
#include <cmath>
#include <cstdio>
#include <fstream>
#include <stdexcept>
#include <string>
#include <vector>

namespace {
void check(cudaError_t e) { if (e != cudaSuccess) throw std::runtime_error(cudaGetErrorString(e)); }
struct Buffer {
    float* p = nullptr;
    size_t n;
    explicit Buffer(size_t count): n(count) { check(cudaMalloc(&p, n * sizeof(float))); }
    ~Buffer() { if(p) cudaFree(p); }
    Buffer(const Buffer&) = delete;
    Buffer& operator=(const Buffer&) = delete;
};
void load(Buffer& b, const std::string& path) {
    std::ifstream f(path, std::ios::binary | std::ios::ate);
    if (!f || static_cast<size_t>(f.tellg()) != b.n * sizeof(float)) throw std::runtime_error("wrong fixture size: " + path);
    f.seekg(0); std::vector<float> v(b.n);
    if (!f.read(reinterpret_cast<char*>(v.data()), static_cast<std::streamsize>(v.size()*sizeof(float)))) throw std::runtime_error("fixture read failed: " + path);
    check(cudaMemcpy(b.p, v.data(), v.size()*sizeof(float), cudaMemcpyHostToDevice));
}
void save(const Buffer& b, const std::string& path) {
    std::vector<float> v(b.n);check(cudaMemcpy(v.data(), b.p, v.size()*sizeof(float),cudaMemcpyDeviceToHost));
    for(float f:v) if(!std::isfinite(f)) throw std::runtime_error("nonfinite GPU output");
    std::ofstream out(path,std::ios::binary);out.write(reinterpret_cast<char*>(v.data()),static_cast<std::streamsize>(v.size()*sizeof(float)));
    if(!out) throw std::runtime_error("output write failed");
}
__global__ void split(const float* kv,float* latent,int T) {
    int i=blockIdx.x*blockDim.x+threadIdx.x;if(i<T*512) latent[i]=kv[(i/512)*576+i%512];
}
__global__ void add(float* x,const float* y,int n) {
    int i=blockIdx.x*blockDim.x+threadIdx.x;if(i<n)x[i]+=y[i];
}
// Undo GGUF's absorbed K transpose and concatenate the original K/V projections.
__global__ void combine_kv(const float* k,const float* v,float* out) {
    const int i=blockIdx.x*blockDim.x+threadIdx.x;if(i>=64*256*512)return;
    const int h=i/(256*512),d=(i/512)%256,l=i%512;
    out[i]=d<128?k[(h*512+l)*128+d]:v[(h*128+d-128)*512+l];
}
// Standard causal K2 MLA after latent expansion. Q has 128 nonrotary + 64 adjacent rotary channels.
__global__ void attention(const float* q,const float* kv,const float* kp,float* out,int T,int past,float scale) {
    int h=blockIdx.x,t=blockIdx.y;int v=threadIdx.x;if(v>=128)return;
    float maximum=-INFINITY,denominator=0,value=0;
    for(int s=0;s<=past+t;++s) {
        float score=0;
        for(int d=0;d<128;++d)score+=q[(t*64+h)*192+d]*kv[(s*64+h)*256+d];
        for(int d=0;d<64;++d)score+=q[(t*64+h)*192+128+d]*kp[s*64+d];
        score*=scale;
        const float next=fmaxf(maximum,score),old=expf(maximum-next),weight=expf(score-next);
        denominator=denominator*old+weight;
        value=value*old+weight*kv[(s*64+h)*256+128+v];maximum=next;
    }
    out[(t*64+h)*128+v]=value/denominator;
}
}

static std::vector<float> run_layer_impl(const std::string& dir,const std::string& output,const std::vector<float>& input,
                                          bool packed,bool expert,bool moe_layer,const std::string& experts, strata::kimi2::MLAState* state=nullptr, strata::core::NetworkExpertExecutor* remote=nullptr, int layer=1, const strata::kimi2::Model* model=nullptr) {
        constexpr int H=7168;
        const int T=input.size()/H;
        if(T<1 || T>8 || input.size()!=static_cast<size_t>(T*H))throw std::runtime_error("K2 window requires one to eight complete hidden rows");
        const int past=state?state->latent.size()/512:0;
        if(state && (state->latent.size()!=static_cast<size_t>(past*512) || state->rotary.size()!=static_cast<size_t>(past*64)))throw std::runtime_error("invalid compressed MLA state");
        int dense_tokens=T;
        const std::string tensor_prefix="blk."+std::to_string(layer)+".";
        auto load_weight=[&](Buffer& buffer,const std::string& name,const std::string& fixture){
            if(!model){load(buffer,fixture);return;}
            const std::string tensor=tensor_prefix+name+".weight";
            if(model->bytes(tensor)!=buffer.n*4)throw std::runtime_error("wrong mapped tensor size: "+tensor);
            check(cudaMemcpy(buffer.p,model->data(tensor),buffer.n*4,cudaMemcpyHostToDevice));
        };
        auto dense=[&](const char* name,int rows,int cols,const Buffer& x,Buffer& y) {
            if(packed && std::string(name)=="kv_b") {
                constexpr size_t count=64*512*128,rawbytes=count/32*34;
                Buffer kraw(rawbytes/4),vraw(rawbytes/4),k(count),v(count),combined(2*count);
                load_weight(kraw,"attn_k_b",dir+"/attn_k_b.quant");load_weight(vraw,"attn_v_b",dir+"/attn_v_b.quant");
                strata::kernels::dequant_q8_0(reinterpret_cast<const uint8_t*>(kraw.p),k.p,count,nullptr);
                strata::kernels::dequant_q8_0(reinterpret_cast<const uint8_t*>(vraw.p),v.p,count,nullptr);
                combine_kv<<<(2*count+255)/256,256>>>(k.p,v.p,combined.p);
                strata::ds4::gemv_f32(combined.p,x.p,rows,cols,dense_tokens,y.p,rows,nullptr);
                check(cudaGetLastError());check(cudaDeviceSynchronize());return;
            }
            if(packed && std::string(name)!="kv_b") {
                int type=0;
                if(model)type=model->type(tensor_prefix+name+".weight");
                else{std::ifstream metadata(dir+"/"+name+".type");metadata>>type;
                    if(!metadata) throw std::runtime_error("missing packed tensor type");}
                if(!strata::ds4::gguf::iq::supported(type)) throw std::runtime_error("unsupported quantized tensor type");
                const size_t rowbytes=strata::ds4::gguf::iq::row_bytes(type,cols);
                if(!rowbytes) throw std::runtime_error("unsupported quantized tensor");
                Buffer raw((static_cast<size_t>(rows)*rowbytes+3)/4);
                load_weight(raw,name,dir+"/"+name+".quant");
                Buffer decoded(static_cast<size_t>(rows)*cols);
                if(!strata::ds4::gguf::iq::dequant_rows_f32(type,raw.p,rowbytes,cols,rows,decoded.p,nullptr))
                    throw std::runtime_error("native quant decode refused tensor");
                strata::ds4::gemv_f32(decoded.p,x.p,rows,cols,dense_tokens,y.p,rows,nullptr);
                check(cudaGetLastError());check(cudaDeviceSynchronize());return;
            }
            Buffer w(static_cast<size_t>(rows)*cols);load(w,dir+"/"+name+".f32");
            strata::ds4::gemv_f32(w.p,x.p,rows,cols,dense_tokens,y.p,rows,nullptr);check(cudaGetLastError());check(cudaDeviceSynchronize());
        };
        auto norm=[&](const char* name,Buffer& x,int n,int stride) {
            Buffer w(n);
            if(model && model->type(tensor_prefix+name+".weight")!=0)throw std::runtime_error("K2 normalization must use original F32 tensor");
            load_weight(w,name,dir+"/"+name+".f32");strata::ds4::Finish f;f.w=w.p;f.eps=1e-6f;
            strata::ds4::finish(x.p,T,n,stride,f,nullptr);check(cudaGetLastError());check(cudaDeviceSynchronize());
        };
        if(expert) {
            Buffer x(T*H),gate(T*2048),up(T*2048),hidden(T*2048),out(T*H);
            check(cudaMemcpy(x.p,input.data(),input.size()*sizeof(float),cudaMemcpyHostToDevice));dense("gate",2048,H,x,gate);dense("up",2048,H,x,up);
            strata::ds4::swiglu(gate.p,up.p,T*2048,0.0f,hidden.p,nullptr);
            dense("down",H,2048,hidden,out);save(out,output+".expert_output.f32");
            std::printf("Kimi K2 native quantized expert0 completed: tokens=%d hidden=%d\n",T,H);
            std::vector<float> result(T*H);check(cudaMemcpy(result.data(),out.p,result.size()*4,cudaMemcpyDeviceToHost));return result;
        }
        Buffer x(T*H),normalized(T*H),qa(T*1536),q(T*64*192),kva(T*576),latent((past+T)*512),kv((past+T)*64*256),att(T*8192),projected(T*H);
        check(cudaMemcpy(x.p,input.data(),input.size()*sizeof(float),cudaMemcpyHostToDevice));check(cudaMemcpy(normalized.p,x.p,x.n*sizeof(float),cudaMemcpyDeviceToDevice));norm("attn_norm",normalized,H,H);
        save(normalized,output+".input_norm.f32");dense("attn_q_a",1536,H,normalized,qa);norm("attn_q_a_norm",qa,1536,1536);dense("attn_q_b",12288,1536,qa,q);
        dense("attn_kv_a_mqa",576,H,normalized,kva);norm("attn_kv_a_norm",kva,512,576);
        split<<<(T*512+255)/256,256>>>(kva.p,latent.p+past*512,T);
        if(past)check(cudaMemcpy(latent.p,state->latent.data(),past*512*4,cudaMemcpyHostToDevice));
        dense_tokens=past+T;dense("kv_b",16384,512,latent,kv);dense_tokens=T;
        Buffer freqs(32);
        if(model)check(cudaMemcpy(freqs.p,model->rope_frequencies().data(),32*4,cudaMemcpyHostToDevice));else load(freqs,dir+"/rope_freqs.f32");strata::ds4::Finish rope;rope.rd=64;rope.pos=past;rope.freqs=freqs.p;rope.vpp=64;
        strata::ds4::finish(q.p,T*64,192,192,rope,nullptr);rope.vpp=1;
        strata::ds4::finish(kva.p+512,T,64,576,rope,nullptr);
        Buffer keys((past+T)*64);
        if(past)check(cudaMemcpy(keys.p,state->rotary.data(),past*64*4,cudaMemcpyHostToDevice));
        for(int t=0;t<T;++t)check(cudaMemcpy(keys.p+(past+t)*64,kva.p+t*576+512,64*4,cudaMemcpyDeviceToDevice));
        float mscale=1.0f+0.1f*std::log(32.0f);attention<<<dim3(64,T),128>>>(q.p,kv.p,keys.p,att.p,T,past,mscale*mscale/std::sqrt(192.0f));
        dense("attn_output",H,8192,att,projected);save(projected,output+".attention_output.f32");add<<<(T*H+255)/256,256>>>(x.p,projected.p,T*H);
        check(cudaMemcpy(normalized.p,x.p,x.n*sizeof(float),cudaMemcpyDeviceToDevice));norm("ffn_norm",normalized,H,H);
        if(moe_layer) {
            std::vector<float> normed(T*H);check(cudaMemcpy(normed.data(),normalized.p,normed.size()*4,cudaMemcpyDeviceToHost));
            const auto mixed=strata::kimi2::run_moe(experts,normed,output+".moe",remote,layer,model);
            check(cudaMemcpy(projected.p,mixed.data(),mixed.size()*4,cudaMemcpyHostToDevice));save(projected,output+".mlp_output.f32");
        }else{
        Buffer gate(T*18432),up(T*18432),hidden(T*18432);dense("ffn_gate",18432,H,normalized,gate);dense("ffn_up",18432,H,normalized,up);
        strata::ds4::swiglu(gate.p,up.p,T*18432,0.0f,hidden.p,nullptr);dense("ffn_down",H,18432,hidden,projected);save(projected,output+".mlp_output.f32");
        }
        add<<<(T*H+255)/256,256>>>(x.p,projected.p,T*H);check(cudaGetLastError());check(cudaDeviceSynchronize());save(x,output+".output.f32");
        if(state){state->latent.resize((past+T)*512);state->rotary.resize((past+T)*64);check(cudaMemcpy(state->latent.data(),latent.p,state->latent.size()*4,cudaMemcpyDeviceToHost));check(cudaMemcpy(state->rotary.data(),keys.p,state->rotary.size()*4,cudaMemcpyDeviceToHost));}
        std::vector<float> result(T*H);check(cudaMemcpy(result.data(),x.p,result.size()*4,cudaMemcpyDeviceToHost));return result;
}
std::vector<float> strata::kimi2::run_layer(int layer,const std::string& weights,const std::string& experts,
                                          const std::vector<float>& input,const std::string& capture) {
    if(layer<0 || layer>1)throw std::runtime_error("K2 bring-up profile currently binds layers 0 and 1");
    return run_layer_impl(weights,capture,input,true,false,layer==1,experts);
}
std::vector<float> strata::kimi2::run_layer(int layer,const std::string& weights,const std::string& experts,
    const std::vector<float>& input,const std::string& capture,MLAState& state,core::NetworkExpertExecutor* remote) {
    if(layer<0 || layer>=61)throw std::runtime_error("invalid K2 layer");
    return run_layer_impl(weights,capture,input,true,false,layer>0,experts,&state,remote,layer);
}
std::vector<float> strata::kimi2::run_layer(int layer,const Model& model,const std::vector<float>& input,
    const std::string& capture,MLAState& state,core::NetworkExpertExecutor* remote){
    if(layer<0 || layer>=61)throw std::runtime_error("invalid mapped K2 layer");
    return run_layer_impl("",capture,input,true,false,layer>0,"",&state,remote,layer,&model);
}
std::vector<float> strata::kimi2::logits(const Model& model,const std::vector<float>& hidden){
    constexpr int H=7168,V=163840;const int T=hidden.size()/H;
    if(T<1 || T>8 || hidden.size()!=static_cast<size_t>(T*H))throw std::runtime_error("invalid final hidden rows");
    if(model.type("output_norm.weight")!=0)throw std::runtime_error("unsupported final norm type");
    Buffer x(hidden.size()),norm(H),raw((model.bytes("output.weight")+3)/4),weights(static_cast<size_t>(V)*H),out(T*V);
    check(cudaMemcpy(x.p,hidden.data(),hidden.size()*4,cudaMemcpyHostToDevice));
    check(cudaMemcpy(norm.p,model.data("output_norm.weight"),H*4,cudaMemcpyHostToDevice));
    strata::ds4::Finish f;f.w=norm.p;f.eps=1e-6f;strata::ds4::finish(x.p,T,H,H,f,nullptr);
    const int type=model.type("output.weight");
    check(cudaMemcpy(raw.p,model.data("output.weight"),model.bytes("output.weight"),cudaMemcpyHostToDevice));
    if(!strata::ds4::gguf::iq::dequant_rows_f32(type,raw.p,strata::ds4::gguf::iq::row_bytes(type,H),H,V,weights.p,nullptr))throw std::runtime_error("unsupported original output head type");
    strata::ds4::gemv_f32(weights.p,x.p,V,H,T,out.p,V,nullptr);check(cudaGetLastError());check(cudaDeviceSynchronize());
    std::vector<float> result(T*V);check(cudaMemcpy(result.data(),out.p,result.size()*4,cudaMemcpyDeviceToHost));
    for(float v:result)if(!std::isfinite(v))throw std::runtime_error("nonfinite full-model logit");return result;
}
#ifndef STRATA_KIMI2_LIBRARY
int main(int argc,char** argv) {
    try {
        if(argc!=3 && argc!=4)throw std::runtime_error("usage: strata-kimi2-layer FIXTURE OUTPUT_PREFIX [--packed-dense|--expert]");
        const bool packed=argc==4,expert=packed && std::string(argv[3])=="--expert";
        if(packed && !expert && std::string(argv[3])!="--packed-dense")throw std::runtime_error("unknown option");
        std::vector<float> input(3*7168);std::ifstream f(std::string(argv[1])+"/input.f32",std::ios::binary);
        f.read(reinterpret_cast<char*>(input.data()),input.size()*4);if(!f)throw std::runtime_error("input read failed");
        run_layer_impl(argv[1],argv[2],input,packed,expert,false,"");return 0;
    } catch(const std::exception& e) { std::fprintf(stderr,"kimi2: %s\n",e.what());return 1; }
}
#endif
