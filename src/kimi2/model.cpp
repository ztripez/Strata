#include "strata/kimi2/model.hpp"
#include "strata/artifact/gguf_reader.hpp"
#include "ggml.h"
#include <algorithm>
#include <cmath>
#include <stdexcept>
namespace strata::kimi2 {
const TensorInfo& Model::tensor(const std::string& name) const {
    size_t shard=0;const auto* t=mapped_->find(name,&shard);
    if(!t || !mapped_->in_bounds(*t,shard))throw std::runtime_error("missing/out-of-bounds K2 tensor: "+name);
    return *t;
}
int Model::type(const std::string& name) const { return tensor(name).type; }
size_t Model::bytes(const std::string& name) const { return tensor_payload_bytes(tensor(name)); }
const uint8_t* Model::data(const std::string& name) const {
    const auto& t=tensor(name);size_t shard=0;mapped_->find(name,&shard);return mapped_->shard(shard).tensor_data(t);
}
Model::Model(const std::string& path):mapped_(std::make_shared<GgufModel>(GgufModel::open(path))) {
    const auto& meta=mapped_->meta();const auto* arch=meta.get("general.architecture");
    if(!arch || arch->s!="deepseek2")throw std::runtime_error("K2 requires original deepseek2 GGUF geometry");
    auto require=[&](const char* key,double expected){const auto* v=meta.get(key);
        if(!v || std::abs(v->num()-expected)>std::max(1e-9,std::abs(expected)*1e-6))throw std::runtime_error(std::string("K2 metadata mismatch: ")+key);};
    require("tokenizer.ggml.eos_token_id",163586);
    require("deepseek2.block_count",61);require("deepseek2.embedding_length",7168);
    require("deepseek2.expert_count",384);require("deepseek2.expert_used_count",8);
    require("deepseek2.expert_feed_forward_length",2048);require("deepseek2.leading_dense_block_count",1);
    require("deepseek2.attention.head_count",64);require("deepseek2.attention.q_lora_rank",1536);
    require("deepseek2.attention.kv_lora_rank",512);require("deepseek2.vocab_size",163840);
    require("deepseek2.expert_gating_func",2);require("deepseek2.expert_weights_norm",1);
    require("deepseek2.expert_weights_scale",2.827);require("deepseek2.rope.freq_base",50000);
    require("deepseek2.rope.scaling.factor",32);require("deepseek2.rope.scaling.original_context_length",4096);
    auto shape=[&](const std::string& name,std::vector<uint64_t> expected){
        if(tensor(name).shape!=expected)throw std::runtime_error("K2 tensor geometry mismatch: "+name);};
    shape("token_embd.weight",{7168,163840});shape("output.weight",{7168,163840});shape("output_norm.weight",{7168});
    for(int l=0;l<61;++l){
        const std::string p="blk."+std::to_string(l)+".";
        shape(p+"attn_norm.weight",{7168});shape(p+"attn_q_a.weight",{7168,1536});
        shape(p+"attn_q_a_norm.weight",{1536});shape(p+"attn_q_b.weight",{1536,12288});
        shape(p+"attn_kv_a_mqa.weight",{7168,576});shape(p+"attn_kv_a_norm.weight",{512});
        shape(p+"attn_k_b.weight",{128,512,64});shape(p+"attn_v_b.weight",{512,128,64});
        shape(p+"attn_output.weight",{8192,7168});shape(p+"ffn_norm.weight",{7168});
        if(!l){shape(p+"ffn_gate.weight",{7168,18432});shape(p+"ffn_up.weight",{7168,18432});shape(p+"ffn_down.weight",{18432,7168});}
        else{
            shape(p+"ffn_gate_inp.weight",{7168,384});shape(p+"exp_probs_b.bias",{384});
            shape(p+"ffn_gate_exps.weight",{7168,2048,384});shape(p+"ffn_up_exps.weight",{7168,2048,384});shape(p+"ffn_down_exps.weight",{2048,7168,384});
            shape(p+"ffn_gate_shexp.weight",{7168,2048});shape(p+"ffn_up_shexp.weight",{7168,2048});shape(p+"ffn_down_shexp.weight",{2048,7168});
        }
    }
    constexpr double pi=3.14159265358979323846;
    const double correction=64*std::log(4096/(2*pi))/(2*std::log(50000.));
    const float lo=std::max(std::floor(correction),0.),hi=std::min(std::ceil(correction),63.);
    for(int i=0;i<32;++i){const float ramp=std::clamp((i-lo)/(hi-lo),0.f,1.f);
        rope_[i]=(1.f/std::pow(50000.f,static_cast<float>(2*i)/64))*((1-ramp)+ramp/32);}
}
int32_t Model::eos_token_id() const { return static_cast<int32_t>(mapped_->meta().get("tokenizer.ggml.eos_token_id")->u); }
const uint8_t* Model::expert(int layer,const std::string& role,int id) const {
    if(layer<1 || layer>=61 || id<0 || id>=384)throw std::runtime_error("invalid native K2 expert address");
    const std::string name="blk."+std::to_string(layer)+".ffn_"+role+"_exps.weight";
    const auto& t=tensor(name);return data(name)+static_cast<size_t>(id)*(tensor_payload_bytes(t)/384);
}
std::vector<float> Model::embeddings(const std::vector<int32_t>& ids) const {
    const auto& t=tensor("token_embd.weight");const auto* tr=ggml_get_type_traits(static_cast<ggml_type>(t.type));
    if(!tr || !tr->to_float)throw std::runtime_error("missing native embedding decoder");
    const size_t row=tensor_payload_bytes(t)/163840;const auto* bytes=data(t.name);std::vector<float> out(ids.size()*7168);
    for(size_t i=0;i<ids.size();++i){if(ids[i]<0 || ids[i]>=163840)throw std::runtime_error("invalid K2 token ID");
        tr->to_float(bytes+ids[i]*row,out.data()+i*7168,7168);}
    return out;
}
}
