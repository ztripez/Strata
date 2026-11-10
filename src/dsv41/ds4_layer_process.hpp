#pragma once

#include <algorithm>
#include <array>
#include <charconv>
#include <cmath>
#include <cstdint>
#include <map>
#include <set>
#include <sstream>
#include <stdexcept>
#include <string>
#include <vector>
#include <arpa/inet.h>

namespace strata::ds4::network {
using Digest=std::array<uint8_t,32>;
/// Frozen initial serving profile; actual Model geometry must match at startup.
inline constexpr int vocabulary=129280,context=2048,prompt_limit=1900,generation_limit=1024;
inline void require(bool ok,const std::string &error) {if(!ok) throw std::runtime_error(error);}
inline uint64_t integer(const std::string &text,uint64_t maximum) {
    uint64_t value=0;const auto parsed=std::from_chars(text.data(),text.data()+text.size(),value);
    require(!text.empty() && parsed.ec==std::errc{} && parsed.ptr==text.data()+text.size() && value<=maximum,"invalid bounded unsigned integer");return value;
}
inline Digest unhex(const std::string &text) {
    require(text.size()==64,"SHA-256 must have 64 lowercase hexadecimal characters");Digest result{};
    auto nibble=[](char c) {return c>='0' && c<='9' ? c-'0':c>='a' && c<='f' ? c-'a'+10:-1;};
    for(size_t i=0;i<result.size();++i) {const int a=nibble(text[2*i]),b=nibble(text[2*i+1]);require(a>=0 && b>=0,"invalid SHA-256");result[i]=static_cast<uint8_t>(16*a+b);}
    require(std::any_of(result.begin(),result.end(),[](auto value){return value!=0;}),"zero SHA-256 identity");return result;
}
inline std::string hex(const Digest &bytes) {
    constexpr char digits[]="0123456789abcdef";std::string text;
    for(const auto byte:bytes) {text+=digits[byte>>4];text+=digits[byte&15];}return text;
}
inline void endpoint(const std::string &text) {
    const auto split=text.find(':');require(split!=std::string::npos && text.find(':',split+1)==std::string::npos,"numeric IPv4 endpoint required");
    in_addr address{};require(inet_pton(AF_INET,text.substr(0,split).c_str(),&address)==1,"numeric IPv4 endpoint required");
    require(integer(text.substr(split+1),65535)>0,"endpoint port must be positive");
}

/// Explicit native process configuration. This executable has no automatic fit,
/// placement, sampling or model-family defaults; an installer supplies the plan.
struct ProcessConfig {
    std::string role,pack,address,expert_address,expert_model,expert_placement;
    Digest model{},pack_identity{},plan{},peer_binary{};
    int cut=0,cpu_threads=0,rpc_ms=0,idle_ms=0,expert_rpc_ms=0,request_ms=0;
    int64_t expert_vram_mib=0;
    bool check_only=false;
};
inline ProcessConfig configuration(const std::vector<std::string> &arguments) {
    std::map<std::string,std::string> values;bool serving=false,checking=false;
    const std::set<std::string> flags{"--role","--pack","--cut","--endpoint","--model-sha256","--pack-sha256","--plan-id","--peer-binary-sha256","--experts","--expert-vram-mib","--cpu-threads","--rpc-timeout-ms","--idle-timeout-ms","--request-timeout-ms","--expert-worker","--expert-worker-model","--expert-worker-placement","--expert-worker-timeout-ms"};
    for(size_t i=0;i<arguments.size();++i) {
        const auto &flag=arguments[i];
        if(flag=="--serve") {require(!serving,"duplicate --serve");serving=true;continue;}
        if(flag=="--check-config") {require(!checking,"duplicate --check-config");checking=true;continue;}
        require(flags.contains(flag) && i+1<arguments.size(),"unknown or incomplete layer-process option: "+flag);
        require(values.emplace(flag,arguments[++i]).second,"duplicate layer-process option: "+flag);
    }
    auto take=[&](const char *flag) {const auto found=values.find(flag);require(found!=values.end() && !found->second.empty(),std::string("missing ")+flag);return found->second;};
    require(serving,"layer process requires explicit --serve");
    ProcessConfig c;c.role=take("--role");require(c.role=="prefix" || c.role=="suffix","layer role must be prefix or suffix");
    c.pack=take("--pack");c.address=take("--endpoint");endpoint(c.address);
    c.model=unhex(take("--model-sha256"));c.pack_identity=unhex(take("--pack-sha256"));c.plan=unhex(take("--plan-id"));
    require(take("--experts")=="6","layer process requires explicit --experts 6");
    c.cut=static_cast<int>(integer(take("--cut"),24));require(c.cut==20 || c.cut==24,"layer cut must be 20 or 24");
    c.cpu_threads=static_cast<int>(integer(take("--cpu-threads"),64));require(c.cpu_threads>0,"CPU thread limit must be positive");
    c.expert_vram_mib=static_cast<int64_t>(integer(take("--expert-vram-mib"),65536));
    c.rpc_ms=static_cast<int>(integer(take("--rpc-timeout-ms"),600000));require(c.rpc_ms>0,"RPC timeout must be positive");
    if(c.role=="prefix") {
        c.peer_binary=unhex(take("--peer-binary-sha256"));require(!values.contains("--idle-timeout-ms"),"idle timeout belongs to suffix");
        c.request_ms=static_cast<int>(integer(take("--request-timeout-ms"),7200000));require(c.request_ms>=c.rpc_ms,"request timeout must cover one RPC timeout");
    } else {
        require(!values.contains("--peer-binary-sha256"),"peer binary belongs to prefix");
        require(!values.contains("--request-timeout-ms"),"request timeout belongs to prefix");
        c.idle_ms=static_cast<int>(integer(take("--idle-timeout-ms"),7200000));require(c.idle_ms>0,"idle timeout must be positive");
    }
    const int expert_flags=static_cast<int>(values.contains("--expert-worker"))+static_cast<int>(values.contains("--expert-worker-model"))+static_cast<int>(values.contains("--expert-worker-placement"))+static_cast<int>(values.contains("--expert-worker-timeout-ms"));
    require(expert_flags==0 || expert_flags==4,"expert endpoint/model/placement/timeout are required together");
    if(expert_flags) {
        c.expert_address=take("--expert-worker");endpoint(c.expert_address);c.expert_model=take("--expert-worker-model");c.expert_placement=take("--expert-worker-placement");
        c.expert_rpc_ms=static_cast<int>(integer(take("--expert-worker-timeout-ms"),600000));require(c.expert_rpc_ms>0,"expert RPC timeout must be positive");
    }
    c.check_only=checking;return c;
}

/// One greedy GEN request in the existing Strata stdio protocol. Every request
/// resets model state; checkpoint, image and non-greedy controls are unsupported.
struct Request {std::vector<int> tokens;int maximum=0;};
inline Request request(const std::string &line) {
    std::istringstream input(line);std::string command,count,part;Request r;
    require(static_cast<bool>(input>>command>>count) && command=="GEN","expected GEN <max_new> [neutral keys] <id,id,...>");
    r.maximum=static_cast<int>(integer(count,generation_limit));require(r.maximum>0,"generation limit must be positive");
    std::set<std::string> seen;bool tokens=false;
    while(input>>part) {
        const auto equal=part.find('=');
        if(equal!=std::string::npos) {
            require(!tokens,"sampling key after token list");
            const auto key=part.substr(0,equal),text=part.substr(equal+1);require(seen.insert(key).second,"duplicate request control");
            if(key=="seed") {(void)integer(text,UINT64_MAX);continue;}
            double value=0;const auto parsed=std::from_chars(text.data(),text.data()+text.size(),value);
            require(!text.empty() && parsed.ec==std::errc{} && parsed.ptr==text.data()+text.size() && std::isfinite(value),"invalid request control");
            const bool neutral=(key=="temperature" && value<=0) || ((key=="top_p" || key=="penalty_repeat") && value==1) ||
                (key=="top_k" && (value==0 || value==64)) || ((key=="min_p" || key=="penalty_freq" || key=="penalty_present" || key=="penalty_last_n" || key=="ckpt") && value==0);
            require(neutral,"layer serving supports greedy sampling and no checkpoints/penalties; refused "+key);continue;
        }
        require(!tokens,"multiple token lists");tokens=true;size_t offset=0;
        for(;;) {
            const auto comma=part.find(',',offset);
            r.tokens.push_back(static_cast<int>(integer(part.substr(offset,comma==std::string::npos ? comma:comma-offset),vocabulary-1)));
            require(r.tokens.size()<=prompt_limit,"prompt exceeds native 1900-token bound");
            if(comma==std::string::npos) break;
            offset=comma+1;
        }
    }
    require(tokens && r.tokens.size()+static_cast<size_t>(r.maximum)<=context,"missing prompt or request exceeds 2048-token context");return r;
}
} // namespace strata::ds4::network
