// Opt-in persistent native layer process. All model math and expert staging
// remain in strata::ds4::Model; this file owns framing, request lifetime and I/O.
#include "ds4_layer_process.hpp"
#include "ds4_layer_input.hpp"
#include "ds4_layer_trace.hpp"
#include "ds4_model.hpp"
#include "ds4_gguf.cuh"
#include "strata/core/deepseek_layer_rpc.hpp"
#include "strata/core/expert_worker_identity.hpp"
#include <openssl/rand.h>
#include <chrono>
#include <cstdlib>
#include <cstdio>
#include <filesystem>
#include <fstream>
#include <iterator>
#include <limits>
#include <poll.h>
#include <sys/socket.h>

namespace N=strata::ds4::network;
using strata::ds4::Model;
using namespace strata::core;
namespace {
using Clock=std::chrono::steady_clock;
uint64_t microseconds(Clock::time_point start,Clock::time_point end) {
    return static_cast<uint64_t>(std::chrono::duration_cast<std::chrono::microseconds>(end-start).count());
}
struct Socket {
    int fd=-1;
    explicit Socket(int value):fd(value) {}
    Socket(const Socket&)=delete;
    Socket &operator=(const Socket&)=delete;
    ~Socket() {expert_worker_shutdown(fd);expert_worker_close(fd);}
};
N::Digest binary_identity() {
    std::ifstream file("/proc/self/exe",std::ios::binary);N::require(static_cast<bool>(file),"cannot read running executable");
    std::vector<uint8_t> bytes((std::istreambuf_iterator<char>(file)),{});N::require(!file.bad() && !bytes.empty(),"cannot hash running executable");
    return expert_worker_spans_sha256({{bytes.data(),bytes.size()}});
}
strata::ds4::Options options(const N::ProcessConfig &c) {
    strata::ds4::Options o;o.pack=c.pack;o.gpus={0};o.tp=1;o.threads=false;o.max_context=N::context;
    o.experts=6;o.dspark=false;o.dspark_auto=false;o.vision=false;o.prefill_chunk=0;
    o.vram_experts=1;o.vram_experts_auto=false;o.expert_cache=false;o.ssd_cache_gb=0;o.mmap_experts=1;
    o.profile="none";o.profile_readonly=true;o.cpu_threads=c.cpu_threads;o.verbose=true;o.expert_vram_mib=c.expert_vram_mib;
    o.layer_begin=c.role=="prefix" ? 0:c.cut;o.layer_end=c.role=="prefix" ? c.cut:40;
    o.expert_worker=c.expert_address;o.expert_worker_model=c.expert_model;o.expert_worker_placement=c.expert_placement;
    if(!c.expert_address.empty()) o.expert_worker_timeout_ms=c.expert_rpc_ms;
    return o;
}
DeepSeekLayerContract contract(Model &model,const N::ProcessConfig &c) {
    DeepSeekLayerContract result;std::string error;
    N::require(model.boundary_artifacts(result.identity.model,result.identity.pack,error),error);
    N::require(result.identity.model==c.model && result.identity.pack==c.pack_identity,"actual mapped model/pack differs from installed plan");
    result.identity.plan=c.plan;result.identity.epoch=1;result.identity.next_layer=static_cast<uint32_t>(c.cut);
    result.effective_experts=6;result.arithmetic=static_cast<uint32_t>(ExpertWorkerArithmetic::DEEPSEEK_IQ_Q8K);
    auto initial=result.identity;initial.session=1;
    std::function<bool(const DeepSeekLayerBoundary&,std::string&)> observer;
    if(c.role=="prefix") observer=[](const auto &,std::string &){return true;};
    N::require(model.configure_boundary_probe(c.cut,initial,std::move(observer),error),error);
    result.layout=model.boundary_layout();model.reset();return result;
}
void info(const N::ProcessConfig &c,const N::Digest &binary) {
    std::printf("INFO engine=deepseek-layer arch=deepseek41 version=%s fabric=1 role=%s context=%d effective_experts=6 arithmetic=2 model_sha256=%s pack_sha256=%s plan_sha256=%s binary_sha256=%s cpu_threads_limit=%d expert_vram_mib=%lld request_timeout_ms=%d rpc_timeout_ms=%d idle_timeout_ms=%d\n",
        STRATA_VERSION,c.role.c_str(),N::context,N::hex(c.model).c_str(),N::hex(c.pack_identity).c_str(),N::hex(c.plan).c_str(),N::hex(binary).c_str(),c.cpu_threads,static_cast<long long>(c.expert_vram_mib),c.request_ms,c.rpc_ms,c.idle_ms);
    if(c.trace_windows) std::puts("INFO window_trace=1");
    std::fflush(stdout);
}
int suffix(Model &model,const N::ProcessConfig &c,const DeepSeekLayerContract &installed,const N::Digest &binary) {
    DeepSeekLayerCapabilities caps;caps.contract=installed;caps.binary_sha256=binary;
#if defined(STRATA_USE_HIP)
    caps.backend=2;
#else
    caps.backend=1;
#endif
    N::require(RAND_bytes(reinterpret_cast<unsigned char*>(&caps.incarnation),sizeof(caps.incarnation))==1 && caps.incarnation,"cannot create layer service incarnation");
    caps.last_session=1;bool native_failed=false;
    DeepSeekLayerService service(caps,[&](const auto &identity,std::string &error) {
        native_failed=true;model.reset();const bool ok=model.configure_boundary_probe(c.cut,identity,{},error);native_failed=!ok;return ok;
    },[&](const auto &capture,uint32_t rows,DeepSeekLayerResult &result,std::string &error) {
        const auto started=Clock::now();const auto deadline=started+std::chrono::milliseconds(c.rpc_ms);
        model.set_cancel([deadline]{return Clock::now()>=deadline;});
        bool ok=false;
        try {ok=model.replay_boundary(capture,static_cast<int>(rows),error);}
        catch(...) {native_failed=true;model.set_cancel({});throw;}
        model.set_cancel({});
        if(!ok || Clock::now()>=deadline) {native_failed=true;if(error.empty()) error="native layer computation deadline exceeded";return false;}
        if(!model.logits(result.logits)) {native_failed=true;error="native suffix did not publish logits";return false;}
        if(c.trace_windows) {
            try {N::suffix_window_trace(stderr,c,binary,capture.header,caps.incarnation,microseconds(started,Clock::now()));}
            catch(...) {native_failed=true;throw;}
        }
        result.request=capture.header;result.logits_rows=rows;result.committed_position=static_cast<uint64_t>(model.pos());return true;
    });
    std::string host,error;uint16_t port=0,bound=0;
    N::require(expert_worker_parse_host_port(c.address,host,port),"invalid suffix endpoint");
    Socket listener(expert_worker_listen(host,port,1,&bound,&error));N::require(listener.fd>=0,error);
    info(c,binary);std::printf("LAYER_READY endpoint=%s incarnation=%llu\n",c.address.c_str(),static_cast<unsigned long long>(caps.incarnation));std::fflush(stdout);
    for(;;) {
        pollfd waiting{listener.fd,POLLIN,0};const int ready=::poll(&waiting,1,c.idle_ms);
        if(ready<0 && errno==EINTR) continue;
        N::require(ready>0,"suffix accept deadline/error");
        Socket peer(::accept(listener.fd,nullptr,nullptr));N::require(peer.fd>=0,"suffix accept failed");
        native_failed=false;error.clear();const bool served=service.serve(peer.fd,c.rpc_ms,error);model.set_cancel({});model.reset();
        std::fprintf(stderr,"FABRIC_SESSION session_floor=%llu completed=%d native_failed=%d reason=%s\n",static_cast<unsigned long long>(service.capabilities().last_session),served,native_failed,error.c_str());
        N::require(!native_failed,"native suffix failure: "+error);
        // A disconnected or cancelled request is visible above. Only a new
        // connection with a higher fenced session may prepare clean model state.
    }
}
struct Cancelled {};
struct Reset {
    Model &model;
    ~Reset() {model.set_cancel({});model.reset();}
};
struct Generation {int count=0;double prompt_ms=0,decode_ms=0;std::string finish="length";};
Generation generate(Model &model,const N::ProcessConfig &c,const DeepSeekLayerContract &installed,const N::Digest &binary,N::CommandInput &input,const N::Request &request,uint64_t &floor) {
    Generation output;const auto start=Clock::now();auto decode=start;
    DeepSeekLayerClient client;DeepSeekLayerResult last;
    Clock::time_point feed_started,callback_started,callback_ended;
    int callbacks=0;DeepSeekLayerTiming transaction;
    N::InputBinding binding(input,[&]{client.interrupt();});Reset reset{model};std::string error;
    auto check_cancel=[&] {if(!input.failure().empty()) throw std::runtime_error(input.failure());if(input.cancelled()) throw Cancelled{};};
    try {
        check_cancel();N::require(client.connect(c.address,installed,c.peer_binary,c.rpc_ms,error),error);check_cancel();
        floor=std::max(floor,client.capabilities().last_session);N::require(floor<std::numeric_limits<uint64_t>::max()-1,"layer session IDs exhausted");
        auto identity=installed.identity;identity.session=++floor;
        N::require(client.begin(floor,error),error);check_cancel();
        model.reset();N::require(model.configure_boundary_probe(c.cut,identity,[&](const auto &capture,std::string &why) {
            if(c.trace_windows) {callback_started=Clock::now();++callbacks;}
            const bool ok=client.execute(capture,1,last,why,c.trace_windows ? &transaction:nullptr);
            if(c.trace_windows) callback_ended=Clock::now();
            return ok;
        },error),error);
        model.set_cancel([&]{return input.cancelled();});
        auto advance=[&](const int *ids,int count) {
            check_cancel();
            if(c.trace_windows) {feed_started=Clock::now();callbacks=0;}
            const bool ok=model.feed(ids,count,0,error);
            Clock::time_point feed_ended;
            if(c.trace_windows) feed_ended=Clock::now();
            if(!ok && input.cancelled() && (model.cancelled() || error.find("layer channel interrupted:")!=std::string::npos)) throw Cancelled{};
            N::require(ok,error);check_cancel();
            N::require(last.committed_position==static_cast<uint64_t>(model.pos()) && last.logits.size()==N::vocabulary,"acknowledged layer position/logits differ");
            N::require(std::all_of(last.logits.begin(),last.logits.end(),[](float value){return std::isfinite(value);}),"nonfinite acknowledged logits");
            if(c.trace_windows) {
                N::require(callbacks==1,"window trace requires exactly one acknowledged boundary per feed");
                N::PrefixWindowTiming measured;
                measured.feed_us=microseconds(feed_started,feed_ended);
                measured.before_rpc_us=microseconds(feed_started,callback_started);
                measured.rpc_us=microseconds(callback_started,callback_ended);
                measured.after_rpc_us=microseconds(callback_ended,feed_ended);
                measured.transaction=transaction;
                N::prefix_window_trace(stderr,c,binary,last.request,client.capabilities().incarnation,measured);
            }
            return static_cast<int>(std::max_element(last.logits.begin(),last.logits.end())-last.logits.begin());
        };
        int next=-1;
        for(size_t offset=0;offset<request.tokens.size();) {
            const int count=static_cast<int>(std::min<size_t>(16,request.tokens.size()-offset));next=advance(request.tokens.data()+offset,count);offset+=static_cast<size_t>(count);
            const double elapsed=std::chrono::duration<double,std::milli>(Clock::now()-start).count();
            std::printf("PP %zu %zu %.1f\n",offset,request.tokens.size(),elapsed);std::fflush(stdout);
        }
        decode=Clock::now();output.prompt_ms=std::chrono::duration<double,std::milli>(decode-start).count();
        while(output.count<request.maximum) {
            check_cancel();std::printf("T %d\n",next);std::fflush(stdout);++output.count;
            if(next==1) {output.finish="stop";break;}
            // Match the qualified window path, including its final extra row.
            next=advance(&next,1);
        }
        check_cancel();N::require(client.close(error),error);
    } catch(const Cancelled &) {N::require(input.failure().empty(),input.failure());output.finish="cancel";}
      catch(const std::exception &) {
        if(input.failure().empty() && input.cancelled() && error.find("layer channel interrupted:")!=std::string::npos) output.finish="cancel";
        else throw;
    }
    output.decode_ms=std::chrono::duration<double,std::milli>(Clock::now()-decode).count();return output;
}
int prefix(Model &model,const N::ProcessConfig &c,const DeepSeekLayerContract &installed,const N::Digest &binary) {
    // Discover and pin the actual suffix before the HTTP engine can see READY.
    {DeepSeekLayerClient probe;std::string error;N::require(probe.connect(c.address,installed,c.peer_binary,c.rpc_ms,error) && probe.close(error),error);}
    N::CommandInput input(STDIN_FILENO,c.request_ms);info(c,binary);std::printf("READY %d stop\n",N::context);std::fflush(stdout);
    uint64_t floor=1;std::string line;
    while(input.next(line)) {
        N::Request request;
        try {request=N::request(line);} catch(const std::exception &error) {input.finish();std::printf("ERR %s\n",error.what());std::fflush(stdout);continue;}
        Generation output;
        try {output=generate(model,c,installed,binary,input,request,floor);} catch(...) {input.finish();throw;}
        N::require(input.failure().empty(),input.failure());input.finish();
        std::printf("DONE %d %zu %.1f %.1f %s 0 0 0 0 0\n",output.count,request.tokens.size(),output.prompt_ms,output.decode_ms,output.finish.c_str());std::fflush(stdout);
    }
    N::require(input.failure().empty(),input.failure());return 0;
}
} // namespace
int main(int argc,char **argv) {
    try {
        const N::ProcessConfig config=N::configuration({argv+1,argv+argc});
        if(config.check_only) {std::puts("CONFIG_CHECKED syntax_only; model/device/quality NOT VERIFIED");return 0;}
        N::require(strata::ds4::gguf::iq_q8k_activations(),"layer process requires explicit DS4_IQ_Q8K=1");
        N::require(setenv("DS4_SYNC","1",1)==0 && setenv("DS4_STREAM","0",1)==0,"cannot set native synchronous profile");
        N::require(cudaSetDevice(0)==cudaSuccess,"cannot select native device");
        Model model;std::string error;N::require(model.load(options(config),error),error);
        N::require(model.vocab()==N::vocabulary && model.max_context()==N::context && model.seq_count()==1,"actual native geometry differs from the serving profile");
        const auto installed=contract(model,config);
        const auto binary=binary_identity();
        return config.role=="prefix" ? prefix(model,config,installed,binary):suffix(model,config,installed,binary);
    } catch(const std::exception &error) {std::fprintf(stderr,"layer process failed: %s\n",error.what());std::printf("ERR layer process failed: %s\n",error.what());std::fflush(stdout);return 1;}
}
