// Bounded native layer-only harness. The native Model owns all layer math/state.
// Transport success is functional evidence; output quality remains a separate gate.
#include "layer_probe.hpp"
#include "ds4_gguf.cuh"
#include "strata/core/deepseek_layer_rpc.hpp"
#include <cerrno>
#include <cmath>
#include <poll.h>
#include <sstream>
#include <sys/random.h>
#include <sys/socket.h>

using namespace strata::core;
using namespace strata::ds4::test;
using strata::ds4::Model;
namespace {
using Clock=std::chrono::steady_clock;
namespace R=deepseek_layer_rpc_detail;
std::string hex(const std::array<uint8_t,32> &bytes) {
    constexpr char digits[]="0123456789abcdef";std::string out;
    for(const auto b:bytes) {out+=digits[b>>4];out+=digits[b&15];}return out;
}
std::array<uint8_t,32> unhex(const std::string &text) {
    check(text.size()==64,"pinned executable SHA-256 must have 64 hexadecimal characters");
    std::array<uint8_t,32> out{};
    auto digit=[](char c) {return c>='0' && c<='9' ? c-'0':c>='a' && c<='f' ? c-'a'+10:-1;};
    for(size_t i=0;i<out.size();++i) {const int a=digit(text[2*i]),b=digit(text[2*i+1]);check(a>=0 && b>=0,"invalid pinned executable SHA-256");out[i]=static_cast<uint8_t>(a*16+b);}return out;
}
uint64_t incarnation() {
    uint64_t out=0;size_t done=0;
    while(done<sizeof(out)) {
        const auto n=::getrandom(reinterpret_cast<char*>(&out)+done,sizeof(out)-done,0);
        if(n<0 && errno==EINTR) continue;
        check(n>0,"cannot obtain worker incarnation entropy");done+=static_cast<size_t>(n);
    }
    check(out!=0,"zero worker incarnation");return out;
}
struct Socket {
    explicit Socket(int descriptor):fd(descriptor) {}
    Socket(const Socket&)=delete;
    Socket &operator=(const Socket&)=delete;
    int fd;
    ~Socket() {expert_worker_shutdown(fd);expert_worker_close(fd);}
};
void memory(const char *phase) {
    size_t free=0,total=0;check(cudaMemGetInfo(&free,&total)==cudaSuccess,"device memory query failed");
    std::printf("E3_TCP_MEMORY phase=%s free_bytes=%zu total_bytes=%zu\n",phase,free,total);
    std::ifstream status("/proc/self/status");check(static_cast<bool>(status),"cannot read native memory status");
    std::string line;while(std::getline(status,line))
        if(line.starts_with("VmRSS:") || line.starts_with("VmHWM:") || line.starts_with("RssAnon:") || line.starts_with("RssFile:"))
            std::printf("E3_TCP_PROCESS phase=%s %s\n",phase,line.c_str());
    std::fflush(stdout);
}
std::vector<int> tokens(const std::string &path,int vocab) {
    std::istringstream input(read(path));std::vector<int> out;std::string item;
    while(input>>item) {const int id=number(item.c_str(),vocab-1);out.push_back(id);}
    check(!out.empty() && out.size()<=1900,"prompt must contain 1..1900 space-separated token IDs");return out;
}
int greedy(const std::vector<float> &logits,int vocab) {
    check(logits.size()==static_cast<size_t>(vocab) && std::all_of(logits.begin(),logits.end(),[](float v){return std::isfinite(v);}),"invalid greedy logit row");
    return static_cast<int>(std::max_element(logits.begin(),logits.end())-logits.begin());
}
DeepSeekLayerContract contract(Model &model,int cut) {
    DeepSeekLayerContract c;std::string e;
    std::fprintf(stderr,"E3 TCP: hashing actual loaded pack mappings; identity domains DS4M1/DS4P1\n");
    check(model.boundary_artifacts(c.identity.model,c.identity.pack,e),e);
    c.identity.epoch=1;c.identity.next_layer=static_cast<uint32_t>(cut);
    c.effective_experts=6;c.arithmetic=static_cast<uint32_t>(ExpertWorkerArithmetic::DEEPSEEK_IQ_Q8K);
    ExpertWireWriter plan;plan.put("strata-layer-only-plan/1");
    plan(c.identity.model,c.identity.pack,c.identity.next_layer,uint32_t{2048},c.effective_experts,c.arithmetic);
    c.identity.plan=digest(std::string(plan.bytes.begin(),plan.bytes.end()));
    return c;
}
void receipt(const char *role,const DeepSeekLayerContract &c,const std::array<uint8_t,32> &binary) {
    std::printf("E3_TCP_IDENTITY role=%s model_identity=DS4M1 pack_identity=DS4P1 effective_experts=6 arithmetic=%u cut=%u model_sha256=%s pack_sha256=%s plan_sha256=%s binary_sha256=%s\n",
        role,c.arithmetic,c.identity.next_layer,hex(c.identity.model).c_str(),hex(c.identity.pack).c_str(),hex(c.identity.plan).c_str(),hex(binary).c_str());
    std::fflush(stdout);
}
void run_suffix(Model &model,int cut,const std::string &endpoint,int timeout_ms,int accept_ms,const std::filesystem::path &out) {
    auto c=contract(model,cut);auto first=c.identity;first.session=1;std::string error;
    check(model.configure_boundary_probe(cut,first,{},error),error);c.layout=model.boundary_layout();model.reset();
    DeepSeekLayerCapabilities caps;caps.contract=c;caps.binary_sha256=digest(read("/proc/self/exe"));
    caps.incarnation=incarnation();caps.last_session=1;
#if defined(STRATA_USE_HIP)
    caps.backend=2;
#else
    caps.backend=1;
#endif
    uint64_t executed=0,capture_bytes=0;
    DeepSeekLayerService service(caps,[&](const auto &identity,std::string &e) {
        model.reset();return model.configure_boundary_probe(cut,identity,{},e);
    },[&](const auto &capture,uint32_t rows,DeepSeekLayerResult &result,std::string &e) {
        const auto deadline=Clock::now()+std::chrono::milliseconds(timeout_ms);
        model.set_cancel([&] {return Clock::now()>=deadline;});
        bool ok=false;
        try {ok=model.replay_boundary(capture,static_cast<int>(rows),e);} catch(...) {model.set_cancel({});throw;}
        model.set_cancel({});if(!ok) return false;
        if(Clock::now()>=deadline) {e="native suffix computation deadline exceeded";return false;}
        if(!model.logits(result.logits)) {e="native suffix did not publish logits";return false;}
        result.request=capture.header;result.logits_rows=rows;result.committed_position=static_cast<uint64_t>(model.pos());
        ++executed;capture_bytes+=encode_deepseek_boundary(c.layout,capture).size();return true;
    });
    std::string host;uint16_t port=0,bound=0;check(expert_worker_parse_host_port(endpoint,host,port),"invalid numeric suffix endpoint");
    Socket listener{expert_worker_listen(host,port,1,&bound,&error)};check(listener.fd>=0,error);
    preserve(out/"capabilities.pending",R::capability_bytes(caps));
    std::filesystem::rename(out/"capabilities.pending",out/"capabilities.bin");
    receipt("suffix",c,caps.binary_sha256);memory("ready");
    std::printf("E3_TCP_LISTEN endpoint=%s incarnation=%llu session_floor=1\n",endpoint.c_str(),static_cast<unsigned long long>(caps.incarnation));std::fflush(stdout);
    pollfd pfd{listener.fd,POLLIN,0};check(::poll(&pfd,1,accept_ms)>0,"suffix accept deadline exceeded");
    Socket peer{::accept(listener.fd,nullptr,nullptr)};check(peer.fd>=0,"suffix accept failed");
    const bool served=service.serve(peer.fd,timeout_ms,error);model.reset();
    memory("finished");check(served,error);
    std::printf("E3_TCP_SUFFIX windows=%llu capture_bytes=%llu PASS; quality/performance NOT QUALIFIED\n",
        static_cast<unsigned long long>(executed),static_cast<unsigned long long>(capture_bytes));
}
void run_generation(Model &model,const DeepSeekLayerContract &c,const std::vector<int> &prompt,int steps,bool ignore_eos,
                    const std::filesystem::path &out,DeepSeekLayerClient *client,int cut) {
    std::string error;DeepSeekLayerResult last;uint64_t windows=0,capture_bytes=0;std::vector<float> logits;
    if(client) {
        auto identity=c.identity;identity.session=client->capabilities().last_session+1;
        check(client->begin(identity.session,error),error);
        check(model.configure_boundary_probe(cut,identity,[&](const auto &capture,std::string &e) {
            DeepSeekLayerResult result;if(!client->execute(capture,1,result,e)) return false;
            // Only acknowledged output reaches the coordinator or its evidence.
            preserve(out/("result-"+std::to_string(windows+1)+".ds4r"),encode_deepseek_layer_result(c.layout,result));
            last=std::move(result);capture_bytes+=encode_deepseek_boundary(c.layout,capture).size();return true;
        },error),error);
    }
    std::ofstream history(out/"windows.jsonl");check(static_cast<bool>(history),"cannot create token/window history");
    auto advance=[&](const int *ids,int count) {
        check(model.feed(ids,count,client ? 0:1,error),error);
        if(client) {check(last.committed_position==static_cast<uint64_t>(model.pos()),"prefix/suffix committed positions differ");logits=last.logits;}
        else check(model.logits(logits),"reference has no logits");
        const int next=greedy(logits,model.vocab());
        if(!client) check(model.sample(0,1,0,0,1,windows)==next,"coordinator greedy selection differs from native sampler");
        ++windows;preserve(out/("window-"+std::to_string(windows)+".f32"),logits);
        history<<"{\"window\":"<<windows<<",\"position\":"<<model.pos()<<",\"tokens\":[";
        for(int i=0;i<count;++i) {if(i) history<<',';history<<ids[i];}
        history<<"],\"next\":"<<next<<"}\n";history.flush();check(static_cast<bool>(history),"could not persist window history");
        return next;
    };
    int next=-1;
    for(size_t offset=0;offset<prompt.size();) {
        const int count=static_cast<int>(std::min<size_t>(16,prompt.size()-offset));
        next=advance(prompt.data()+offset,count);offset+=static_cast<size_t>(count);
    }
    int generated=0;bool eos=false;
    for(;generated<steps;++generated) {
        std::printf("E3_TCP_TOKEN index=%d token=%d position=%lld\n",generated,next,static_cast<long long>(model.pos()));std::fflush(stdout);
        // Token 1 is the pinned DeepSeek pack's EOS. Long functional probes
        // explicitly continue through it; the separate EOS probe stops here.
        if(next==1 && !ignore_eos) {eos=true;++generated;break;}
        next=advance(&next,1);
    }
    if(client) check(client->close(error),error);
    memory("finished");
    std::printf("E3_TCP_GENERATION role=%s generated=%d windows=%llu capture_bytes=%llu ignore_eos=%d stop=%s PASS; quality/performance NOT QUALIFIED\n",
        client ? "prefix":"reference",generated,static_cast<unsigned long long>(windows),static_cast<unsigned long long>(capture_bytes),ignore_eos,eos ? "eos":"limit");
}
}
int main(int argc,char **argv) {
    try {
        check(argc>=4 && std::strcmp(argv[1],"--experts")==0 && std::strcmp(argv[2],"6")==0,
              "native layer campaign requires explicit --experts 6 before the role");
        argc-=2;argv+=2;
        check(argc>=2,"missing suffix/prefix/reference role");const std::string role=argv[1];
        check((role=="suffix" && argc==8) || (role=="prefix" && argc==10) || (role=="reference" && argc==7),
            "usage: suffix PACK CUT LISTEN RPC_MS ACCEPT_MS OUT | prefix PACK CUT ENDPOINT WORKER_SHA PROMPT_IDS DECODE IGNORE_EOS OUT | reference PACK PROMPT_IDS DECODE IGNORE_EOS OUT");
        check(setenv("DS4_SYNC","1",1)==0 && setenv("DS4_STREAM","0",1)==0,"cannot select native synchronous windows");
        const bool suffix=role=="suffix",prefix=role=="prefix";
        const int cut=role=="reference" ? 20:number(argv[3],24);check(cut==20 || cut==24,"unsupported native layer cut");
        const int timeout_ms=suffix ? number(argv[5],600000):600000;
        const int accept_ms=suffix ? number(argv[6],7200000):0;
        check(timeout_ms>0 && (!suffix || accept_ms>0),"invalid suffix deadline");
        if(prefix || suffix) {
            std::string host;uint16_t port=0;check(expert_worker_parse_host_port(argv[4],host,port),"invalid numeric layer endpoint");
        }
        if(prefix) (void)unhex(argv[5]);
        if(!suffix) {
            const int at=prefix ? 6:3;
            check(number(argv[at+1],1024)>0,"decode count must be positive");(void)number(argv[at+2],1);
        }
        const std::filesystem::path out=argv[argc-1];check(std::filesystem::create_directory(out),"evidence output directory already exists");
        auto o=options(argv[2]);o.verbose=true;
        if(suffix) {o.layer_begin=cut;o.layer_end=40;}
        if(prefix) {o.layer_begin=0;o.layer_end=cut;}
        check(cudaSetDevice(0)==cudaSuccess,"native device selection failed");
        Model model;std::string error;
        std::array<uint8_t,32> absent_model{42},absent_pack{43};
        const auto unchanged_model=absent_model,unchanged_pack=absent_pack;
        check(!model.boundary_artifacts(absent_model,absent_pack,error) && absent_model==unchanged_model && absent_pack==unchanged_pack,
              "unloaded native model published an artifact receipt");
        error.clear();
        memory("before");check(model.load(o,error),error);memory("loaded");
        check(strata::ds4::gguf::iq_q8k_activations(),"layer campaign requires explicit DS4_IQ_Q8K=1");
        if(suffix) {run_suffix(model,cut,argv[4],timeout_ms,accept_ms,out);return 0;}
        const int prompt_at=prefix ? 6:3;
        const auto prompt=tokens(argv[prompt_at],model.vocab());const int steps=number(argv[prompt_at+1],1024);
        const int ignore=number(argv[prompt_at+2],1);check(steps>0 && prompt.size()+static_cast<size_t>(steps)<=2048,"generation exceeds declared context");
        auto c=contract(model,cut);const auto binary=digest(read("/proc/self/exe"));receipt(role.c_str(),c,binary);
        if(prefix) {
            auto first=c.identity;first.session=1;
            check(model.configure_boundary_probe(cut,first,[](const auto &,std::string &){return true;},error),error);
            c.layout=model.boundary_layout();model.reset();
            DeepSeekLayerClient client;check(client.connect(argv[4],c,unhex(argv[5]),timeout_ms,error),error);
            run_generation(model,c,prompt,steps,ignore!=0,out,&client,cut);
        } else run_generation(model,c,prompt,steps,ignore!=0,out,nullptr,cut);
        return 0;
    } catch(const std::exception &e) {std::fprintf(stderr,"E3 TCP native: %s\n",e.what());return 1;}
}
