// Bounded native layer-only harness. The native Model owns all layer math/state.
// Transport success is functional evidence; output quality remains a separate gate.
#include "layer_probe.hpp"
#include "ds4_gguf.cuh"
#include "strata/core/deepseek_layer_rpc.hpp"
#include <arpa/inet.h>
#include <cerrno>
#include <cmath>
#include <poll.h>
#include <limits>
#include <set>
#include <sstream>
#include <sys/random.h>
#include <sys/socket.h>

using namespace strata::core;
using namespace strata::ds4::test;
using strata::ds4::Model;
namespace {
using Clock=std::chrono::steady_clock;
namespace R=deepseek_layer_rpc_detail;
bool numeric_endpoint(const std::string &endpoint) {
    std::string host;uint16_t port=0;
    if(!expert_worker_parse_host_port(endpoint,host,port)) return false;
    in_addr ipv4{};
    return inet_pton(AF_INET,host.c_str(),&ipv4)==1;
}
std::string hex(const std::array<uint8_t,32> &bytes) {
    constexpr char digits[]="0123456789abcdef";std::string out;
    for(const auto b:bytes) {out+=digits[b>>4];out+=digits[b&15];}return out;
}
std::array<uint8_t,32> unhex(const std::string &text) {
    check(text.size()==64,"pinned executable SHA-256 must have 64 hexadecimal characters");
    std::array<uint8_t,32> out{};
    auto digit=[](char c) {return c>='0' && c<='9' ? c-'0':c>='a' && c<='f' ? c-'a'+10:-1;};
    for(size_t i=0;i<out.size();++i) {const int a=digit(text[2*i]),b=digit(text[2*i+1]);check(a>=0 && b>=0,"invalid pinned executable SHA-256");out[i]=static_cast<uint8_t>(a*16+b);}
    check(std::any_of(out.begin(),out.end(),[](uint8_t b){return b!=0;}),"missing nonzero SHA-256 identity");return out;
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
    check(!out.empty() && out.size()<=1900,"token file must contain 1..1900 space-separated token IDs");return out;
}
int greedy(const std::vector<float> &logits,int vocab) {
    check(logits.size()==static_cast<size_t>(vocab) && std::all_of(logits.begin(),logits.end(),[](float v){return std::isfinite(v);}),"invalid greedy logit row");
    return static_cast<int>(std::max_element(logits.begin(),logits.end())-logits.begin());
}
DeepSeekLayerContract contract(Model &model,int cut,const std::array<uint8_t,32> &planned) {
    DeepSeekLayerContract c;std::string e;
    std::fprintf(stderr,"E3 TCP: hashing actual loaded pack mappings; identity domains DS4M1/DS4P1\n");
    check(model.boundary_artifacts(c.identity.model,c.identity.pack,e),e);
    c.identity.epoch=1;c.identity.next_layer=static_cast<uint32_t>(cut);
    c.effective_experts=6;c.arithmetic=static_cast<uint32_t>(ExpertWorkerArithmetic::DEEPSEEK_IQ_Q8K);
    ExpertWireWriter plan;plan.put("strata-layer-only-plan/1");
    plan(c.identity.model,c.identity.pack,c.identity.next_layer,uint32_t{2048},c.effective_experts,c.arithmetic);
    c.identity.plan=std::any_of(planned.begin(),planned.end(),[](uint8_t b){return b!=0;}) ? planned : digest(std::string(plan.bytes.begin(),plan.bytes.end()));
    return c;
}
void receipt(const char *role,const DeepSeekLayerContract &c,const std::array<uint8_t,32> &binary) {
    std::printf("E3_TCP_IDENTITY role=%s model_identity=DS4M1 pack_identity=DS4P1 effective_experts=6 arithmetic=%u cut=%u model_sha256=%s pack_sha256=%s plan_sha256=%s binary_sha256=%s\n",
        role,c.arithmetic,c.identity.next_layer,hex(c.identity.model).c_str(),hex(c.identity.pack).c_str(),hex(c.identity.plan).c_str(),hex(binary).c_str());
    std::fflush(stdout);
}
void run_suffix(Model &model,int cut,const std::string &endpoint,int timeout_ms,int accept_ms,int connections,const std::array<uint8_t,32> &planned,const std::filesystem::path &out) {
    auto c=contract(model,cut,planned);auto first=c.identity;first.session=1;std::string error;
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
    int aborted=0;
    for(int connection=1;connection<=connections;++connection) {
        pollfd pfd{listener.fd,POLLIN,0};check(::poll(&pfd,1,accept_ms)>0,"suffix accept deadline exceeded");
        Socket peer{::accept(listener.fd,nullptr,nullptr)};check(peer.fd>=0,"suffix accept failed");
        const uint64_t before=executed;
        error.clear();
        const bool served=service.serve(peer.fd,timeout_ms,error);model.reset();
        std::printf("E3_TCP_SESSION connection=%d session_floor=%llu windows=%llu success=%d\n",connection,
            static_cast<unsigned long long>(service.capabilities().last_session),static_cast<unsigned long long>(executed-before),served);
        if(!served) {++aborted;std::fprintf(stderr,"E3 TCP native session aborted: %s\n",error.c_str());}
        std::fflush(stdout);
    }
    memory("finished");
    std::printf("E3_TCP_SUFFIX connections=%d aborted=%d windows=%llu capture_bytes=%llu %s; quality/performance NOT QUALIFIED\n",
        connections,aborted,static_cast<unsigned long long>(executed),static_cast<unsigned long long>(capture_bytes),aborted ? "FAIL":"PASS");
    check(aborted==0,"one or more native suffix sessions aborted; per-connection failures retained");
}
void run_generation(Model &model,const DeepSeekLayerContract &c,const std::vector<int> &prompt,const std::vector<int> &targets,int steps,bool ignore_eos,
                    const std::filesystem::path &out,DeepSeekLayerClient *client,int cut,uint64_t &session_floor) {
    std::string error;DeepSeekLayerResult last;uint64_t windows=0,capture_bytes=0;std::vector<float> logits;
    if(client) {
        auto identity=c.identity;
        const uint64_t floor=std::max(session_floor,client->capabilities().last_session);
        check(floor<std::numeric_limits<uint64_t>::max()-1,"coordinator session IDs exhausted");
        identity.session=floor+1;session_floor=identity.session;
        check(client->begin(identity.session,error),error);
        check(model.configure_boundary_probe(cut,identity,[&](const auto &capture,std::string &e) {
            DeepSeekLayerResult result;if(!client->execute(capture,1,result,e)) return false;
            // Only acknowledged output reaches the coordinator or its evidence.
            preserve(out/("result-"+std::to_string(windows+1)+".ds4r"),encode_deepseek_layer_result(c.layout,result));
            last=std::move(result);capture_bytes+=encode_deepseek_boundary(c.layout,capture).size();return true;
        },error),error);
    }
    std::printf("E3_TCP_REQUEST session=%llu output=%s\n",static_cast<unsigned long long>(client ? session_floor:0),out.string().c_str());
    std::fflush(stdout);
    std::ofstream history(out/"windows.jsonl");check(static_cast<bool>(history),"cannot create token/window history");
    std::ofstream forced;
    if(!targets.empty()) {forced.open(out/"targets.jsonl");check(static_cast<bool>(forced),"cannot create teacher-forced target history");}
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
        if(!targets.empty()) {
            const int target=targets[static_cast<size_t>(generated)];
            // Score the current row against the fixed target, then consume that
            // target rather than this executor's prediction. The final advance
            // saves one extra post-target row, which is not a scored position.
            forced<<"{\"index\":"<<generated<<",\"window\":"<<windows<<",\"position\":"<<model.pos()
                  <<",\"target\":"<<target<<",\"predicted\":"<<next<<"}\n";
            forced.flush();check(static_cast<bool>(forced),"could not persist teacher-forced target history");
            std::printf("E5_TCP_TARGET index=%d target=%d predicted=%d position=%lld window=%llu\n",generated,target,next,
                static_cast<long long>(model.pos()),static_cast<unsigned long long>(windows));std::fflush(stdout);
            next=advance(&target,1);continue;
        }
        std::printf("E3_TCP_TOKEN index=%d token=%d position=%lld\n",generated,next,static_cast<long long>(model.pos()));std::fflush(stdout);
        // Token 1 is the pinned DeepSeek pack's EOS. Long functional probes
        // explicitly continue through it; the separate EOS probe stops here.
        if(next==1 && !ignore_eos) {eos=true;++generated;break;}
        next=advance(&next,1);
    }
    if(client) check(client->close(error),error);
    memory("finished");
    if(!targets.empty()) {
        std::printf("E5_TCP_TEACHER_FORCING role=%s targets=%d windows=%llu capture_bytes=%llu CAPTURED; quality/performance NOT QUALIFIED\n",
            client ? "prefix":"reference",generated,static_cast<unsigned long long>(windows),static_cast<unsigned long long>(capture_bytes));
        return;
    }
    std::printf("E3_TCP_GENERATION role=%s generated=%d windows=%llu capture_bytes=%llu ignore_eos=%d stop=%s PASS; quality/performance NOT QUALIFIED\n",
        client ? "prefix":"reference",generated,static_cast<unsigned long long>(windows),static_cast<unsigned long long>(capture_bytes),ignore_eos,eos ? "eos":"limit");
}
}
int main(int argc,char **argv) {
    try {
        check(argc>=4 && std::strcmp(argv[1],"--experts")==0 && std::strcmp(argv[2],"6")==0,
              "native layer campaign requires explicit --experts 6 before the role");
        argc-=2;argv+=2;
        auto o=options("");o.verbose=true;
        std::array<uint8_t,32> planned{};bool plan_provided=false,expert_timeout_provided=false;
        bool teacher_forced=false,decode_mode_provided=false;
        std::set<std::string> configured;
        while(argc>=2 && std::string(argv[1]).starts_with("--")) {
            check(argc>=3,"layer adapter option lacks a value");const std::string flag=argv[1],value=argv[2];
            check(configured.insert(flag).second,"repeated layer adapter option");
            if(flag=="--expert-worker") o.expert_worker=value;
            else if(flag=="--expert-worker-model") o.expert_worker_model=value;
            else if(flag=="--expert-worker-placement") o.expert_worker_placement=value;
            else if(flag=="--expert-worker-timeout") {o.expert_worker_timeout_ms=number(value.c_str(),600000);expert_timeout_provided=true;}
            else if(flag=="--plan-id") {planned=unhex(value);plan_provided=true;}
            else if(flag=="--decode-mode") {
                check(value=="greedy" || value=="teacher-forced","decode mode must be greedy or teacher-forced");
                teacher_forced=value=="teacher-forced";decode_mode_provided=true;
            }
            else if(flag=="--expert-vram-mib") {o.expert_vram_mib=number(value.c_str(),65536);o.vram_experts=1;}
            else check(false,"unknown layer adapter option");
            argc-=2;argv+=2;
        }
        const bool experts=!o.expert_worker.empty();
        check(experts==!o.expert_worker_model.empty() && experts==!o.expert_worker_placement.empty(),
              "expert-worker endpoint, GGUF model and placement are required together");
        check(!expert_timeout_provided || experts,"expert-worker timeout requires an expert worker");
        check(o.expert_vram_mib<=0 || plan_provided,"nonzero expert VRAM budget requires an explicit --plan-id");
        if(experts) {
            check(plan_provided,"hybrid layer campaign requires an explicit shared --plan-id");
            check(o.expert_worker_timeout_ms>0,"expert-worker timeout must be positive");
            check(numeric_endpoint(o.expert_worker),"invalid numeric expert-worker endpoint");
            check(std::filesystem::is_regular_file(o.expert_worker_model) && std::filesystem::is_regular_file(o.expert_worker_placement),
                  "expert-worker source GGUF or placement is absent");
        }
        check(argc>=2,"missing suffix/prefix/reference role");const std::string role=argv[1];
        const bool suffix=role=="suffix",prefix=role=="prefix",reference=role=="reference";
        check((suffix && argc==9) || (prefix && argc>=11 && (argc-7)%4==0) || (reference && argc>=7 && (argc-3)%4==0),
            "usage: suffix PACK CUT LISTEN RPC_MS ACCEPT_MS CONNECTIONS OUT | prefix PACK CUT ENDPOINT WORKER_SHA RPC_MS [PROMPT_IDS DECODE IGNORE_EOS OUT]... | reference PACK [PROMPT_IDS DECODE IGNORE_EOS OUT]...");
        check(!suffix || !decode_mode_provided,"decode mode belongs to prefix/reference requests, not the suffix service");
        check(setenv("DS4_SYNC","1",1)==0 && setenv("DS4_STREAM","0",1)==0,"cannot select native synchronous windows");
        const int cut=reference ? 20:number(argv[3],24);check(cut==20 || cut==24,"unsupported native layer cut");
        const int timeout_ms=suffix ? number(argv[5],600000):prefix ? number(argv[6],600000):600000;
        const int accept_ms=suffix ? number(argv[6],7200000):0;
        const int connections=suffix ? number(argv[7],16):0;
        check(timeout_ms>0 && (!suffix || (accept_ms>0 && connections>0)),"invalid layer deadline or connection count");
        if(prefix || suffix) {
            check(numeric_endpoint(argv[4]),"invalid numeric layer endpoint");
        }
        if(prefix) (void)unhex(argv[5]);
        struct Request {std::vector<int> prompt,targets;int steps;bool ignore;std::filesystem::path out;};
        std::vector<Request> requests;std::set<std::filesystem::path> destinations;
        if(!suffix) {
            const int first=prefix ? 7:3;check((argc-first)/4<=16,"too many requests in one bounded campaign");
            for(int at=first;at<argc;at+=4) {
                std::vector<int> targets;
                if(teacher_forced) targets=tokens(argv[at+1],129280);
                const int steps=teacher_forced ? static_cast<int>(targets.size()):number(argv[at+1],1024);
                const bool ignore=number(argv[at+2],1)!=0;
                check(!teacher_forced || (steps<=1024 && ignore),"teacher forcing requires 1..1024 targets and IGNORE_EOS=1");
                check(steps>0,"decode count must be positive");
                auto prompt=tokens(argv[at],129280);
                check(prompt.size()+static_cast<size_t>(steps)<=2048,"generation exceeds declared context");
                const auto out=std::filesystem::absolute(argv[at+3]).lexically_normal();
                check(destinations.insert(out).second && !std::filesystem::exists(out),"duplicate or existing request evidence directory");
                requests.push_back({std::move(prompt),std::move(targets),steps,ignore,out});
            }
        }
        check(strata::ds4::gguf::iq_q8k_activations(),"layer campaign requires explicit DS4_IQ_Q8K=1");
        const std::filesystem::path suffix_out=suffix ? argv[8]:"";
        if(suffix) check(std::filesystem::create_directory(suffix_out),"suffix evidence output directory already exists");
        o.pack=argv[2];
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
        check(model.vocab()==129280,"native layer campaign requires the pinned DeepSeek vocabulary");
        if(suffix) {run_suffix(model,cut,argv[4],timeout_ms,accept_ms,connections,planned,suffix_out);return 0;}
        auto c=contract(model,cut,planned);const auto binary=digest(read("/proc/self/exe"));receipt(role.c_str(),c,binary);
        if(prefix) {
            auto first=c.identity;first.session=1;
            check(model.configure_boundary_probe(cut,first,[](const auto &,std::string &){return true;},error),error);
            c.layout=model.boundary_layout();model.reset();
        }
        uint64_t session_floor=1;
        for(const auto &request:requests) {
            check(std::filesystem::create_directory(request.out),"request evidence output directory already exists");
            model.reset();
            if(prefix) {
                DeepSeekLayerClient client;check(client.connect(argv[4],c,unhex(argv[5]),timeout_ms,error),error);
                run_generation(model,c,request.prompt,request.targets,request.steps,request.ignore,request.out,&client,cut,session_floor);
            } else run_generation(model,c,request.prompt,request.targets,request.steps,request.ignore,request.out,nullptr,cut,session_floor);
        }
        return 0;
    } catch(const std::exception &e) {std::fprintf(stderr,"E3 TCP native: %s\n",e.what());return 1;}
}
