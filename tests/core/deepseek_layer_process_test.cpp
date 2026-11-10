// CPU-only native process admission/input tests. No model or GPU work.
#include "ds4_layer_process.hpp"
#include "ds4_layer_input.hpp"
#include "ds4_layer_trace.hpp"
#include <chrono>
#include <cstdio>

namespace N=strata::ds4::network;
namespace {
using Clock=std::chrono::steady_clock;
std::vector<std::string> arguments() {
    return {"--serve","--role","prefix","--pack","/immutable/pack","--cut","20","--endpoint","127.0.0.1:19000",
        "--model-sha256",std::string(64,'a'),"--pack-sha256",std::string(64,'b'),"--plan-id",std::string(64,'c'),
        "--peer-binary-sha256",std::string(64,'d'),"--experts","6","--expert-vram-mib","384","--cpu-threads","8","--rpc-timeout-ms","30000","--request-timeout-ms","1800000"};
}
template<class F> void refused(F action) {
    bool caught=false;try {action();} catch(const std::runtime_error &) {caught=true;}
    N::require(caught,"invalid process/request input accepted");
}
void configuration() {
    const auto original=arguments();const auto config=N::configuration(original);
    N::require(config.role=="prefix" && config.cpu_threads==8 && config.expert_vram_mib==384 && config.cut==20,"valid configuration changed");
    N::require(!config.trace_windows,"window tracing changed the default profile");
    auto traced=original;traced.push_back("--trace-windows");
    N::require(N::configuration(traced).trace_windows,"opt-in window tracing refused");
    traced.push_back("--trace-windows");refused([&]{N::configuration(traced);});
    for(const auto &flag:{"--role","--pack","--cut","--endpoint","--model-sha256","--pack-sha256","--plan-id","--peer-binary-sha256","--experts","--expert-vram-mib","--cpu-threads","--rpc-timeout-ms","--request-timeout-ms"}) {
        auto missing=original;const auto at=std::find(missing.begin(),missing.end(),flag);missing.erase(at,at+2);refused([&]{N::configuration(missing);});
    }
    for(const auto &[flag,value]:std::vector<std::pair<std::string,std::string>>{
        {"--role","other"},{"--cut","21"},{"--endpoint","astor:1"},{"--endpoint","127.0.0.1:0"},
        {"--model-sha256",std::string(64,'0')},{"--pack-sha256",std::string(64,'X')},{"--experts","5"},
        {"--cpu-threads","0"},{"--cpu-threads","65"},{"--expert-vram-mib","-1"},{"--expert-vram-mib","65537"},
        {"--rpc-timeout-ms","0"},{"--rpc-timeout-ms","600001"},{"--request-timeout-ms","29999"},{"--request-timeout-ms","7200001"}}) {
        auto changed=original;*std::next(std::find(changed.begin(),changed.end(),flag))=value;refused([&]{N::configuration(changed);});
    }
    for(const auto &extra:std::vector<std::vector<std::string>>{{"--serve"},{"--experts","6"},{"--unknown","1"},{"--idle-timeout-ms","1"},{"--expert-worker","127.0.0.1:2"}}) {
        auto changed=original;changed.insert(changed.end(),extra.begin(),extra.end());refused([&]{N::configuration(changed);});
    }
    auto suffix=original;*std::next(std::find(suffix.begin(),suffix.end(),"--role"))="suffix";
    auto peer=std::find(suffix.begin(),suffix.end(),"--peer-binary-sha256");suffix.erase(peer,peer+2);suffix.insert(suffix.end(),{"--idle-timeout-ms","7200000"});
    auto timeout=std::find(suffix.begin(),suffix.end(),"--request-timeout-ms");suffix.erase(timeout,timeout+2);
    N::require(N::configuration(suffix).role=="suffix","valid suffix refused");
    suffix.push_back("--check-config");N::require(N::configuration(suffix).check_only,"syntax-check mode missing");
}
void trace_output() {
    const auto c=N::configuration(arguments());
    strata::core::DeepSeekBoundaryHeader h;
    h.model=c.model;h.pack=c.pack_identity;h.plan=c.plan;h.epoch=1;h.session=2;h.sequence=1;h.rows=1;h.next_layer=20;
    struct File {FILE *value;~File(){if(value) std::fclose(value);}};
    File file{std::tmpfile()};N::require(file.value!=nullptr,"temporary trace file failed");
    N::PrefixWindowTiming t;t.feed_us=20;t.before_rpc_us=5;t.rpc_us=12;t.after_rpc_us=3;
    t.transaction={11,7,2,123,456};
    N::prefix_window_trace(file.value,c,c.peer_binary,h,1234,t);N::suffix_window_trace(file.value,c,c.peer_binary,h,1234,6);
    std::rewind(file.value);std::string text;char buffer[1024];
    while(const auto count=std::fread(buffer,1,sizeof(buffer),file.value)) text.append(buffer,count);
    N::require(text.find("\"role\":\"prefix\"")!=std::string::npos && text.find("\"role\":\"suffix\"")!=std::string::npos &&
               text.find("\"tx_bytes\":123")!=std::string::npos && text.find("\"native_us\":6")!=std::string::npos,
               "trace serialization omitted spans or actual wire bytes");
    File full{std::fopen("/dev/full","w")};N::require(full.value!=nullptr,"Linux full-device test unavailable");
    refused([&]{N::prefix_window_trace(full.value,c,c.peer_binary,h,1234,t);});
}
void requests() {
    const auto r=N::request("GEN 128 seed=1 ckpt=0 temperature=0 top_k=64 top_p=1 0,1,129279");
    N::require(r.maximum==128 && r.tokens==std::vector<int>({0,1,129279}),"valid greedy request changed");
    for(const auto &text:{"GEN 0 1","GEN 1025 1","GEN 1","GEN 1 1,","GEN 1 ,1","GEN 1 -1","GEN 1 129280","GEN 1 1.0","GEN 1 1 2",
        "GENI 1 1","GEN 1 temperature=0.1 1","GEN 1 top_p=0.9 1","GEN 1 logprobs=0 1","GEN 1 pin=1 1","GEN 1 marks=1 1",
        "GEN 1 unknown=0 1","GEN 1 temperature=nan 1","GEN 1 seed=-1 1","GEN 1 ckpt=0 ckpt=0 1","GEN 1 1 ckpt=0"}) refused([&]{N::request(text);});
    std::string long_prompt;
    for(int i=0;i<1900;++i) {if(i) long_prompt+=',';long_prompt+='4';}
    N::require(N::request("GEN 148 "+long_prompt).tokens.size()==1900,"full context refused");
    refused([&]{N::request("GEN 149 "+long_prompt);});
    refused([&]{N::request("GEN 1 "+long_prompt+",4");});
}
struct Pipe {
    int fd[2]{-1,-1};
    Pipe() {N::require(::pipe(fd)==0,"pipe failed");}
    ~Pipe() {::close(fd[0]);::close(fd[1]);}
    void send(const std::string &text) {
        size_t sent=0;while(sent<text.size()) {const auto n=::write(fd[1],text.data()+sent,text.size()-sent);if(n<0 && errno==EINTR) continue;N::require(n>0,"pipe write failed");sent+=static_cast<size_t>(n);}
    }
};
template<class F> void until(F predicate) {
    const auto end=Clock::now()+std::chrono::seconds(1);
    while(!predicate() && Clock::now()<end) std::this_thread::sleep_for(std::chrono::milliseconds(1));
    N::require(predicate(),"input event deadline exceeded");
}
void input() {
    Pipe pipe;N::CommandInput input(pipe.fd[0]);std::atomic<int> old{0},fresh{0};std::string line;
    pipe.send("GEN 1 4\n");N::require(input.next(line) && line=="GEN 1 4","first command missing");
    {N::InputBinding binding(input,[&]{++old;});pipe.send("STOP\n");until([&]{return old.load()==1;});N::require(input.cancelled(),"STOP did not cancel");}
    input.finish();pipe.send("GEN 1 5\n");N::require(input.next(line) && !input.cancelled(),"cancel leaked into fresh request");
    {N::InputBinding binding(input,[&]{++fresh;});pipe.send("STOP\n");until([&]{return fresh.load()==1;});N::require(old==1,"expired callback invoked");}
    input.finish();pipe.send("QUIT\n");N::require(!input.next(line) && input.failure().empty(),"QUIT did not end input cleanly");
}
void input_faults() {
    for(int fault=0;fault<3;++fault) {
        Pipe pipe;N::CommandInput input(pipe.fd[0]);std::string line;
        if(fault==0) {pipe.send("GEN 1 4\n");N::require(input.next(line),"missing initial request");pipe.send("GEN 1 5\n");}
        if(fault==1) pipe.send(std::string(32769,'x'));
        if(fault==2) {pipe.send("GEN 1 4");::close(pipe.fd[1]);pipe.fd[1]=-1;}
        until([&]{return !input.failure().empty();});
        N::require(input.cancelled(),"input fault did not cancel active/pending request");input.finish();
    }
    Pipe pipe;const auto start=Clock::now();{N::CommandInput input(pipe.fd[0]);}
    N::require(Clock::now()-start<std::chrono::milliseconds(500),"reader destructor waited on open stdin");
    Pipe timed;N::CommandInput deadline(timed.fd[0],20);std::string line;std::atomic<int> interrupted{0};
    timed.send("GEN 1 4\n");N::require(deadline.next(line),"deadline request missing");
    {N::InputBinding binding(deadline,[&]{++interrupted;});until([&]{return interrupted.load()==1;});}
    N::require(deadline.cancelled() && deadline.failure()=="layer request deadline exceeded","request deadline did not interrupt blocked work");deadline.finish();
}
} // namespace
int main() {
    try {configuration();trace_output();requests();input();input_faults();std::puts("Layer process configuration/trace/greedy/input/cancellation tests PASS; no model/device execution");return 0;}
    catch(const std::exception &error) {std::fprintf(stderr,"%s\n",error.what());return 1;}
}
