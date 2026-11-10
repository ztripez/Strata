// Synthetic process/metadata fixture. No model parsing, GPU or inference.
#include <openssl/sha.h>
#include <fstream>
#include <iomanip>
#include <iostream>
#include <iterator>
#include <map>
#include <sstream>
#include <string>
#include <thread>
#include <chrono>
#include <cstdlib>

std::string env(const char *key) {const char *v=std::getenv(key);return v ? v:"";}
int main(int argc,char **argv) {
    std::map<std::string,std::string> a;
    for(int i=1;i<argc;++i) {
        const std::string key=argv[i];
        if(key=="--check-config") {std::cout<<"CONFIG_CHECKED syntax_only; model/device/quality NOT VERIFIED\n";return 0;}
        if(key=="--serve" || key=="--verbose") continue;
        if(i+1<argc) a[key]=argv[++i];
    }
    const auto mode=env("STRATA_FIXTURE_MODE");
    if(mode=="exit") return 42;
    std::cerr<<"SYNTHETIC CONTROL ONLY; no model or GPU execution\n";
    if(mode!="hold") {
        if(a.contains("--model")) {
            std::cerr<<"ds4-network-identity: role=expert-worker effective_experts=6 engine=synthetic arithmetic=2 model_sha256="<<env("STRATA_FIXTURE_EXPERT_SHA")<<'\n';
            std::cout<<"WORKER_LISTENING "<<a["--bind"]<<'\n';
            if(mode!="listen-only") {
                const auto count=env("STRATA_FIXTURE_OWNED");
                std::cerr<<"strata-worker-tiers: owned="<<count<<" gpu=0 ram=0 mapped="<<(mode=="bad-coverage" ? "0":count)<<" gpu_bytes=0 ram_bytes=0\n";
                std::cerr<<"strata-expert-worker: READY owned "<<count<<" experts (0 bytes VRAM); synthetic\n";
            }
        } else {
            std::ifstream image("/proc/self/exe",std::ios::binary);
            const std::string bytes((std::istreambuf_iterator<char>(image)),{});
            unsigned char hash[SHA256_DIGEST_LENGTH];SHA256(reinterpret_cast<const unsigned char*>(bytes.data()),bytes.size(),hash);
            std::ostringstream hex;for(auto b:hash) hex<<std::hex<<std::setfill('0')<<std::setw(2)<<static_cast<unsigned>(b);
            if(mode!="bad-artifacts") std::cerr<<env("STRATA_FIXTURE_ARTIFACTS");
            const bool prefix=a["--role"]=="prefix";
            std::cout<<"INFO arch=deepseek41 fabric=1 role="<<(prefix ? "prefix":"suffix")<<" context=2048 effective_experts=6 arithmetic=2 model_sha256="<<a["--model-sha256"]
                <<" pack_sha256="<<a["--pack-sha256"]<<" plan_sha256="<<a["--plan-id"]<<" binary_sha256="<<hex.str()
                <<" plan_epoch="<<(mode=="wrong-epoch" ? "1":a["--plan-epoch"])
                <<" cpu_threads_limit="<<a["--cpu-threads"]<<" expert_vram_mib="<<a["--expert-vram-mib"]
                <<" request_timeout_ms="<<(prefix ? a["--request-timeout-ms"]:"0")<<" rpc_timeout_ms="<<a["--rpc-timeout-ms"]<<" idle_timeout_ms="<<(prefix ? "0":a["--idle-timeout-ms"])<<'\n';
            if(prefix) {
                std::cout<<"READY 2048 stop\n"<<std::flush;
                std::string line;
                while(std::getline(std::cin,line)) {
                    if(line=="QUIT") return 0;
                    if(!line.starts_with("GEN ")) continue;
                    std::istringstream command(line);std::string op;int maximum;command>>op>>maximum;
                    const auto last=line.substr(line.find_last_of(' ')+1);
                    int prompt=1;for(char ch:last) if(ch==',') ++prompt;
                    std::istringstream tokens(env("STRATA_FIXTURE_OUTPUT"));int id,count=0;
                    while(count<maximum && tokens>>id) {std::cout<<"T "<<id<<'\n';++count;}
                    std::cout<<"DONE "<<count<<' '<<prompt<<" 1.0 2.0 stop 0 0 0 0 0\n"<<std::flush;
                }
                return 0;
            }
            std::cout<<"LAYER_READY endpoint="<<a["--endpoint"]<<" incarnation=1234\n";
        }
    }
    std::cout.flush();std::cerr.flush();
    for(;;) std::this_thread::sleep_for(std::chrono::seconds(1));
}
