// K6 bring-up client using the existing expert executor and original routing assignments.
#include "strata/core/network_experts.hpp"
#include <fstream>
#include <stdexcept>
#include <cstdio>
#include <chrono>
#include <thread>
int main(int argc,char** argv){
 try{
  if(argc!=4 && argc!=5)throw std::runtime_error("usage: strata-kimi2-expert-probe ENDPOINT FIXTURE OUTPUT");
  const std::string dir=argv[2];std::string identity;std::ifstream id(dir+"/fixture.identity");id>>identity;
  if(identity.size()!=64)throw std::runtime_error("missing fixture identity");
  std::array<uint8_t,32> sha{};for(int i=0;i<32;++i)sha[i]=std::stoul(identity.substr(i*2,2),nullptr,16);
  std::vector<float> input(3*7168),sums(input.size());std::vector<uint8_t> mask(3);
  std::ifstream f(dir+"/probe-input.f32",std::ios::binary);
  if(!f.read(reinterpret_cast<char*>(input.data()),input.size()*4))throw std::runtime_error("input read failed");
  int token,slot,expert;float weight;std::ifstream route(dir+"/probe-assignment.txt");
  if(!(route>>token>>slot>>expert>>weight))throw std::runtime_error("routing assignment missing");
  strata::core::NetworkExpertExecutor net;std::string error;
  if(!net.connect_and_handshake(argv[1],61,61,384,7168,2048,8,sha,30000,error) || !net.install_plan({{1,expert}},error))throw std::runtime_error(error);
  strata::core::ExpertWorkerAssignment assignment{static_cast<uint16_t>(token),static_cast<uint16_t>(slot),expert,weight};
  if(argc==5){
   if(std::string(argv[4])=="--idle-check"){
    net.set_keepalive_interval(1000);
    std::puts("PROBE_IDLE_BEGIN seconds=12 keepalive_ms=1000");std::fflush(stdout);
    std::this_thread::sleep_for(std::chrono::seconds{12});
    if(!net.healthy())throw std::runtime_error("worker unhealthy after idle keepalive");
   }else{
   if(std::string(argv[4])!="--expect-disconnect")throw std::runtime_error("unknown probe mode");
   std::puts("PROBE_PLAN_READY: terminate this probe worker, then send newline");std::fflush(stdout);
   if(std::getchar()==EOF)throw std::runtime_error("disconnect control input closed");
   const auto begin=std::chrono::steady_clock::now();
   const bool ok=net.execute_sync(1,3,input.data(),{assignment},sums.data(),mask.data(),&error);
   const auto elapsed=std::chrono::duration_cast<std::chrono::milliseconds>(std::chrono::steady_clock::now()-begin).count();
   if(ok || net.healthy() || elapsed>30000)throw std::runtime_error("disconnect did not fail within the declared deadline");
   std::printf("EXPECTED_DISCONNECT_FAILURE elapsed_ms=%lld fallbacks=%llu healthy=%d error=%s\n",(long long)elapsed,(unsigned long long)net.stats().fallbacks,net.healthy(),error.c_str());net.close();return 0;
   }
  }
  if(!net.execute_sync(1,3,input.data(),{assignment},sums.data(),mask.data(),&error))throw std::runtime_error(error);
  if(net.stats().fallbacks || net.stats().assignments_executed!=1 || mask[token]!=1)throw std::runtime_error("unexpected assignment/fallback receipt");
  std::ofstream out(argv[3],std::ios::binary);out.write(reinterpret_cast<const char*>(sums.data()),sums.size()*4);
  if(!out)throw std::runtime_error("output write failed");net.print_summary();net.close();return 0;
 }catch(const std::exception& e){std::fprintf(stderr,"K2 expert probe: %s\n",e.what());return 1;}
}
