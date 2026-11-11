// K5 contiguous native K2 layer sequence with owner-local compressed MLA state.
#include "strata/kimi2/engine.hpp"
#ifdef STRATA_NETWORK_EXPERTS
#include "strata/core/network_experts.hpp"
#endif
#include <memory>
#include <cstdio>
#include <fstream>
#include <stdexcept>
#include <string>
#include <vector>
int main(int argc,char** argv){
 try{
  if(argc!=4 && argc!=5)throw std::runtime_error("usage: strata-kimi2-sequence FIXTURE OUTPUT_PREFIX prefill|incremental");
  const std::string root=argv[1],prefix=argv[2],mode=argv[3];
  if(mode!="prefill" && mode!="incremental")throw std::runtime_error("invalid sequence mode");
  std::vector<float> input(4*7168);std::ifstream f(root+"/embeddings.f32",std::ios::binary);
  if(!f.read(reinterpret_cast<char*>(input.data()),input.size()*4))throw std::runtime_error("embedding read failed");
  strata::core::NetworkExpertExecutor* remote=nullptr;
#ifdef STRATA_NETWORK_EXPERTS
  std::unique_ptr<strata::core::NetworkExpertExecutor> peer;
  if(argc==5){
   std::string identity;std::ifstream id(root+"/fixture.identity");id>>identity;
   if(identity.size()!=64)throw std::runtime_error("missing fixture identity");
   std::array<uint8_t,32> sha{};for(int i=0;i<32;++i)sha[i]=std::stoul(identity.substr(i*2,2),nullptr,16);
   peer=std::make_unique<strata::core::NetworkExpertExecutor>();std::string error;
   if(!peer->connect_and_handshake(argv[4],61,61,384,7168,2048,8,sha,30000,error) ||
      !peer->install_plan({{1,0},{2,2},{3,2}},error))throw std::runtime_error(error);
   remote=peer.get();
  }
#else
  if(argc==5)throw std::runtime_error("network support not built");
#endif
  strata::kimi2::MLAState states[4];std::vector<float> outputs[4];
  for(int begin=0;begin<4;){
   const int end=mode=="prefill"?4:(begin==0?2:begin+1);
   std::vector<float> x(input.begin()+begin*7168,input.begin()+end*7168);
   for(int layer=0;layer<4;++layer){
    x=strata::kimi2::run_layer(layer,root+"/layer"+std::to_string(layer),root+"/experts"+std::to_string(layer),x,
      prefix+".position"+std::to_string(begin)+".layer"+std::to_string(layer),states[layer],remote);
    outputs[layer].insert(outputs[layer].end(),x.begin(),x.end());
   }
   begin=end;
  }
  for(int l=0;l<4;++l){
   std::ofstream out(prefix+".layer"+std::to_string(l)+".output.f32",std::ios::binary);
   out.write(reinterpret_cast<const char*>(outputs[l].data()),outputs[l].size()*4);
   if(!out)throw std::runtime_error("output write failed");
   for(int part=0;part<2;++part){
    const auto& values=part?states[l].rotary:states[l].latent;
    std::ofstream cache(prefix+".layer"+std::to_string(l)+(part?".cache_rope.f32":".cache_latent.f32"),std::ios::binary);
    cache.write(reinterpret_cast<const char*>(values.data()),values.size()*4);
    if(!cache)throw std::runtime_error("cache capture write failed");
   }
   if(states[l].latent.size()!=4*512 || states[l].rotary.size()!=4*64)throw std::runtime_error("incorrect final MLA position");
  }
#ifdef STRATA_NETWORK_EXPERTS
  if(peer){peer->print_summary();if(peer->stats().fallbacks)throw std::runtime_error("remote fallback occurred");peer->close();}
#endif
  std::puts("Four native K2 layers completed on four real embedding rows with compressed MLA state");return 0;
 }catch(const std::exception& e){std::fprintf(stderr,"kimi2 sequence: %s\n",e.what());return 1;}
}
