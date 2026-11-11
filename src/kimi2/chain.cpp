// Native K2 layer0 -> layer1 bring-up on actual token embeddings.
#include "strata/kimi2/engine.hpp"
#include <cstdio>
#include <fstream>
#include <stdexcept>
#include <vector>
int main(int argc,char** argv){
 try{
  if(argc!=6 && argc!=7)throw std::runtime_error("usage: strata-kimi2-chain LAYER0 LAYER1 EXPERTS EMBEDDINGS_F32 OUTPUT_PREFIX");
  std::vector<float> input(3*7168);std::ifstream f(argv[4],std::ios::binary);
  f.read(reinterpret_cast<char*>(input.data()),input.size()*4);if(!f)throw std::runtime_error("embedding read failed");
  if(argc==7){
   if(std::string(argv[6])!="--stream")throw std::runtime_error("unknown chain option");
   strata::kimi2::MLAState states[2];
   std::vector<float> outputs[2];
   for(int t=0;t<3;++t){
    std::vector<float> row(input.begin()+t*7168,input.begin()+(t+1)*7168);
    for(int layer=0;layer<2;++layer){
     row=strata::kimi2::run_layer(layer,argv[1+layer],argv[3],row,
      std::string(argv[5])+".token"+std::to_string(t)+".layer"+std::to_string(layer),states[layer]);
     outputs[layer].insert(outputs[layer].end(),row.begin(),row.end());
    }
   }
   for(int layer=0;layer<2;++layer){
    std::ofstream out(std::string(argv[5])+".layer"+std::to_string(layer)+".output.f32",std::ios::binary);
    out.write(reinterpret_cast<const char*>(outputs[layer].data()),outputs[layer].size()*4);
    if(!out)throw std::runtime_error("stream output write failed");
   }
   std::puts("K2 incremental native chain completed: three positions, compressed per-layer KV");return 0;
  }
  const auto layer0=strata::kimi2::run_layer(0,argv[1],argv[3],input,std::string(argv[5])+".layer0");
  strata::kimi2::run_layer(1,argv[2],argv[3],layer0,std::string(argv[5])+".layer1");
  std::puts("Kimi K2 native layer0 -> layer1 completed: 3 embedding rows; full MLA/residual and tiered MoE");return 0;
 }catch(const std::exception& e){std::fprintf(stderr,"kimi2 chain: %s\n",e.what());return 1;}
}
