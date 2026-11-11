// K2 partial-model adapter for Strata's existing whole-layer session/TCP fabric.
// The legacy result.logits container holds explicitly declared hidden-width rows here.
#include "strata/kimi2/engine.hpp"
#include "strata/kimi2/model.hpp"
#include "strata/core/deepseek_layer_rpc.hpp"
#include "strata/core/network_experts.hpp"
#include <openssl/evp.h>
#include <unistd.h>
#include <sys/socket.h>
#include <array>
#include <cstdio>
#include <fstream>
#include <memory>
#include <stdexcept>
namespace C=strata::core;
namespace K=strata::kimi2;
using Hash=std::array<uint8_t,32>;
Hash identity(const std::string& path){
 std::string hex;std::ifstream f(path);f>>hex;if(hex.size()!=64)throw std::runtime_error("missing identity: "+path);
 Hash out{};for(int i=0;i<32;++i)out[i]=std::stoul(hex.substr(2*i,2),nullptr,16);return out;
}
Hash executable_hash(){
 std::ifstream f("/proc/self/exe",std::ios::binary);if(!f)throw std::runtime_error("executable identity read failed");
 std::unique_ptr<EVP_MD_CTX,decltype(&EVP_MD_CTX_free)> ctx(EVP_MD_CTX_new(),EVP_MD_CTX_free);
 if(!ctx || EVP_DigestInit_ex(ctx.get(),EVP_sha256(),nullptr)!=1)throw std::runtime_error("SHA initialization failed");
 std::array<char,65536> bytes;
 while(f){f.read(bytes.data(),bytes.size());if(EVP_DigestUpdate(ctx.get(),bytes.data(),f.gcount())!=1)throw std::runtime_error("SHA update failed");}
 if(!f.eof())throw std::runtime_error("executable read failed");
 Hash sha{};unsigned n=0;if(EVP_DigestFinal_ex(ctx.get(),sha.data(),&n)!=1 || n!=32)throw std::runtime_error("SHA final failed");return sha;
}
void save(const std::string& path,const std::vector<float>& data){
 std::ofstream f(path,std::ios::binary);f.write(reinterpret_cast<const char*>(data.data()),data.size()*4);
 if(!f)throw std::runtime_error("capture write failed: "+path);
}
C::DeepSeekLayerContract contract(const std::string& root,bool full){
 C::DeepSeekLayerContract c;c.identity.model=identity(root+"/model.identity");c.identity.pack=identity(root+"/fixture.identity");
 c.identity.plan=identity(root+"/split-plan.identity");c.identity.epoch=1;c.identity.next_layer=2;
 c.layout.hidden=7168;c.layout.streams=1;c.layout.topk_stride=1;c.layout.cand_stride=1;
 c.layout.vocab=163840;c.layout.max_context=2048;c.layout.result_width=full?163840:7168;c.effective_experts=8;c.arithmetic=3;return c;
}
std::unique_ptr<C::NetworkExpertExecutor> peer(const std::string& endpoint,const std::string& root,bool suffix){
 auto p=std::make_unique<C::NetworkExpertExecutor>();std::string error;
 const std::vector<C::ExpertWorkerPair> pairs=suffix?std::vector<C::ExpertWorkerPair>{{2,2},{3,2}}:std::vector<C::ExpertWorkerPair>{{1,0}};
 if(!p->connect_and_handshake(endpoint,61,61,384,7168,2048,8,identity(root+"/fixture.identity"),120000,error) || !p->install_plan(pairs,error))throw std::runtime_error(error);
 p->set_keepalive_interval(30000);
 return p;
}
std::vector<float> layer(int index,const std::string& root,const std::vector<float>& x,const std::string& prefix,K::MLAState& state,C::NetworkExpertExecutor* p,const K::Model* model){
 if(p && !p->healthy())throw std::runtime_error("K2 expert connection is unhealthy; no implicit fallback");
 if(model)return K::run_layer(index,*model,x,prefix+".layer"+std::to_string(index),state,p);
 return K::run_layer(index,root+"/layer"+std::to_string(index),root+"/experts"+std::to_string(index),x,prefix+".layer"+std::to_string(index),state,p);
}
int main(int argc,char** argv){
 try{
  if(argc==6 && std::string(argv[1])=="moe-replay"){
   K::Model model(argv[2]);const int layer_index=std::stoi(argv[3]);
   std::ifstream input(argv[4],std::ios::binary|std::ios::ate);
   const auto bytes=input.tellg();
   if(!input || bytes<=0 || bytes%(7168*4)!=0 || bytes>8*7168*4)throw std::runtime_error("invalid captured K2 MoE input");
   std::vector<float> x(static_cast<size_t>(bytes)/4);input.seekg(0);
   if(!input.read(reinterpret_cast<char*>(x.data()),bytes))throw std::runtime_error("captured K2 MoE input read failed");
   K::run_moe("",x,argv[5],nullptr,layer_index,&model);return 0;
  }
  if(argc!=6 && argc!=8)throw std::runtime_error("usage: strata-kimi2-split worker|coordinator FIXTURE LAYER_ENDPOINT EXPERT_ENDPOINT CAPTURE_PREFIX [--model FIRST_SHARD]");
  const std::string mode=argv[1],root=argv[2],layer_endpoint=argv[3],expert_endpoint=argv[4],prefix=argv[5];
  const bool suffix=mode=="worker";if(!suffix && mode!="coordinator")throw std::runtime_error("unknown split mode");
  const bool full=argc==8;
  if(full && std::string(argv[6])!="--model")throw std::runtime_error("unknown full-model option");
  std::unique_ptr<K::Model> model;if(full)model=std::make_unique<K::Model>(argv[7]);
  const int layers=full?61:4,timeout=full?3600000:120000;
  auto remote=peer(expert_endpoint,root,suffix);const auto c=contract(root,full);
  std::vector<K::MLAState> states(layers);std::vector<std::vector<float>> outputs(layers);std::string error;
  if(suffix){
   C::DeepSeekLayerCapabilities caps;caps.contract=c;caps.binary_sha256=executable_hash();
#ifdef STRATA_USE_HIP
   caps.backend=2;
#else
   caps.backend=1;
#endif
   caps.incarnation=static_cast<uint64_t>(getpid());
   C::DeepSeekLayerService service(caps,[&](const C::DeepSeekBoundaryHeader&,std::string&){
     for(auto& s:states)s=K::MLAState{};return true;
    },[&](const C::DeepSeekLayerBoundary& b,uint32_t rows,C::DeepSeekLayerResult& result,std::string& err){
     try{
      if((full?rows!=1:rows!=b.header.rows) || states[2].latent.size()/512!=b.header.position || states[layers-1].latent.size()/512!=b.header.position)throw std::runtime_error("K2 suffix MLA position mismatch");
      auto x=b.h;const auto capture=prefix+".position"+std::to_string(b.header.position);
      for(int l=2;l<layers;++l){x=layer(l,root,x,capture,states[l],remote.get(),model.get());outputs[l].insert(outputs[l].end(),x.begin(),x.end());}
      result.request=b.header;result.committed_position=states[layers-1].latent.size()/512;result.logits_rows=rows;
      if(full){std::vector<float> last(x.end()-7168,x.end());result.logits=K::logits(*model,last);save(capture+".logits.f32",result.logits);}
      else result.logits=std::move(x);
      std::printf("K2 suffix executed layers2-%d position=%llu rows=%u committed=%llu result_width=%u\n",layers-1,(unsigned long long)b.header.position,b.header.rows,(unsigned long long)result.committed_position,c.layout.result_width);std::fflush(stdout);return true;
     }catch(const std::exception& e){err=e.what();return false;}
    });
   std::string host;uint16_t port=0;if(!C::expert_worker_parse_host_port(layer_endpoint,host,port))throw std::runtime_error("bad layer endpoint");
   uint16_t actual=0;int listen=C::expert_worker_listen(host,port,1,&actual,&error);if(listen<0)throw std::runtime_error(error);
   std::printf("K2_SUFFIX_LISTENING %s result_width=%u layers=2-%d epoch=1\n",layer_endpoint.c_str(),c.layout.result_width,layers-1);std::fflush(stdout);
   int fd=accept(listen,nullptr,nullptr);::close(listen);if(fd<0)throw std::runtime_error("layer accept failed");
   const bool ok=service.serve(fd,timeout,error);::close(fd);if(!ok)throw std::runtime_error(error);
   for(int l=2;l<layers;++l)save(prefix+".layer"+std::to_string(l)+".output.f32",outputs[l]);
  }else{
   C::DeepSeekLayerClient client;if(!client.connect(layer_endpoint,c,identity(root+"/suffix-binary.identity"),timeout,error) || !client.begin(1,error))throw std::runtime_error(error);
   std::vector<int32_t> ids;int32_t token_id;std::ifstream tokens(root+"/token-ids.txt");while(tokens>>token_id)ids.push_back(token_id);
   if((full && ids.empty()) || (!full && ids.size()!=4))throw std::runtime_error("incorrect prompt token IDs");
   std::vector<float> input;
   if(!full){input.resize(4*7168);std::ifstream f(root+"/embeddings.f32",std::ios::binary);if(!f.read(reinterpret_cast<char*>(input.data()),input.size()*4))throw std::runtime_error("embedding read failed");}
   uint64_t sequence=1,position=0;
   auto execute=[&](const std::vector<int32_t>& chunk,std::vector<float> x){
    const int rows=chunk.size();const auto capture=prefix+".position"+std::to_string(position);
    for(int l=0;l<2;++l){x=layer(l,root,x,capture,states[l],remote.get(),model.get());outputs[l].insert(outputs[l].end(),x.begin(),x.end());}
    C::DeepSeekLayerBoundary boundary;boundary.header=c.identity;boundary.header.session=1;boundary.header.sequence=sequence;boundary.header.position=position;boundary.header.rows=rows;
    boundary.h=std::move(x);boundary.tokens=chunk;boundary.pre.resize(rows*8);boundary.topk.resize(rows);boundary.cand.resize(rows);boundary.topk_valid.resize(rows);boundary.cand_valid.resize(rows);
    C::DeepSeekLayerResult result;C::DeepSeekLayerTiming timing;
    if(!client.execute(boundary,full?1:rows,result,error,&timing))throw std::runtime_error(error);
    if(!full)outputs[3].insert(outputs[3].end(),result.logits.begin(),result.logits.end());
    else save(capture+".logits.f32",result.logits);
    std::printf("K2 split committed position=%llu rows=%d layer_rpc_us=%llu tx_bytes=%llu rx_bytes=%llu\n",(unsigned long long)result.committed_position,rows,(unsigned long long)timing.total_us,(unsigned long long)timing.tx_bytes,(unsigned long long)timing.rx_bytes);std::fflush(stdout);
    position+=rows;++sequence;return result.logits;
   };
   std::vector<float> last;
   for(size_t begin=0;begin<ids.size();){
    const size_t end=full?std::min(begin+8,ids.size()):(begin==0?2:begin+1);
    std::vector<int32_t> chunk(ids.begin()+begin,ids.begin()+end);
    auto x=full?model->embeddings(chunk):std::vector<float>(input.begin()+begin*7168,input.begin()+end*7168);
    last=execute(chunk,std::move(x));begin=end;
   }
   if(full){
    std::ofstream generated(prefix+".generated-ids.txt");
    int limit=16;std::ifstream limit_file(root+"/max-new-tokens.txt");
    if(limit_file && (!(limit_file>>limit) || limit<1 || limit>16))throw std::runtime_error("invalid decode token limit");
    int count=0;
    for(;count<limit;++count){
     const int32_t next=std::max_element(last.begin(),last.end())-last.begin();generated<<next<<' ';generated.flush();
     std::printf("K2 GREEDY_TOKEN index=%d id=%d actual_position=%llu\n",count,next,(unsigned long long)position);std::fflush(stdout);
     if(next==model->eos_token_id()){std::puts("K2 native EOS");break;}
     last=execute({next},model->embeddings({next}));
    }
    if(!generated)throw std::runtime_error("token capture write failed");
    std::printf("K2 full model decode executed=%d maximum_position=%llu\n",count,(unsigned long long)position);
   }
   if(!client.close(error))throw std::runtime_error(error);
   for(int l:{0,1})save(prefix+".layer"+std::to_string(l)+".output.f32",outputs[l]);
   if(!full)save(prefix+".layer3.output.f32",outputs[3]);
  }
  remote->print_summary();if(remote->stats().fallbacks)throw std::runtime_error("remote expert fallback occurred");remote->close();
  std::puts(full?"K2 full split completed: native layers0-60, final vocabulary logits and greedy decode":"K2 partial split completed: native layers0-3, real token embeddings; no logits/generation claim");return 0;
 }catch(const std::exception& e){std::fprintf(stderr,"K2 split: %s\n",e.what());return 1;}
}
