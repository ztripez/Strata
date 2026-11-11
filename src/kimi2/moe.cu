// K2 model glue: existing native routing, ExpertCache, GPU kernels and CPU ExpertPool.
#include "ds4_kernels.cuh"
#include "strata/kimi2/engine.hpp"
#include "strata/kimi2/model.hpp"
#ifdef STRATA_NETWORK_EXPERTS
#include "strata/core/network_experts.hpp"
#endif
#include <algorithm>
#include "ds4_iq.cuh"
#include "strata/kernels/native_router.hpp"
#include "strata/core/expert_cache.hpp"
#include "strata/kernels/cpu/pool.hpp"
#include <cuda_runtime.h>
#include <sys/mman.h>
#include <fcntl.h>
#include <unistd.h>
#include <fstream>
#include <cstdio>
#include <cmath>
#include <cstring>
#include <memory>
#include <set>
#include <stdexcept>
#include <string>
#include <vector>
namespace C=strata::kernels::cpu;
namespace {
constexpr int H=7168, FF=2048, E=384, K=8;
void check(cudaError_t e){if(e!=cudaSuccess)throw std::runtime_error(cudaGetErrorString(e));}
struct Device {
 float* p=nullptr;size_t n;
 explicit Device(size_t count):n(count){check(cudaMalloc(&p,n*4));}
 ~Device(){if(p)cudaFree(p);}
 Device(const Device&)=delete;Device& operator=(const Device&)=delete;
 void put(const float* src){check(cudaMemcpy(p,src,n*4,cudaMemcpyHostToDevice));}
 void get(float* dst)const{check(cudaMemcpy(dst,p,n*4,cudaMemcpyDeviceToHost));}
};
struct Mapping {
 int fd=-1;const uint8_t* p=nullptr;size_t n;
 Mapping(const std::string& path,size_t bytes,const uint8_t* borrowed=nullptr):n(bytes){
  if(borrowed){p=borrowed;return;}
  fd=open(path.c_str(),O_RDONLY);if(fd<0)throw std::runtime_error("open failed: "+path);
  if(lseek(fd,0,SEEK_END)!=static_cast<off_t>(n))throw std::runtime_error("wrong weight size: "+path);
  void* a=mmap(nullptr,n,PROT_READ,MAP_PRIVATE,fd,0);if(a==MAP_FAILED)throw std::runtime_error("mmap failed");p=static_cast<const uint8_t*>(a);
 }
 ~Mapping(){if(p && fd>=0)munmap(const_cast<uint8_t*>(p),n);if(fd>=0)close(fd);}
};
std::vector<float> read(const std::string& path,size_t n){
 std::vector<float> data(n);std::ifstream f(path,std::ios::binary);f.read(reinterpret_cast<char*>(data.data()),n*4);
 if(!f)throw std::runtime_error("input read failed: "+path);return data;
}
void save(const std::string& path,const void* data,size_t n){
 std::ofstream f(path,std::ios::binary);f.write(static_cast<const char*>(data),n);if(!f)throw std::runtime_error("output write failed");
}
void dense(int type,const uint8_t* raw,int rows,int cols,int nt,const Device& x,Device& y){
 Device weights(static_cast<size_t>(rows)*cols);
 if(!strata::ds4::gguf::iq::dequant_rows_f32(type,raw,strata::ds4::gguf::iq::row_bytes(type,cols),cols,rows,weights.p,nullptr))throw std::runtime_error("GPU quant type refused");
 strata::ds4::gemv_f32(weights.p,x.p,rows,cols,nt,y.p,rows,nullptr);check(cudaGetLastError());check(cudaDeviceSynchronize());
}
struct Expert {
 Mapping gate,up,down;C::NativeFmt fmt;
 Expert(const std::string& dir,int expert,const C::NativeFmt& f,const strata::kimi2::Model* model=nullptr,int layer=1):
 gate(dir+"/ffn_gate_exps.weight.expert"+std::to_string(expert)+".bin",f.gu_row*FF,model?model->expert(layer,"gate",expert):nullptr),
 up(dir+"/ffn_up_exps.weight.expert"+std::to_string(expert)+".bin",f.gu_row*FF,model?model->expert(layer,"up",expert):nullptr),
 down(dir+"/ffn_down_exps.weight.expert"+std::to_string(expert)+".bin",f.d_row*H,model?model->expert(layer,"down",expert):nullptr),fmt(f){
  fmt.up_off=reinterpret_cast<uintptr_t>(up.p)-reinterpret_cast<uintptr_t>(gate.p);
  fmt.down_off=reinterpret_cast<uintptr_t>(down.p)-reinterpret_cast<uintptr_t>(gate.p);
 }
};
}
std::vector<float> strata::kimi2::run_cached_expert(const uint8_t* blob,int gu_type,int down_type,
    const std::vector<float>& input){
 const int nt=input.size()/H;
 if(nt<1 || nt>16 || input.size()!=static_cast<size_t>(nt*H))throw std::runtime_error("invalid cached K2 expert rows");
 C::NativeFmt format;std::string error;
 if(!C::native_fmt(gu_type,down_type,H,FF,format,error,true))throw std::runtime_error(error);
 Device x(input.size()),gate(nt*FF),up(nt*FF),hidden(nt*FF),out(input.size());x.put(input.data());
 dense(gu_type,blob,FF,H,nt,x,gate);dense(gu_type,blob+format.up_off,FF,H,nt,x,up);
 strata::ds4::swiglu(gate.p,up.p,nt*FF,0,hidden.p,nullptr);
 dense(down_type,blob+format.down_off,H,FF,nt,hidden,out);
 std::vector<float> result(input.size());out.get(result.data());return result;
}
std::vector<float> strata::kimi2::run_moe(const std::string& dir,const std::vector<float>& input,const std::string& out,core::NetworkExpertExecutor* remote,int layer,const Model* model){
  const int T=input.size()/H;
  if(T<1 || T>C::MAXT || input.size()!=static_cast<size_t>(T*H))throw std::runtime_error("K2 MoE window exceeds existing pool batch size");
  const std::string tensor_prefix="blk."+std::to_string(layer)+".";
  auto tensor_type=[&](const char* role,int original){
   if(model)return model->type(tensor_prefix+role+".weight");
   std::ifstream f(dir+"/"+role+".type");int type=original;
   if(f && !(f>>type))throw std::runtime_error("invalid tensor type metadata");return type;
  };
  const int gu_type=tensor_type("ffn_gate_exps",21),down_type=tensor_type("ffn_down_exps",23);
  const int shared_gu=tensor_type("ffn_gate_shexp",23),shared_down=tensor_type("ffn_down_shexp",14);
  C::NativeFmt format;std::string error;
  if(!C::native_fmt(gu_type,down_type,H,FF,format,error,true))throw std::runtime_error(error);
  if(layer<1 || layer>=61)throw std::runtime_error("invalid K2 MoE layer");
  bool remote_owned[E]={};
#ifdef STRATA_NETWORK_EXPERTS
  if(remote && remote->plan_holds_layer(layer)){
   if(!remote->healthy())throw std::runtime_error("K2 remote layer plan is unhealthy; no implicit fallback");
   for(int e=0;e<E;++e)remote_owned[e]=remote->holds(layer,e);
  }
#else
  if(remote)throw std::runtime_error("K2 network support was not built");
#endif
  auto remotely_owned=[&](int expert){return remote_owned[expert];};
  strata::core::ExpertCache cache;
  if(!cache.open(1,61,E,format.bytes,error))throw std::runtime_error(error);
  if(!remotely_owned(0)){
   Expert resident(dir,0,format,model,layer);const int slot=cache.admit(layer,0);
   strata::core::BlobPart parts[]={{resident.gate.p,resident.gate.n,0},{resident.up.p,resident.up.n,format.up_off},{resident.down.p,resident.down.n,format.down_off}};
   if(slot<0 || !cache.fill_slot_queued_parts(slot,parts,3,error) || !cache.sync_queued(error))throw std::runtime_error(error);
  }
  const auto& host_x=input;
  auto floats=[&](const std::string& tensor,size_t n,const std::string& fixture){
   if(!model)return read(fixture,n);
   if(model->type(tensor)!=0 || model->bytes(tensor)!=n*4)throw std::runtime_error("wrong mapped K2 F32 tensor: "+tensor);
   std::vector<float> values(n);std::memcpy(values.data(),model->data(tensor),n*4);return values;
  };
  auto gate_weights=floats(tensor_prefix+"ffn_gate_inp.weight",E*H,dir+"/gate.f32"),bias=floats(tensor_prefix+"exp_probs_b.bias",E,dir+"/bias.f32");
  Device x(T*H),gate(E*H),correction(E),logits(T*E),ids(T*K),weights(T*K);
  x.put(host_x.data());gate.put(gate_weights.data());correction.put(bias.data());
  cudaStream_t stream;check(cudaStreamCreate(&stream));
  strata::ds4::gemv_f32(gate.p,x.p,E,H,T,logits.p,E,stream);
  strata::kernels::native_router_kimi2(logits.p,correction.p,reinterpret_cast<int32_t*>(ids.p),weights.p,T,stream);
  check(cudaStreamSynchronize(stream));check(cudaStreamDestroy(stream));
  std::vector<int> selected(T*K);std::vector<float> routing(T*K),values(T*K*H),shared(T*H);
  check(cudaMemcpy(selected.data(),ids.p,T*K*4,cudaMemcpyDeviceToHost));weights.get(routing.data());
  std::vector<float> remote_sums(T*H);int remote_assignments=0;
#ifdef STRATA_NETWORK_EXPERTS
  std::vector<strata::core::ExpertWorkerAssignment> assignments;
  for(int i=0;i<T*K;++i)if(remotely_owned(selected[i]))assignments.push_back({static_cast<uint16_t>(i/K),static_cast<uint16_t>(i%K),selected[i],routing[i]});
  remote_assignments=assignments.size();
  if(remote_assignments && !remote->begin_async(layer,T,host_x.data(),assignments))throw std::runtime_error("K2 remote dispatch failed");
#endif
  save(out+".input.f32",host_x.data(),host_x.size()*4);
  std::set<int> unique(selected.begin(),selected.end());std::vector<std::unique_ptr<Expert>> mapped;
  std::vector<C::ExpertJobMulti> jobs;int gpu_assignments=0,cpu_assignments=0;
  for(int expert:unique){
   if(remotely_owned(expert))continue;
   std::vector<int> positions;
   for(int i=0;i<T*K;++i)if(selected[i]==expert)positions.push_back(i);
   if(cache.slot_of(layer,expert)>=0){
    std::vector<float> inputs(positions.size()*H),result(positions.size()*H);
    for(size_t i=0;i<positions.size();++i)std::copy_n(host_x.data()+(positions[i]/K)*H,H,inputs.data()+i*H);
    const uint8_t* blob=cache.device_slot(cache.slot_of(layer,expert));
    result=run_cached_expert(blob,gu_type,down_type,inputs);
    for(size_t i=0;i<positions.size();++i)std::copy_n(result.data()+i*H,H,values.data()+positions[i]*H);
    gpu_assignments+=(int)positions.size();
   }else{
    mapped.push_back(std::make_unique<Expert>(dir,expert,format,model,layer));auto& m=*mapped.back();
    C::ExpertJobMulti job;job.blob=m.gate.p;job.fmt=&m.fmt;job.nt=positions.size();
    for(size_t i=0;i<positions.size();++i){job.f32act[i]=host_x.data()+(positions[i]/K)*H;job.out[i]=values.data()+positions[i]*H;}
    jobs.push_back(job);cpu_assignments+=(int)positions.size();
   }
  }
  C::ExpertPool pool(2,false,false);
  if(model)pool.set_stall_timeout(std::chrono::seconds{600});
  pool.run_split_multi_native(format,jobs.data(),jobs.size(),C::NativeArithmetic::DequantizedF32);
  Device sg(T*FF),su(T*FF),sh(T*FF),sy(T*H);
  Mapping wg(dir+"/ffn_gate_shexp.weight.bin",static_cast<size_t>(FF)*strata::ds4::gguf::iq::row_bytes(shared_gu,H),model?model->data(tensor_prefix+"ffn_gate_shexp.weight"):nullptr);
  Mapping wu(dir+"/ffn_up_shexp.weight.bin",wg.n,model?model->data(tensor_prefix+"ffn_up_shexp.weight"):nullptr);
  Mapping wd(dir+"/ffn_down_shexp.weight.bin",static_cast<size_t>(H)*strata::ds4::gguf::iq::row_bytes(shared_down,FF),model?model->data(tensor_prefix+"ffn_down_shexp.weight"):nullptr);
  for(auto role:{0,1,2}){
   const Mapping& m=role==0?wg:role==1?wu:wd;Device raw((m.n+3)/4);check(cudaMemcpy(raw.p,m.p,m.n,cudaMemcpyHostToDevice));
   if(role==0)dense(shared_gu,reinterpret_cast<uint8_t*>(raw.p),FF,H,T,x,sg);
   if(role==1)dense(shared_gu,reinterpret_cast<uint8_t*>(raw.p),FF,H,T,x,su);
   if(role==2){strata::ds4::swiglu(sg.p,su.p,T*FF,0,sh.p,nullptr);dense(shared_down,reinterpret_cast<uint8_t*>(raw.p),H,FF,T,sh,sy);}
  }
#ifdef STRATA_NETWORK_EXPERTS
  if(remote_assignments){
   std::vector<uint8_t> mask(T);
   if(!remote->wait_result(remote_sums.data(),mask.data(),&error))throw std::runtime_error("K2 remote execution failed: "+error);
  }
#endif
  save(out+".remote_sums.f32",remote_sums.data(),remote_sums.size()*4);
  sy.get(shared.data());std::vector<float> reduced(T*H),result(T*H);
  for(int t=0;t<T;++t)for(int j=0;j<H;++j){
   float sum=0;for(int k=0;k<K;++k)sum+=routing[t*K+k]*values[(t*K+k)*H+j];sum+=remote_sums[t*H+j];reduced[t*H+j]=sum;result[t*H+j]=sum+shared[t*H+j];
   if(!std::isfinite(result[t*H+j]))throw std::runtime_error("nonfinite MoE output");
  }
  save(out+".ids.i32",selected.data(),selected.size()*4);save(out+".weights.f32",routing.data(),routing.size()*4);
  save(out+".parts.f32",values.data(),values.size()*4);save(out+".reduced.f32",reduced.data(),reduced.size()*4);
  save(out+".shared.f32",shared.data(),shared.size()*4);save(out+".output.f32",result.data(),result.size()*4);
  std::printf("K2 MoE complete: GPU assignments=%d CPU mapped assignments=%d remote assignments=%d cache_resident=%lld cache_bytes=%lld CPU_arithmetic=DequantizedF32\n",gpu_assignments,cpu_assignments,remote_assignments,(long long)cache.resident(),(long long)cache.bytes());return result;
}

#ifndef STRATA_KIMI2_LIBRARY
int main(int argc,char** argv){
 try{
  if(argc!=3)throw std::runtime_error("usage: strata-kimi2-moe FIXTURE OUTPUT_PREFIX");
  strata::kimi2::run_moe(argv[1],read(std::string(argv[1])+"/input.f32",3*H),argv[2]);return 0;
 }catch(const std::exception& e){std::fprintf(stderr,"kimi2 MoE: %s\n",e.what());return 1;}
}
#endif
