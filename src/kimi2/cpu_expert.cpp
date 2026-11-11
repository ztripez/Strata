// K2 real mapped expert through the existing CPU pool, with a serial native-row reference.
#include "strata/kernels/cpu/pool.hpp"
#include <sys/mman.h>
#include <fcntl.h>
#include <unistd.h>
#include <fstream>
#include <cstdio>
#include <stdexcept>
#include <string>
#include <vector>
namespace C=strata::kernels::cpu;
namespace {
struct Mapping {
    int fd=-1;const uint8_t* p=nullptr;size_t size;
    Mapping(const std::string& path,size_t bytes):size(bytes){
        fd=open(path.c_str(),O_RDONLY);if(fd<0)throw std::runtime_error("open failed: "+path);
        if(lseek(fd,0,SEEK_END)!=static_cast<off_t>(size))throw std::runtime_error("wrong expert tensor size");
        void* address=mmap(nullptr,size,PROT_READ,MAP_PRIVATE,fd,0);
        if(address==MAP_FAILED)throw std::runtime_error("mmap failed");p=static_cast<const uint8_t*>(address);
    }
    ~Mapping(){if(p)munmap(const_cast<uint8_t*>(p),size);if(fd>=0)close(fd);}
};
void save(const std::string& path,const std::vector<float>& data){
    std::ofstream f(path,std::ios::binary);f.write(reinterpret_cast<const char*>(data.data()),data.size()*4);
    if(!f)throw std::runtime_error("output write failed");
}
}
int main(int argc,char** argv){
 try{
    if(argc!=3 && argc!=4)throw std::runtime_error("usage: strata-kimi2-cpu-expert FIXTURE OUTPUT_PREFIX [--f32]");
    const bool full=argc==4;
    if(full && std::string(argv[3])!="--f32")throw std::runtime_error("unknown arithmetic option");
    C::NativeFmt f;std::string error;
    if(!C::native_fmt(21,23,7168,2048,f,error,true))throw std::runtime_error(error);
    const std::string dir=argv[1],out=argv[2];
    Mapping gate(dir+"/gate.quant",f.gu_row*f.n_ff),up(dir+"/up.quant",f.gu_row*f.n_ff),down(dir+"/down.quant",f.d_row*f.n_embd);
    // Same modular split-base offsets as ExpertSource's existing mapped-GGUF path.
    f.up_off=reinterpret_cast<uintptr_t>(up.p)-reinterpret_cast<uintptr_t>(gate.p);
    f.down_off=reinterpret_cast<uintptr_t>(down.p)-reinterpret_cast<uintptr_t>(gate.p);
    constexpr int T=3;std::vector<float>x(T*7168),output(T*7168),serial(T*7168);
    std::ifstream input(dir+"/input.f32",std::ios::binary);input.read(reinterpret_cast<char*>(x.data()),x.size()*4);
    if(!input)throw std::runtime_error("input read failed");
    std::vector<std::vector<uint8_t>> acts(T,std::vector<uint8_t>(f.act_bytes));
    C::ExpertJobMulti job;job.blob=gate.p;job.nt=T;
    for(int t=0;t<T;++t){C::native_quant_act(f,x.data()+t*7168,acts[t].data());job.nact[t]=acts[t].data();job.f32act[t]=x.data()+t*7168;job.out[t]=output.data()+t*7168;}
    C::ExpertPool pool(2,false,false);
    pool.run_split_multi_native(f,&job,1,full?C::NativeArithmetic::DequantizedF32:C::NativeArithmetic::WindowInvariant);
    for(int t=0;t<T;++t){
        std::vector<float> hidden(2048);std::vector<uint8_t> hq(f.h_bytes);
        const void* a=acts[t].data();float* h=hidden.data();float* y=serial.data()+t*7168;
        if(full){
            const float* original=x.data()+t*7168;
            C::native_gu_rows_f32(f,gate.p,&original,1,&h,0,2048);
            const float* hidden_input=h;C::native_down_rows_f32(f,gate.p,&hidden_input,1,&y,0,7168);
        }else{
            C::native_gu_rows_invariant(f,gate.p,&a,1,&h,0,2048);C::native_quant_h(f,h,hq.data());
            const void* q=hq.data();C::native_down_rows_invariant(f,gate.p,&q,1,&y,0,7168);
        }
    }
    if(output!=serial)throw std::runtime_error("pool output differs from serial native-row reference");
    save(out+".output.f32",output);save(out+".serial.f32",serial);
    std::printf("K2 mapped CPU pool PASS: act_bytes=%zu h_bytes=%zu tokens=3, exact serial parity\n",f.act_bytes,f.h_bytes);
    return 0;
 }catch(const std::exception& e){std::fprintf(stderr,"kimi2 CPU expert: %s\n",e.what());return 1;}
}
