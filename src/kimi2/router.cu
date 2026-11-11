// K2 router bring-up through Strata's existing dense and native top-k kernels.
#include "ds4_kernels.cuh"
#include "strata/kernels/native_router.hpp"
#include <cuda_runtime.h>
#include <fstream>
#include <cstdio>
#include <stdexcept>
#include <string>
#include <vector>
namespace {
void check(cudaError_t e) {if(e!=cudaSuccess)throw std::runtime_error(cudaGetErrorString(e));}
struct Device {
    float* p=nullptr;size_t n;
    explicit Device(size_t count):n(count){check(cudaMalloc(&p,n*4));}
    ~Device(){if(p)cudaFree(p);}
    Device(const Device&)=delete;Device& operator=(const Device&)=delete;
    void load(const std::string& path){
        std::ifstream f(path,std::ios::binary|std::ios::ate);
        if(!f || static_cast<size_t>(f.tellg())!=n*4)throw std::runtime_error("wrong tensor size: "+path);
        std::vector<float> data(n);f.seekg(0);f.read(reinterpret_cast<char*>(data.data()),n*4);
        if(!f)throw std::runtime_error("tensor read failed");check(cudaMemcpy(p,data.data(),n*4,cudaMemcpyHostToDevice));
    }
    void save(const std::string& path){
        std::vector<float> data(n);check(cudaMemcpy(data.data(),p,n*4,cudaMemcpyDeviceToHost));
        std::ofstream f(path,std::ios::binary);f.write(reinterpret_cast<const char*>(data.data()),n*4);
        if(!f)throw std::runtime_error("output write failed");
    }
};
}
int main(int argc,char** argv){
    try{
        if(argc!=3)throw std::runtime_error("usage: strata-kimi2-router FIXTURE OUTPUT_PREFIX");
        const std::string path=argv[1],out=argv[2];Device x(3*7168),gate(384*7168),bias(384),logits(3*384),ids(3*8),weights(3*8);
        x.load(path+"/input.f32");gate.load(path+"/gate.f32");bias.load(path+"/bias.f32");
        cudaStream_t stream;check(cudaStreamCreate(&stream));
        strata::ds4::gemv_f32(gate.p,x.p,384,7168,3,logits.p,384,stream);
        strata::kernels::native_router_kimi2(logits.p,bias.p,reinterpret_cast<int32_t*>(ids.p),weights.p,3,stream);
        check(cudaStreamSynchronize(stream));check(cudaStreamDestroy(stream));
        logits.save(out+".logits.f32");ids.save(out+".ids.i32");weights.save(out+".weights.f32");
        std::puts("Kimi K2 native biased sigmoid top8 completed");return 0;
    }catch(const std::exception& e){std::fprintf(stderr,"kimi2 router: %s\n",e.what());return 1;}
}
