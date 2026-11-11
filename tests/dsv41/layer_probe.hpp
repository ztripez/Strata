#pragma once

#include "ds4_model.hpp"
#include <openssl/evp.h>
#include <array>
#include <charconv>
#include <cstring>
#include <cstdio>
#include <filesystem>
#include <fstream>
#include <iterator>
#include <stdexcept>

namespace strata::ds4::test {
inline void check(bool ok, const std::string &why) { if (!ok) throw std::runtime_error(why); }
inline int number(const char *text, int maximum) {
    int n = 0; const auto end = text+std::strlen(text);
    const auto p = std::from_chars(text,end,n);
    if (p.ec != std::errc() || p.ptr != end || n < 0 || n > maximum) throw std::runtime_error("invalid probe argument");
    return n;
}
inline std::array<uint8_t,32> digest(const std::string &input) {
    std::array<uint8_t,32> out{}; unsigned int n=0;
    check(EVP_Digest(input.data(),input.size(),out.data(),&n,EVP_sha256(),nullptr)==1 && n==out.size(),"probe digest failed");
    return out;
}
inline void identity_receipt(const char *role, const std::array<uint8_t,32> &model,
                             const std::array<uint8_t,32> &pack) {
    std::printf("%s_IDENTITY engine=%s model_sha256=",role,STRATA_VERSION);
    for (const auto b : model) std::printf("%02x",b);
    std::printf(" pack_sha256=");
    for (const auto b : pack) std::printf("%02x",b);
    std::printf("\n"); std::fflush(stdout);
}
inline std::string read(const std::string &path) {
    std::ifstream f(path,std::ios::binary);
    check(static_cast<bool>(f),"cannot read "+path);
    std::string bytes((std::istreambuf_iterator<char>(f)),{});
    check(!f.bad(),"failed reading "+path);
    return bytes;
}
inline void state(Model &m, std::vector<uint8_t> &out, bool prefix = false) {
    std::string error; check(m.boundary_state(out,error,prefix),error);
}
/// Reject indices inside the allocated context but beyond this window's
/// committed source rows. Check each selection kind at its first live query.
inline unsigned reject_future_selections(Model &suffix, const strata::core::DeepSeekBoundaryLayout &layout,
    const strata::core::DeepSeekLayerBoundary &capture, unsigned tested) {
    const uint64_t future=capture.header.position+capture.header.rows;
    if (future>=layout.max_context) return tested;
    for (unsigned kind=0;kind<2;++kind) {
        if (tested&(1u<<kind)) continue;
        const auto &counts=kind ? capture.cand_valid : capture.topk_valid;
        for (size_t row=0;row<counts.size();++row) {
            if (!counts[row]) continue;
            auto wrong=capture;
            auto &values=kind ? wrong.cand : wrong.topk;
            values[row*(kind ? layout.cand_stride : layout.topk_stride)]=static_cast<int32_t>(future);
            std::vector<uint8_t> before,after; state(suffix,before); std::string error;
            check(!suffix.replay_boundary(wrong,1,error) && error.find("uncommitted")!=std::string::npos,
                  "accepted a future source selection or failed for another reason");
            state(suffix,after); check(before==after,"future source selection mutated state");
            tested|=1u<<kind;
            std::printf("reject=uncommitted-%s position=%llu PASS\n",kind ? "candidate" : "topk",
                        static_cast<unsigned long long>(capture.header.position));
            break;
        }
    }
    return tested;
}
template<class T> void preserve(const std::filesystem::path &path, const std::vector<T> &values) {
    std::ofstream f(path,std::ios::binary);
    f.write(reinterpret_cast<const char*>(values.data()),static_cast<std::streamsize>(values.size()*sizeof(T)));
    f.close(); check(static_cast<bool>(f),"could not preserve mismatch at "+path.string());
}
/// The same deterministic native execution settings for full and assigned
/// oracles. Each probe records its exact source/binary and arithmetic separately.
inline Options options(const std::string &pack) {
    Options o;
    o.pack=pack; o.gpus={0}; o.tp=1; o.threads=false; o.max_context=2048;
    o.experts=6; o.dspark=false; o.dspark_auto=false; o.vision=false; o.prefill_chunk=0;
    o.vram_experts=0; o.vram_experts_auto=false; o.expert_cache=false;
    o.ssd_cache_gb=0; o.mmap_experts=1; o.profile="none"; o.profile_readonly=true; o.cpu_threads=2;
    return o;
}
} // namespace strata::ds4::test
