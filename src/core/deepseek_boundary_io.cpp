#include "strata/core/deepseek_boundary_io.hpp"
#include "strata/core/network_io.hpp"

namespace strata::core {
bool send_deepseek_boundary(int fd, const DeepSeekBoundaryLayout &layout,
    const DeepSeekLayerBoundary &capture, int timeout_ms, std::string &error) {
    if (timeout_ms <= 0) { error = "invalid boundary send deadline"; return false; }
    const auto deadline = std::chrono::steady_clock::now()+std::chrono::milliseconds(timeout_ms);
    try {
        auto bytes = encode_deepseek_boundary(layout,capture);
        return network_detail::transfer(fd,bytes.data(),bytes.size(),true,deadline,&error);
    } catch (const std::exception &e) { error = e.what(); return false; }
}

bool receive_deepseek_boundary(int fd, const DeepSeekBoundaryLayout &layout,
    const DeepSeekBoundaryHeader &expected, DeepSeekLayerBoundary &capture,
    int timeout_ms, std::string &error) {
    if (timeout_ms <= 0) { error = "invalid boundary receive deadline"; return false; }
    const auto deadline = std::chrono::steady_clock::now()+std::chrono::milliseconds(timeout_ms);
    try {
        namespace D = deepseek_boundary_detail;
        D::layout(layout,expected);
        std::vector<uint8_t> bytes(12);
        if (!network_detail::transfer(fd,bytes.data(),bytes.size(),false,deadline,&error)) return false;
        ExpertWireReader r{bytes};
        uint32_t magic = 0, length = 0; uint16_t version = 0, flags = 0;
        r(magic,version,flags,length);
        D::require(r.done() && magic == D::magic && version == D::version && !flags &&
                   length >= 140 && length <= D::max_bytes-12,"invalid boundary stream header");
        bytes.resize(12+length);
        if (!network_detail::transfer(fd,bytes.data()+12,length,false,deadline,&error)) return false;
        auto decoded = decode_deepseek_boundary(layout,expected,bytes);
        capture = std::move(decoded);
        return true;
    } catch (const std::exception &e) { error = e.what(); return false; }
}
} // namespace strata::core
