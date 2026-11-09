// Real stream framing/fault tests with synthetic state. No model replay claim.
#include "strata/core/deepseek_boundary_io.hpp"
#include <cerrno>
#include <chrono>
#include <cstdio>
#include <sys/socket.h>
#include <thread>
#include <unistd.h>

using namespace strata::core;
namespace {
using Clock = std::chrono::steady_clock;
void check(bool ok, const char *why) { if (!ok) throw std::runtime_error(why); }
struct Pair {
    int fd[2]{-1,-1};
    Pair() { check(::socketpair(AF_UNIX,SOCK_STREAM,0,fd)==0,"socketpair failed"); }
    ~Pair() { for (auto &n : fd) expert_worker_close(n); }
};
DeepSeekLayerBoundary record(const DeepSeekBoundaryLayout &l, uint32_t rows) {
    DeepSeekLayerBoundary b;
    b.header.model[0]=1; b.header.pack[0]=2; b.header.plan[0]=3;
    b.header.epoch=1; b.header.session=2; b.header.sequence=3;
    b.header.position=127; b.header.rows=rows; b.header.next_layer=24;
    b.tokens.assign(rows,42); b.topk_valid.assign(rows,1); b.cand_valid.assign(rows,1);
    b.h.assign(static_cast<size_t>(rows)*l.streams*l.hidden,.125f); b.h[0]=-0.0f;
    b.pre.assign(static_cast<size_t>(rows)*8,.25f);
    b.topk.assign(static_cast<size_t>(rows)*l.topk_stride,0);
    b.cand.assign(static_cast<size_t>(rows)*l.cand_stride,0);
    for (const auto &s : l.sources) {
        DeepSeekBoundaryCache c;
        c.source_layer=s.layer; c.first_row=b.header.position/s.ratio;
        c.rows=static_cast<uint32_t>((b.header.position+rows)/s.ratio-c.first_row);
        c.kv.assign(static_cast<size_t>(c.rows)*s.kv_row_bytes,0xa5);
        c.keys.assign(static_cast<size_t>(c.rows)*s.key_row_bytes,0x5a);
        b.caches.push_back(std::move(c));
    }
    return b;
}
// The writer deliberately does not call the receiver's shared transfer helper.
bool write_chunks(int fd, const std::vector<uint8_t> &bytes, size_t chunk, bool trickle=false) {
    size_t offset=0;
    while (offset<bytes.size()) {
        const size_t count=std::min(chunk,bytes.size()-offset);
        const auto n=::send(fd,bytes.data()+offset,count,MSG_NOSIGNAL);
        if (n<0 && errno==EINTR) continue;
        if (n<=0) return false;
        offset+=static_cast<size_t>(n);
        if (trickle) std::this_thread::sleep_for(std::chrono::milliseconds(15));
    }
    return true;
}
void put32(std::vector<uint8_t> &bytes, size_t offset, uint32_t value) {
    for (size_t i=0;i<4;++i) bytes.at(offset+i)=static_cast<uint8_t>(value>>(8*i));
}
}

int main() {
    try {
        DeepSeekBoundaryLayout l{5120,4,512,2048,129280,2048,{{20,1,32,16}}};
        for (uint32_t rows : {1u,16u}) {
            const auto b=record(l,rows); const auto bytes=encode_deepseek_boundary(l,b);
            // Ordinary TCP uses the same socket creation helpers as expert RPC,
            // but the separately versioned DS4B bytes are never STEX messages.
            std::string error;
            uint16_t port=0;
            int listener=expert_worker_listen("127.0.0.1",0,1,&port,&error);
            check(listener>=0,error.c_str());
            int client=expert_worker_connect("127.0.0.1",port,1000,&error);
            if (client<0) { expert_worker_close(listener); throw std::runtime_error(error); }
            int server=::accept(listener,nullptr,nullptr); expert_worker_close(listener);
            if (server<0) { expert_worker_close(client); throw std::runtime_error("accept failed"); }
            bool sent=false; std::string send_error;
            std::jthread writer([&]{ sent=send_deepseek_boundary(client,l,b,2000,send_error); });
            DeepSeekLayerBoundary got;
            const bool received=receive_deepseek_boundary(server,l,b.header,got,2000,error);
            if (!received) expert_worker_shutdown(server);
            writer.join(); expert_worker_close(client); expert_worker_close(server);
            check(received,error.c_str()); check(sent,send_error.c_str());
            check(encode_deepseek_boundary(l,got)==bytes,"TCP roundtrip changed capture bits");
            // One-byte prefix followed by uneven body fragments.
            Pair sockets; bool fragmented=false;
            std::jthread fragments([&]{
                const std::vector<uint8_t> prefix(bytes.begin(),bytes.begin()+12);
                const std::vector<uint8_t> body(bytes.begin()+12,bytes.end());
                fragmented=write_chunks(sockets.fd[0],prefix,1) && write_chunks(sockets.fd[0],body,1023);
            });
            const bool assembled=receive_deepseek_boundary(sockets.fd[1],l,b.header,got,2000,error);
            if (!assembled) expert_worker_shutdown(sockets.fd[1]);
            fragments.join(); check(assembled,error.c_str()); check(fragmented,"fragment writer failed");
            check(encode_deepseek_boundary(l,got)==bytes,"fragment assembly changed bits");
            std::printf("DS4B rows=%u TCP and fragmented stream PASS\n",rows);
        }
        const auto b=record(l,1); const auto bytes=encode_deepseek_boundary(l,b);
        auto refuse=[&](std::vector<uint8_t> bad, const DeepSeekBoundaryHeader &expected) {
            Pair sockets; std::string error; auto got=b;
            const auto original=encode_deepseek_boundary(l,got);
            std::jthread writer([&]{
                (void)write_chunks(sockets.fd[0],bad,1023);
                ::shutdown(sockets.fd[0],SHUT_WR);
            });
            const bool accepted=receive_deepseek_boundary(sockets.fd[1],l,expected,got,500,error);
            expert_worker_shutdown(sockets.fd[1]); writer.join();
            check(!accepted && !error.empty(),"invalid stream accepted without error");
            check(encode_deepseek_boundary(l,got)==original,"invalid stream changed output capture");
        };
        for (size_t length : {size_t{0},size_t{7},size_t{12},size_t{150},bytes.size()-1})
            refuse(std::vector<uint8_t>(bytes.begin(),bytes.begin()+length),b.header);
        for (size_t at : {size_t{0},size_t{4},size_t{6}}) {
            auto bad=bytes; bad.resize(12); bad[at]^=0x10; refuse(std::move(bad),b.header);
        }
        for (uint32_t length : {0u,139u,static_cast<uint32_t>(deepseek_boundary_detail::max_bytes),0xffffffffu}) {
            auto bad=bytes; bad.resize(12); put32(bad,8,length); refuse(std::move(bad),b.header);
        }
        auto stale=b.header; ++stale.session; refuse(bytes,stale);
        {
            Pair sockets; std::string error; auto got=b; bool sent=true;
            std::jthread trickle([&]{sent=write_chunks(sockets.fd[0],bytes,1,true);});
            const auto start=Clock::now();
            const bool accepted=receive_deepseek_boundary(sockets.fd[1],l,b.header,got,80,error);
            const auto elapsed=Clock::now()-start;
            expert_worker_shutdown(sockets.fd[1]); trickle.join();
            check(!accepted && !sent && error.find("deadline")!=std::string::npos,"trickle did not fail by deadline");
            check(elapsed<std::chrono::milliseconds(500),"progress renewed the receive deadline");
        }
        {
            Pair sockets; std::string error; auto got=b;
            check(!receive_deepseek_boundary(sockets.fd[1],l,b.header,got,0,error),"accepted zero deadline");
            check(!send_deepseek_boundary(sockets.fd[0],l,b,0,error),"accepted zero send deadline");
            auto bad=b; bad.h.pop_back();
            check(!send_deepseek_boundary(sockets.fd[0],l,bad,500,error),"sent malformed capture");
            char probe=0;
            check(::recv(sockets.fd[1],&probe,1,MSG_DONTWAIT)<0 && (errno==EAGAIN || errno==EWOULDBLOCK),
                  "malformed send emitted bytes");
            const int tiny=1024;
            check(::setsockopt(sockets.fd[0],SOL_SOCKET,SO_SNDBUF,&tiny,sizeof(tiny))==0,"send buffer setup failed");
            const auto start=Clock::now();
            check(!send_deepseek_boundary(sockets.fd[0],l,record(l,16),80,error) &&
                  error.find("deadline")!=std::string::npos,"blocked sender ignored deadline");
            check(Clock::now()-start<std::chrono::milliseconds(500),"blocked sender exceeded deadline bound");
        }
        std::puts("DS4B bounded stream/fault tests PASS; executor replay remains separate");
        return 0;
    } catch (const std::exception &e) { std::fprintf(stderr,"%s\n",e.what()); return 1; }
}
