// Real TCP delivery and fault tests with synthetic state; no GPU/math claim.
#include "strata/core/deepseek_layer_rpc.hpp"
#include "strata/core/network_io.hpp"
#include "deepseek_layer_test_fixture.hpp"
#include <algorithm>
#include <chrono>
#include <cstdio>
#include <exception>
#include <poll.h>
#include <sys/socket.h>
#include <thread>

using namespace strata::core;
using namespace strata::core::test;
namespace {
namespace R=deepseek_layer_rpc_detail;
namespace D=deepseek_boundary_detail;
using Clock=std::chrono::steady_clock;
DeepSeekLayerCapabilities receipt() {
    DeepSeekLayerCapabilities c;
    c.contract.identity=capture().header;
    c.contract.identity.session=c.contract.identity.sequence=c.contract.identity.position=c.contract.identity.rows=0;
    c.contract.layout=layout;c.contract.effective_experts=6;
    c.contract.arithmetic=static_cast<uint32_t>(ExpertWorkerArithmetic::DEEPSEEK_IQ_Q8K);
    c.binary_sha256[0]=42;c.backend=2;c.incarnation=1234;c.last_session=1;return c;
}
// Shutdown before joining even when a test throws, including a blocked accept.
struct Peer {
    int listener=-1;std::atomic<int> accepted{-1};uint16_t port=0;
    std::exception_ptr failure;std::jthread thread;
    template<class F> explicit Peer(F f) {
        std::string error;listener=expert_worker_listen("127.0.0.1",0,1,&port,&error);
        check(listener>=0,error.c_str());
        thread=std::jthread([this,f=std::move(f)] {
            int fd=-1;
            try {
                pollfd pfd{listener,POLLIN,0};check(::poll(&pfd,1,2000)>0,"accept deadline exceeded");
                fd=::accept(listener,nullptr,nullptr);accepted=fd;
                check(fd>=0,"accept failed");f(fd);
            } catch(...) {failure=std::current_exception();}
            expert_worker_close(fd);accepted=-1;
        });
    }
    ~Peer() {expert_worker_shutdown(listener);expert_worker_shutdown(accepted.load());if(thread.joinable()) thread.join();expert_worker_close(listener);}
    std::string endpoint() const {return "127.0.0.1:"+std::to_string(port);}
    void finish() {thread.join();if(failure) std::rethrow_exception(failure);}
};
struct Fd {
    int value=-1;
    ~Fd() {expert_worker_close(value);}
};
R::Frame receive(int fd) {R::Frame f;std::string e;check(R::receive(fd,f,1000,e),e.c_str());return f;}
void send(int fd,R::Type type,uint64_t id,std::vector<uint8_t> bytes={}) {
    std::string e;check(R::send(fd,{type,id,std::move(bytes)},1000,e),e.c_str());
}
std::vector<uint8_t> header(const DeepSeekBoundaryHeader &h) {ExpertWireWriter w;D::fields(w,h);return std::move(w.bytes);}
std::vector<uint8_t> begin_bytes(uint64_t boot=1234,uint64_t session=7) {
    auto h=receipt().contract.identity;h.session=session;ExpertWireWriter w;w.put(boot);D::fields(w,h);return std::move(w.bytes);
}
std::vector<uint8_t> execute_bytes(const DeepSeekLayerBoundary &b,uint32_t rows=1) {
    ExpertWireWriter w;w.put(rows);const auto bytes=encode_deepseek_boundary(layout,b);
    w.bytes.insert(w.bytes.end(),bytes.begin(),bytes.end());return std::move(w.bytes);
}
void discover_begin(int fd) {
    send(fd,R::Type::discover,1);check(receive(fd).type==R::Type::capabilities,"missing CAPS");
    send(fd,R::Type::begin,2,begin_bytes());check(receive(fd).type==R::Type::ready,"missing READY");
}
void fake_ready(int fd) {
    auto c=receipt();auto f=receive(fd);check(f.type==R::Type::discover,"expected discovery");
    send(fd,R::Type::capabilities,f.request_id,R::capability_bytes(c));
    f=receive(fd);check(f.type==R::Type::begin,"expected BEGIN");c.last_session=7;
    send(fd,R::Type::ready,f.request_id,R::capability_bytes(c));
}
DeepSeekLayerResult mock_result() {NativeMock n;DeepSeekLayerResult r;std::string e;check(n(capture(),1,r,e),e.c_str());return r;}
void happy() {
    NativeMock native;int preparations=0;bool served=false;std::string server_error;
    auto caps=receipt();DeepSeekLayerService service(caps,[&](const auto &h,std::string &) {++preparations;return h.session==7;},std::ref(native));
    Peer peer([&](int fd){served=service.serve(fd,1000,server_error);});
    DeepSeekLayerClient client;std::string e;
    check(client.connect(peer.endpoint(),caps.contract,caps.binary_sha256,1000,e),e.c_str());
    check(preparations==0,"discovery prepared native state");
    check(client.begin(7,e),e.c_str());DeepSeekLayerResult r;
    check(client.execute(capture(),1,r,e),e.c_str());
    check(r.committed_position==3 && r.logits_rows==1 && r.logits[0]==.25f,"first TCP output wrong");
    const auto second=capture(2,3,16);DeepSeekLayerTiming timing;
    const bool executed=client.execute(second,16,r,e,&timing);check(executed,e.c_str());
    NativeMock independent;DeepSeekLayerResult expected;
    check(independent(capture(),1,expected,e) && independent(second,16,expected,e),e.c_str());
    check(r.logits==expected.logits && r.committed_position==expected.committed_position,
          "observed output differs from the independent native callback");
    check(timing.tx_bytes==48+execute_bytes(second,16).size()+header(second.header).size() &&
          timing.rx_bytes==48+encode_deepseek_layer_result(layout,r).size()+header(second.header).size(),
          "timing did not count actual EXECUTE/RESULT/ACK/ACKED framing bytes");
    check(timing.total_us>=timing.execute_us+timing.ack_us,"transaction timing double-counted spans");
    check(r.committed_position==19 && r.logits.size()==16*layout.vocab,"second TCP output wrong");
    check(client.close(e),e.c_str());peer.finish();
    check(served,server_error.c_str());check(preparations==1 && native.calls==2 && native.position==19,"TCP executed wrong number of windows");
    check(service.capabilities().last_session==7,"service did not retain session fence");
}
void fresh_connections() {
    NativeMock native;int preparations=0;
    DeepSeekLayerService service(receipt(),[&](const auto &,std::string &) {++preparations;native.position=0;return true;},std::ref(native));
    for(uint64_t session:{7u,8u}) {
        bool served=false;std::string why;
        Peer peer([&](int fd){served=service.serve(fd,1000,why);});
        DeepSeekLayerClient client;std::string e;const auto caps=receipt();
        check(client.connect(peer.endpoint(),caps.contract,caps.binary_sha256,1000,e),e.c_str());
        check(client.capabilities().last_session==(session==7 ? 1u:7u),"fresh connection forgot consumed session floor");
        check(client.begin(session,e),e.c_str());auto b=capture();b.header.session=session;DeepSeekLayerResult out;
        check(client.execute(b,1,out,e) && out.committed_position==3,"fresh native session reused prior position");
        check(client.close(e),e.c_str());peer.finish();check(served,why.c_str());
    }
    check(preparations==2 && native.calls==2 && native.position==3 && service.capabilities().last_session==8,
          "reconnected delivery failed to reset/fence native state");
}
void discovery_faults() {
    const auto caps=receipt();
    for(int fault=0;fault<8;++fault) {
        int prepared=0;NativeMock native;DeepSeekLayerService service(caps,[&](const auto &,std::string &){++prepared;return true;},std::ref(native));
        Peer peer([&](int fd){std::string e;check(!service.serve(fd,500,e),"mismatched client unexpectedly completed");});
        auto expected=caps.contract;auto binary=caps.binary_sha256;
        if(fault==0) expected.identity.model[0]^=8;
        if(fault==1) expected.identity.pack[0]^=1;
        if(fault==2) expected.identity.plan[0]^=1;
        if(fault==3) ++expected.identity.epoch;
        if(fault==4) ++expected.layout.hidden;
        if(fault==5) binary[0]^=1;
        // Invalid local contracts fail before connecting, so use valid wire
        // differences here. Invalid policies are tested separately below.
        if(fault==6) ++expected.layout.max_context;
        if(fault==7) ++expected.layout.sources[0].kv_row_bytes;
        DeepSeekLayerClient client;std::string e;
        check(!client.connect(peer.endpoint(),expected,binary,500,e) && !e.empty() && !client.healthy(),"pin mismatch accepted");
        peer.finish();check(prepared==0 && native.calls==0,"pin mismatch reached native executor");
    }
    for(int fault=0;fault<5;++fault) {
        Peer peer([&](int fd) {
            const auto request=receive(fd);auto bytes=R::capability_bytes(caps);
            if(fault==0) bytes.at(188)=1; // Q8_K -> unqualified per32
            if(fault==1) bytes.at(184)=5; // experts
            if(fault==2) bytes.at(224)=3; // backend
            if(fault==3) bytes.resize(bytes.size()-1);
            if(fault==4) bytes.push_back(0);
            send(fd,R::Type::capabilities,request.request_id,std::move(bytes));
        });
        DeepSeekLayerClient client;std::string e;
        check(!client.connect(peer.endpoint(),caps.contract,caps.binary_sha256,500,e) && !client.healthy(),"malformed capability accepted");peer.finish();
    }
    auto invalid=caps;invalid.contract.effective_experts=5;bool rejected=false;
    try {R::capability_bytes(invalid);} catch(const std::runtime_error &) {rejected=true;}
    check(rejected,"unqualified local policy accepted");
}
void raw_delivery_faults() {
    for(int fault=0;fault<7;++fault) {
        NativeMock native;int prepared=0;bool served=true;std::string why;
        DeepSeekLayerService service(receipt(),[&](const auto &,std::string &){++prepared;return true;},std::ref(native));
        Peer peer([&](int fd){served=service.serve(fd,1000,why);});
        std::string e;Fd fd{expert_worker_connect("127.0.0.1",peer.port,1000,&e)};check(fd.value>=0,e.c_str());
        if(fault<2) {
            send(fd.value,R::Type::discover,1);receive(fd.value);
            send(fd.value,R::Type::begin,2,begin_bytes(fault==0 ? 1235:1234,fault==1 ? 1:7));
            check(receive(fd.value).type==R::Type::error,"stale boot/session accepted");
        } else {
            discover_begin(fd.value);const auto b=capture();
            send(fd.value,R::Type::execute,3,execute_bytes(b));const auto result=receive(fd.value);
            check(result.type==R::Type::result,"missing result");
            send(fd.value,R::Type::execute,4,execute_bytes(b));const auto duplicate=receive(fd.value);
            check(duplicate.payload==result.payload && native.calls==1,"duplicate window executed twice");
            if(fault==2) {auto changed=b;changed.h[0]+=1;send(fd.value,R::Type::execute,5,execute_bytes(changed));}
            if(fault==3) send(fd.value,R::Type::execute,5,execute_bytes(capture(2,3,1)));
            if(fault==4) send(fd.value,R::Type::close,5);
            if(fault==5) {send(fd.value,R::Type::acknowledge,5,header(b.header));check(receive(fd.value).type==R::Type::acknowledged,"ACK failed");send(fd.value,R::Type::execute,6,execute_bytes(b));}
            if(fault==6) {auto bad=b.header;++bad.sequence;send(fd.value,R::Type::acknowledge,5,header(bad));}
            check(receive(fd.value).type==R::Type::error,"delivery conflict accepted");
        }
        peer.finish();check(!served && !why.empty(),"invalid service session remained healthy");
        check(native.calls==(fault<2 ? 0:1) && prepared==(fault<2 ? 0:1),"fault mutated native state");
    }
}
void preparation_fence() {
    NativeMock native;int calls=0;std::string why;
    DeepSeekLayerService service(receipt(),[&](const auto &,std::string &e) {++calls;e="partial native preparation failure";return false;},std::ref(native));
    for(int attempt=0;attempt<2;++attempt) {
        Peer peer([&](int fd){check(!service.serve(fd,500,why),"failed preparation accepted");});
        std::string e;Fd fd{expert_worker_connect("127.0.0.1",peer.port,500,&e)};check(fd.value>=0,e.c_str());
        send(fd.value,R::Type::discover,1);auto discovered=R::capabilities(receive(fd.value).payload);
        check(discovered.last_session==(attempt ? 7u:1u),"failed native preparation lost its session fence");
        send(fd.value,R::Type::begin,2,begin_bytes());
        check(receive(fd.value).type==R::Type::error,"failed/reused preparation accepted");peer.finish();
    }
    check(calls==1 && native.calls==0 && service.capabilities().last_session==7,"failed preparation retried same native session");
}
void unexplained_preparation_failure() {
    NativeMock native;std::string why="obsolete failure from an earlier connection";
    DeepSeekLayerService service(receipt(),[](const auto &,std::string &) {return false;},std::ref(native));
    Peer peer([&](int fd){check(!service.serve(fd,500,why),"unexplained native preparation accepted");});
    std::string e;Fd fd{expert_worker_connect("127.0.0.1",peer.port,500,&e)};check(fd.value>=0,e.c_str());
    send(fd.value,R::Type::discover,1);receive(fd.value);send(fd.value,R::Type::begin,2,begin_bytes());
    const auto reply=receive(fd.value);peer.finish();
    check(reply.type==R::Type::error && why=="native layer preparation failed without a reason" && native.calls==0,
          "stale connection error hid an unexplained native preparation failure");
}
void response_faults() {
    const auto caps=receipt();
    for(int fault=0;fault<9;++fault) {
        Peer peer([&](int fd) {
            fake_ready(fd);const auto request=receive(fd);check(request.type==R::Type::execute,"expected EXECUTE");
            if(fault==0) return; // disconnect after receipt, before result
            if(fault==1) {std::this_thread::sleep_for(std::chrono::milliseconds(180));return;}
            auto result=mock_result();auto bytes=encode_deepseek_layer_result(layout,result);
            if(fault==2) ++result.request.session;
            if(fault==2) bytes=encode_deepseek_layer_result(layout,result);
            if(fault==3) {bytes.at(160)=0;bytes.at(161)=0;bytes.at(162)=0xc0;bytes.at(163)=0x7f;}
            send(fd,fault==4 ? R::Type::ready:R::Type::result,request.request_id+(fault==5 ? 1:0),std::move(bytes));
            if(fault<6) return;
            const auto ack=receive(fd);check(ack.type==R::Type::acknowledge,"expected ACK");
            if(fault==6) return; // execution succeeded, ACK response lost
            auto id=capture().header;if(fault==7) ++id.sequence;
            send(fd,fault==8 ? R::Type::result:R::Type::acknowledged,ack.request_id,header(id));
        });
        DeepSeekLayerClient client;std::string e;
        check(client.connect(peer.endpoint(),caps.contract,caps.binary_sha256,120,e),e.c_str());check(client.begin(7,e),e.c_str());
        DeepSeekLayerResult result;result.logits={42};result.committed_position=99;
        check(!client.execute(capture(),1,result,e) && !e.empty() && !client.healthy(),"bad response accepted or left channel open");
        check(result.logits==std::vector<float>({42}) && result.committed_position==99,"failed transaction published output");
        check(!client.execute(capture(),1,result,e),"failed transaction transparently retried");peer.finish();
    }
    for(int fault=0;fault<4;++fault) {
        Peer peer([&](int fd) {
            const auto discover=receive(fd);send(fd,R::Type::capabilities,discover.request_id,R::capability_bytes(caps));
            const auto begin=receive(fd);auto ready=caps;ready.last_session=7;
            if(fault==0) ++ready.incarnation;
            if(fault==1) ++ready.last_session;
            if(fault==2) ready.backend=1;
            if(fault==3) ++ready.contract.identity.plan[0];
            send(fd,R::Type::ready,begin.request_id,R::capability_bytes(ready));
        });
        DeepSeekLayerClient client;std::string e;
        check(client.connect(peer.endpoint(),caps.contract,caps.binary_sha256,500,e),e.c_str());
        check(!client.begin(7,e) && !client.healthy(),"changed READY receipt accepted");peer.finish();
    }
}
void transaction_deadline() {
    const auto caps=receipt();
    Peer peer([&](int fd) {
        fake_ready(fd);const auto request=receive(fd);
        std::this_thread::sleep_for(std::chrono::milliseconds(100));
        send(fd,R::Type::result,request.request_id,encode_deepseek_layer_result(layout,mock_result()));
        const auto ack=receive(fd);std::this_thread::sleep_for(std::chrono::milliseconds(100));
        std::string e; // Expected: peer has already timed out and closed.
        R::send(fd,{R::Type::acknowledged,ack.request_id,ack.payload},500,e);
    });
    DeepSeekLayerClient client;std::string e;
    check(client.connect(peer.endpoint(),caps.contract,caps.binary_sha256,160,e),e.c_str());check(client.begin(7,e),e.c_str());
    DeepSeekLayerResult result;result.logits={42};const auto start=Clock::now();
    DeepSeekLayerTiming timing{77,2,3,4,5};const auto sentinel=timing;
    check(!client.execute(capture(),1,result,e,&timing) && e.find("deadline")!=std::string::npos,"EXECUTE/ACK renewed transaction deadline");
    check(timing==sentinel,"failed transaction published timing");
    check(Clock::now()-start<std::chrono::milliseconds(400) && !client.healthy() && result.logits==std::vector<float>({42}),"deadline published output or left channel open");peer.finish();
}
void interruption() {
    const auto caps=receipt();
    // Cancel discovery, BEGIN, result delivery, ACK delivery and CLOSE while
    // the peer deliberately withholds that operation's response.
    for(int stage=0;stage<5;++stage) {
        DeepSeekLayerClient client;std::atomic<bool> waiting{false},observed{false};
        Clock::time_point interrupted_at;
        Peer peer([&](int fd) {
            auto request=receive(fd);check(request.type==R::Type::discover,"missing discovery");
            if(stage>0) {
                send(fd,R::Type::capabilities,request.request_id,R::capability_bytes(caps));
                request=receive(fd);check(request.type==R::Type::begin,"missing BEGIN");
                if(stage>1) {
                    auto ready=caps;ready.last_session=7;
                    send(fd,R::Type::ready,request.request_id,R::capability_bytes(ready));
                    request=receive(fd);
                    if(stage==4) check(request.type==R::Type::close,"missing CLOSE");
                    else {
                        check(request.type==R::Type::execute,"missing EXECUTE");
                        if(stage==3) {
                            send(fd,R::Type::result,request.request_id,encode_deepseek_layer_result(layout,mock_result()));
                            request=receive(fd);check(request.type==R::Type::acknowledge,"missing ACK");
                        }
                    }
                }
            }
            waiting=true;
            R::Frame next;std::string error;
            check(!R::receive(fd,next,1500,error),"interrupted client sent another request");
        });
        std::jthread canceller([&] {
            const auto deadline=Clock::now()+std::chrono::milliseconds(1000);
            while(!waiting && Clock::now()<deadline) std::this_thread::sleep_for(std::chrono::milliseconds(1));
            observed=waiting.load();interrupted_at=Clock::now();client.interrupt();
        });
        std::string error;
        bool ok=client.connect(peer.endpoint(),caps.contract,caps.binary_sha256,2000,error);
        if(stage>0) {check(ok,error.c_str());ok=client.begin(7,error);}
        DeepSeekLayerResult result;result.logits={42};result.committed_position=99;
        DeepSeekLayerTiming timing{77,2,3,4,5};const auto sentinel=timing;
        if(stage>1) {
            check(ok,error.c_str());
            ok=stage==4 ? client.close(error):client.execute(capture(),1,result,error,&timing);
        }
        canceller.join();
        check(observed && !ok && error.find("interrupted")!=std::string::npos && !client.healthy(),
              "interrupted operation succeeded, lacked reason or left channel healthy");
        check(Clock::now()-interrupted_at<std::chrono::milliseconds(1000),"interruption waited for full transaction deadline");
        check(result.logits==std::vector<float>({42}) && result.committed_position==99,"interruption published unacknowledged output");
        check(timing==sentinel,"interruption published unacknowledged timing");
        client.interrupt(); // An already-failed channel remains safely interrupted.
        check(!client.execute(capture(),1,result,error),"interrupted transaction retried");peer.finish();

        // A new connection is explicit and starts a fresh service/session. No
        // cancellation caller remains alive when the same client is reused.
        NativeMock native;
        DeepSeekLayerService service(caps,[](const auto &,std::string &){return true;},std::ref(native));
        Peer fresh([&](int fd){std::string why;check(service.serve(fd,1000,why),why.c_str());});
        check(client.connect(fresh.endpoint(),caps.contract,caps.binary_sha256,1000,error),error.c_str());
        check(client.begin(7,error) && client.execute(capture(),1,result,error) && client.close(error),error.c_str());
        fresh.finish();check(native.calls==1,"fresh explicit session failed or retried work");
    }
    DeepSeekLayerClient idle;idle.interrupt();idle.interrupt();
    check(!idle.healthy(),"unconnected interrupted channel is healthy");
}
void framing() {
    int fd[2]{-1,-1};check(::socketpair(AF_UNIX,SOCK_STREAM,0,fd)==0,"socketpair failed");Fd a{fd[0]},b{fd[1]};
    send(a.value,R::Type::discover,0x0102030405060708ULL);
    std::vector<uint8_t> bytes(24);check(::recv(b.value,bytes.data(),bytes.size(),MSG_WAITALL)==24,"golden recv failed");
    const std::vector<uint8_t> expected{0x44,0x53,0x34,0x4c,1,0,1,0,8,7,6,5,4,3,2,1,0,0,0,0,0,0,0,0};
    check(bytes==expected,"DS4L golden frame changed");
    for(int fault=0;fault<7;++fault) {
        auto bad=expected;
        if(fault==0) bad[0]^=1;
        if(fault==1) bad[4]=2;
        if(fault==2) bad[6]=99;
        if(fault==3) bad[16]=1; // DISCOVER cannot allocate any payload
        if(fault==4) bad[20]=1;
        if(fault==5) for(size_t i=8;i<16;++i) bad[i]=0;
        if(fault==6) {bad[6]=static_cast<uint8_t>(R::Type::result);bad[19]=0xff;}
        check(::send(a.value,bad.data(),bad.size(),MSG_NOSIGNAL)==24,"fault write failed");
        R::Frame got{R::Type::closed,42,{1}};std::string e;
        check(!R::receive(b.value,got,200,e) && !e.empty() && got.request_id==42 && got.payload==std::vector<uint8_t>({1}),"bad frame accepted/changed output");
    }
    for(size_t i=0;i<expected.size();++i) check(::send(a.value,&expected[i],1,MSG_NOSIGNAL)==1,"fragment write failed");
    R::Frame got;std::string e;check(R::receive(b.value,got,200,e) && got.request_id==0x0102030405060708ULL,"fragmented frame failed");
    check(!R::send(a.value,{R::Type::discover,1,{}},0,e),"zero send deadline accepted");
    check(!R::receive(b.value,got,0,e),"zero receive deadline accepted");
    check(::send(a.value,expected.data(),7,MSG_NOSIGNAL)==7,"truncated write failed");::shutdown(a.value,SHUT_WR);
    check(!R::receive(b.value,got,200,e) && !e.empty(),"truncated frame accepted");
}
// Build wire bytes independently of R::send so read tests do not share the
// encoder under test. TCP writes deliberately cross header/payload boundaries.
std::vector<uint8_t> wire_frame(const R::Frame &frame) {
    ExpertWireWriter w;
    w(uint32_t{0x4c345344},uint16_t{1},static_cast<uint16_t>(frame.type),
      frame.request_id,static_cast<uint32_t>(frame.payload.size()),uint32_t{0});
    w.bytes.insert(w.bytes.end(),frame.payload.begin(),frame.payload.end());
    return std::move(w.bytes);
}
void payload_stream_framing() {
    const std::vector<R::Frame> expected{
        {R::Type::capabilities,1,R::capability_bytes(receipt())},
        {R::Type::execute,2,execute_bytes(capture(2,3,16),16)},
        {R::Type::result,3,encode_deepseek_layer_result(layout,mock_result())},
        {R::Type::acknowledged,4,header(capture().header)},
        {R::Type::closed,5,{}}
    };
    std::vector<uint8_t> stream;
    for(const auto &frame:expected) {
        const auto bytes=wire_frame(frame);stream.insert(stream.end(),bytes.begin(),bytes.end());
    }
    // One write coalesces frames; irregular writes fragment both headers and
    // native boundary payloads. Receivers must preserve exactly one frame each.
    for(bool fragmented:{false,true}) {
        Peer peer([&](int fd) {
            const auto deadline=Clock::now()+std::chrono::seconds(2);
            const size_t chunks[]{1,2,7,31,4093};size_t offset=0,index=0;
            while(offset<stream.size()) {
                const size_t size=std::min(stream.size()-offset,
                    fragmented ? chunks[index++%5]:stream.size());
                std::string e;
                check(network_detail::transfer(fd,stream.data()+offset,size,true,deadline,&e),e.c_str());
                offset+=size;
                if(fragmented) std::this_thread::sleep_for(std::chrono::milliseconds(1));
            }
        });
        std::string e;Fd fd{expert_worker_connect("127.0.0.1",peer.port,1000,&e)};
        check(fd.value>=0,e.c_str());
        for(const auto &frame:expected) {
            R::Frame got;
            check(R::receive(fd.value,got,1000,e),e.c_str());
            check(got.type==frame.type && got.request_id==frame.request_id && got.payload==frame.payload,
                  "TCP split/coalesced payload changed frame boundaries or bytes");
        }
        peer.finish();
    }
    const auto bytes=wire_frame(expected[1]);
    for(size_t cut:{size_t{0},size_t{23},size_t{24},size_t{25},bytes.size()/2,bytes.size()-1}) {
        Peer peer([&](int fd) {
            auto prefix=bytes;std::string e;
            check(network_detail::transfer(fd,prefix.data(),cut,true,Clock::now()+std::chrono::seconds(1),&e),e.c_str());
            check(::shutdown(fd,SHUT_WR)==0,"could not half-close truncated TCP frame");
        });
        std::string e;Fd fd{expert_worker_connect("127.0.0.1",peer.port,1000,&e)};
        check(fd.value>=0,e.c_str());R::Frame got{R::Type::closed,42,{1,2,3}};
        check(!R::receive(fd.value,got,1000,e) && e.find("disconnected")!=std::string::npos,
              "truncated TCP payload did not report disconnection");
        check(got.type==R::Type::closed && got.request_id==42 && got.payload==std::vector<uint8_t>({1,2,3}),
              "partial TCP frame published output");
        peer.finish();
    }
}
}
int main() {
    try {framing();payload_stream_framing();happy();fresh_connections();discovery_faults();raw_delivery_faults();preparation_fence();unexplained_preparation_failure();response_faults();transaction_deadline();interruption();
        std::puts("DS4L real TCP contract/delivery/fault/deadline/interruption tests PASS; native inference remains separate");return 0;
    } catch(const std::exception &e) {std::fprintf(stderr,"%s\n",e.what());return 1;}
}
