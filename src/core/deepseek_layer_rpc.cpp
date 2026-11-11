#include "strata/core/deepseek_layer_rpc.hpp"
#include "strata/core/network_io.hpp"
#include <limits>
#include <memory>

namespace strata::core {
namespace {
namespace D=deepseek_boundary_detail;
namespace R=deepseek_layer_rpc_detail;
using Clock=std::chrono::steady_clock;
#ifdef STRATA_KIMI2_LIBRARY
constexpr int max_timeout_ms=3600000;
#else
constexpr int max_timeout_ms=600000;
#endif
constexpr uint32_t magic=0x4c345344; // DS4L
constexpr uint16_t version=1;
constexpr size_t result_limit=16u<<20;
int remaining(Clock::time_point deadline) {
    const auto n=std::chrono::duration_cast<std::chrono::milliseconds>(deadline-Clock::now()).count();
    return n<=0 ? 0 : static_cast<int>(std::min<int64_t>(n,max_timeout_ms));
}
size_t limit(R::Type type) {
    switch(type) {
    case R::Type::discover:case R::Type::close:case R::Type::closed:return 0;
    case R::Type::capabilities:case R::Type::ready:return 4096;
    case R::Type::begin:return 144;
    case R::Type::acknowledge:case R::Type::acknowledged:return 136;
    case R::Type::execute:return D::max_bytes+4;
    case R::Type::result:return result_limit;
    case R::Type::error:return 1024;
    }
    throw std::runtime_error("unknown layer frame type");
}
void validate_contract(const DeepSeekLayerContract &c) {
    const auto &h=c.identity;
    D::require(!h.session && !h.sequence && !h.position && !h.rows && deepseek_layer_supported_cut(h.next_layer),
               "layer contract is not a canonical fresh supported-cut identity");
    auto first=h;first.session=1;first.sequence=1;first.rows=1;
    D::layout(c.layout,first);
#ifdef STRATA_KIMI2_LIBRARY
    D::require(c.effective_experts==8 && c.arithmetic==3 && c.layout.hidden==7168 && c.layout.streams==1 &&
               c.layout.result_width && c.layout.result_width<=c.layout.vocab,
               "K2 layer service requires top8, verified decoded-F32 arithmetic and native hidden width");
#else
    D::require(c.effective_experts==6 && c.arithmetic==static_cast<uint32_t>(ExpertWorkerArithmetic::DEEPSEEK_IQ_Q8K),
               "layer service requires effective six experts and qualified Q8_K arithmetic");
#endif
}
template<class Archive,class Contract> void contract_fields(Archive &a,Contract &c) {
    D::fields(a,c.identity);
    a(c.layout.hidden,c.layout.streams,c.layout.topk_stride,c.layout.cand_stride,c.layout.vocab,c.layout.max_context);
#ifdef STRATA_KIMI2_LIBRARY
    a(c.layout.result_width);
#endif
}
void write_contract(ExpertWireWriter &w,const DeepSeekLayerContract &c) {
    validate_contract(c);contract_fields(w,c);
    w.put(static_cast<uint32_t>(c.layout.sources.size()));
    for(const auto &s:c.layout.sources) w(s.layer,s.ratio,s.kv_row_bytes,s.key_row_bytes);
    w(c.effective_experts,c.arithmetic);
}
DeepSeekLayerContract read_contract(ExpertWireReader &r) {
    DeepSeekLayerContract c;contract_fields(r,c);uint32_t count=0;r.get(count);
    D::require(r.ok && count<=40,"invalid layer source descriptor count");
    c.layout.sources.resize(count);
    for(auto &s:c.layout.sources) r(s.layer,s.ratio,s.kv_row_bytes,s.key_row_bytes);
    r(c.effective_experts,c.arithmetic);
    D::require(r.ok,"truncated layer contract");validate_contract(c);return c;
}
void validate_capabilities(const DeepSeekLayerCapabilities &c) {
    validate_contract(c.contract);
    D::require((c.backend==1 || c.backend==2) && c.incarnation && c.last_session<std::numeric_limits<uint64_t>::max() &&
               std::any_of(c.binary_sha256.begin(),c.binary_sha256.end(),[](uint8_t b){return b!=0;}),"invalid layer worker receipt");
}
std::vector<uint8_t> header_bytes(const DeepSeekBoundaryHeader &h) {ExpertWireWriter w;D::fields(w,h);return std::move(w.bytes);}
DeepSeekBoundaryHeader read_header(const std::vector<uint8_t> &bytes) {
    ExpertWireReader r{bytes};DeepSeekBoundaryHeader h;D::fields(r,h);
    D::require(r.done(),"invalid layer acknowledgement identity");return h;
}
DeepSeekLayerBoundary decode_capture(const DeepSeekLayerContract &contract,const DeepSeekBoundaryHeader &session,
                                    const std::vector<uint8_t> &bytes) {
    D::require(bytes.size()>=148 && bytes.size()<=D::max_bytes,"invalid layer capture length");
    ExpertWireReader r{bytes};uint32_t m=0,n=0;uint16_t v=0,f=0;r(m,v,f,n);
    DeepSeekBoundaryHeader declared;D::fields(r,declared);
    auto expected=session;expected.position=declared.position;expected.sequence=declared.sequence;expected.rows=declared.rows;
    D::require(r.ok && D::equal(expected,declared),"layer capture changed session/plan identity");
    // The codec validates the complete record. DeepSeekLayerSession alone owns
    // the subsequent delivery-position/sequence and retained-duplicate checks.
    return decode_deepseek_boundary(contract.layout,expected,bytes);
}
}

namespace deepseek_layer_rpc_detail {
std::vector<uint8_t> contract_bytes(const DeepSeekLayerContract &c) {ExpertWireWriter w;write_contract(w,c);return std::move(w.bytes);}
std::vector<uint8_t> capability_bytes(const DeepSeekLayerCapabilities &c) {
    validate_capabilities(c);ExpertWireWriter w;write_contract(w,c.contract);
    w(c.binary_sha256,c.backend,c.incarnation,c.last_session);return std::move(w.bytes);
}
DeepSeekLayerCapabilities capabilities(const std::vector<uint8_t> &bytes) {
    ExpertWireReader r{bytes};DeepSeekLayerCapabilities c;c.contract=read_contract(r);
    r(c.binary_sha256,c.backend,c.incarnation,c.last_session);
    D::require(r.done(),"truncated layer capabilities or trailing bytes");validate_capabilities(c);return c;
}
bool send_until(int fd,const Frame &frame,Clock::time_point deadline,std::string &error) {
    try {
        D::require(deadline>Clock::now() && frame.request_id && frame.payload.size()<=limit(frame.type),"invalid layer frame or deadline");
        ExpertWireWriter w;w.bytes.reserve(24+frame.payload.size());
        w(magic,version,static_cast<uint16_t>(frame.type),frame.request_id,static_cast<uint32_t>(frame.payload.size()),uint32_t{0});
        w.bytes.insert(w.bytes.end(),frame.payload.begin(),frame.payload.end());
        return network_detail::transfer(fd,w.bytes.data(),w.bytes.size(),true,deadline,&error);
    } catch(const std::exception &e) {error=e.what();return false;}
}
bool receive_until(int fd,Frame &frame,Clock::time_point deadline,std::string &error) {
    try {
        D::require(deadline>Clock::now(),"invalid layer receive deadline");
        std::vector<uint8_t> prefix(24);
        if(!network_detail::transfer(fd,prefix.data(),prefix.size(),false,deadline,&error)) return false;
        ExpertWireReader r{prefix};uint32_t m=0,n=0,flags=0;uint16_t v=0,type=0;Frame candidate;
        r(m,v,type,candidate.request_id,n,flags);candidate.type=static_cast<Type>(type);
        D::require(r.done() && m==magic && v==version && !flags && candidate.request_id && n<=limit(candidate.type),"invalid layer frame header");
        candidate.payload.resize(n);
        if(!network_detail::transfer(fd,candidate.payload.data(),candidate.payload.size(),false,deadline,&error)) return false;
        frame=std::move(candidate);return true;
    } catch(const std::exception &e) {error=e.what();return false;}
}
bool send(int fd,const Frame &frame,int timeout_ms,std::string &error) {
    if(timeout_ms<=0 || timeout_ms>max_timeout_ms) {error="invalid layer send deadline";return false;}
    return send_until(fd,frame,Clock::now()+std::chrono::milliseconds(timeout_ms),error);
}
bool receive(int fd,Frame &frame,int timeout_ms,std::string &error) {
    if(timeout_ms<=0 || timeout_ms>max_timeout_ms) {error="invalid layer receive deadline";return false;}
    return receive_until(fd,frame,Clock::now()+std::chrono::milliseconds(timeout_ms),error);
}

}

DeepSeekLayerService::DeepSeekLayerService(DeepSeekLayerCapabilities c,Prepare prepare,DeepSeekLayerSession::Executor executor)
    : capabilities_(std::move(c)),prepare_(std::move(prepare)),executor_(std::move(executor)) {
    validate_capabilities(capabilities_);
    D::require(static_cast<bool>(prepare_) && static_cast<bool>(executor_),"missing native layer service adapter");
}
bool DeepSeekLayerService::serve(int fd,int timeout_ms,std::string &error) {
    if(timeout_ms<=0 || timeout_ms>max_timeout_ms) {error="invalid layer service deadline";return false;}
    R::Frame request;uint64_t previous=0;bool discovered=false;
    std::unique_ptr<DeepSeekLayerSession> session;
    DeepSeekBoundaryHeader identity;
    auto reject=[&](const std::string &reason) {
        error=reason;std::string send_error;
        std::vector<uint8_t> message(reason.begin(),reason.begin()+std::min<size_t>(reason.size(),1024));
        if(request.request_id && !R::send(fd,{R::Type::error,request.request_id,std::move(message)},timeout_ms,send_error))
            error+="; failure reply could not be delivered: "+send_error;
        return false;
    };
    if(!expert_worker_set_socket_opts(fd,timeout_ms,&error)) return false;
    try {
        for(;;) {
            if(!R::receive(fd,request,timeout_ms,error)) return false;
            if(request.request_id<=previous) return reject("layer request IDs must strictly increase");
            previous=request.request_id;
            auto reply=[&](R::Type type,std::vector<uint8_t> bytes) {return R::send(fd,{type,request.request_id,std::move(bytes)},timeout_ms,error);};
            switch(request.type) {
            case R::Type::discover:
                if(discovered || session || !request.payload.empty()) return reject("repeated or malformed layer discovery");
                discovered=true;if(!reply(R::Type::capabilities,R::capability_bytes(capabilities_))) return false;
                break;
            case R::Type::begin: {
                if(!discovered || session) return reject("layer BEGIN before discovery or during an active session");
                ExpertWireReader r{request.payload};uint64_t incarnation=0;r.get(incarnation);D::fields(r,identity);
                auto expected=capabilities_.contract.identity;expected.session=identity.session;
                if(!r.done() || incarnation!=capabilities_.incarnation || !D::equal(identity,expected) ||
                   identity.session<=capabilities_.last_session || identity.session==std::numeric_limits<uint64_t>::max())
                    return reject("stale layer incarnation/session or different plan");
                // Consume the session before preparation: a callback can fail after
                // partially resetting native state, so this ID cannot be reused.
                capabilities_.last_session=identity.session;
                std::string preparation_error;
                if(!prepare_(identity,preparation_error)) return reject(preparation_error.empty() ? "native layer preparation failed without a reason" : preparation_error);
                session=std::make_unique<DeepSeekLayerSession>(capabilities_.contract.layout,identity,identity.next_layer);
                if(!reply(R::Type::ready,R::capability_bytes(capabilities_))) return false;
                break;
            }
            case R::Type::execute: {
                if(!session || request.payload.size()<4) return reject("layer EXECUTE before READY or without row count");
                ExpertWireReader r{request.payload};uint32_t rows=0;r.get(rows);
                const std::vector<uint8_t> bytes(request.payload.begin()+4,request.payload.end());
                const auto capture=decode_capture(capabilities_.contract,identity,bytes);
                DeepSeekLayerResult result;
                if(!session->execute(capture,rows,executor_,result,error)) return reject(error);
                if(!reply(R::Type::result,encode_deepseek_layer_result(capabilities_.contract.layout,result))) return false;
                break;
            }
            case R::Type::acknowledge: {
                if(!session) return reject("layer ACK before READY");
                const auto acknowledged=read_header(request.payload);
                if(!session->acknowledge(acknowledged,error)) return reject(error);
                if(!reply(R::Type::acknowledged,header_bytes(acknowledged))) return false;
                break;
            }
            case R::Type::close:
                if(!request.payload.empty() || (session && session->awaiting_acknowledgement())) return reject("layer CLOSE before acknowledgement");
                return reply(R::Type::closed,{});
            default:return reject("unexpected layer request type");
            }
        }
    } catch(const std::exception &e) {return reject(e.what());}
}

DeepSeekLayerClient::~DeepSeekLayerClient() {expert_worker_close(fd_);}
void DeepSeekLayerClient::interrupt() {
    std::lock_guard lock(socket_mutex_);
    interrupted_=true;
    // Shutdown wakes receive/send without releasing the descriptor. Only the
    // owning transaction closes it, so cancellation cannot hit a reused fd.
    expert_worker_shutdown(fd_);
}
bool DeepSeekLayerClient::fail(const std::string &reason,std::string &error) {
    std::lock_guard lock(socket_mutex_);
    error=interrupted_ ? "layer channel interrupted: "+reason:reason;
    expert_worker_shutdown(fd_);expert_worker_close(fd_);begun_=false;return false;
}
bool DeepSeekLayerClient::exchange(R::Type request,const std::vector<uint8_t> &payload,R::Type expected,
    std::vector<uint8_t> &reply,Clock::time_point deadline,std::string &error) {
    if(fd_<0 || next_id_==std::numeric_limits<uint64_t>::max()) return fail("closed or exhausted layer channel",error);
    const uint64_t id=next_id_++;
    if(!R::send_until(fd_,{request,id,payload},deadline,error)) return fail(error,error);
    R::Frame response;
    if(!R::receive_until(fd_,response,deadline,error)) return fail(error,error);
    if(response.request_id!=id) return fail("unexpected layer response request ID",error);
    if(response.type==R::Type::error) return fail("layer worker rejected request: "+std::string(response.payload.begin(),response.payload.end()),error);
    if(response.type!=expected) return fail("unexpected layer response type",error);
    reply=std::move(response.payload);return true;
}
bool DeepSeekLayerClient::connect(const std::string &endpoint,const DeepSeekLayerContract &expected,
    const std::array<uint8_t,32> &binary,int timeout_ms,std::string &error) {
    if(fd_>=0) {error="layer channel is already connected";return false;}
    try {
        validate_contract(expected);D::require(timeout_ms>0 && timeout_ms<=max_timeout_ms,"invalid layer transaction deadline");
        D::require(std::any_of(binary.begin(),binary.end(),[](uint8_t b){return b!=0;}),"missing pinned worker executable identity");
        timeout_ms_=timeout_ms;next_id_=1;begun_=false;
        {std::lock_guard lock(socket_mutex_);interrupted_=false;}
        std::string host;uint16_t port=0;
        D::require(expert_worker_parse_host_port(endpoint,host,port),"invalid numeric layer endpoint");
        const auto deadline=Clock::now()+std::chrono::milliseconds(timeout_ms_);
        const int connect_ms=remaining(deadline);
        D::require(connect_ms>0,"layer connection deadline exceeded");
        const int connected=expert_worker_connect(host,port,connect_ms,&error);
        {
            std::unique_lock lock(socket_mutex_);fd_=connected;
            if(interrupted_) {lock.unlock();return fail("connection completed after cancellation",error);}
        }
        if(fd_<0) return false;
        std::vector<uint8_t> reply;
        if(!exchange(R::Type::discover,{},R::Type::capabilities,reply,deadline,error)) return false;
        auto actual=R::capabilities(reply);
        D::require(R::contract_bytes(actual.contract)==R::contract_bytes(expected) && actual.binary_sha256==binary,"layer worker model/pack/plan/arithmetic/binary differs");
        std::unique_lock lock(socket_mutex_);
        if(interrupted_) {lock.unlock();return fail("discovery completed after cancellation",error);}
        capabilities_=std::move(actual);return true;
    } catch(const std::exception &e) {return fail(e.what(),error);}
}
bool DeepSeekLayerClient::begin(uint64_t id,std::string &error) {
    if(fd_<0 || begun_) return fail("layer session is disconnected or already begun",error);
    try {
        D::require(id>capabilities_.last_session && id<std::numeric_limits<uint64_t>::max(),"layer session ID is stale or exhausted");
        auto identity=capabilities_.contract.identity;identity.session=id;
        ExpertWireWriter w;w.put(capabilities_.incarnation);D::fields(w,identity);
        std::vector<uint8_t> reply;
        if(!exchange(R::Type::begin,w.bytes,R::Type::ready,reply,Clock::now()+std::chrono::milliseconds(timeout_ms_),error)) return false;
        auto ready=R::capabilities(reply);
        D::require(ready.last_session==id && ready.incarnation==capabilities_.incarnation && ready.binary_sha256==capabilities_.binary_sha256 &&
                   ready.backend==capabilities_.backend && R::contract_bytes(ready.contract)==R::contract_bytes(capabilities_.contract),"layer READY differs from discovery/preparation");
        std::unique_lock lock(socket_mutex_);
        if(interrupted_) {lock.unlock();return fail("preparation completed after cancellation",error);}
        capabilities_=std::move(ready);session_=identity;begun_=true;return true;
    } catch(const std::exception &e) {return fail(e.what(),error);}
}
bool DeepSeekLayerClient::execute(const DeepSeekLayerBoundary &capture,uint32_t rows,DeepSeekLayerResult &result,std::string &error,DeepSeekLayerTiming *timing) {
    if(fd_<0 || !begun_) return fail("layer session is not READY",error);
    try {
        const auto started=Clock::now();
        const auto deadline=started+std::chrono::milliseconds(timeout_ms_);
        DeepSeekLayerTiming measured;
        auto elapsed=[](Clock::time_point begin,Clock::time_point end) {
            return static_cast<uint64_t>(std::chrono::duration_cast<std::chrono::microseconds>(end-begin).count());
        };
        auto identity=session_;identity.position=capture.header.position;identity.sequence=capture.header.sequence;identity.rows=capture.header.rows;
        D::require(D::equal(identity,capture.header),"capture does not belong to this layer session");
        D::require(rows && rows<=capture.header.rows,"invalid requested layer logit rows");
        ExpertWireWriter w;w.put(rows);auto bytes=encode_deepseek_boundary(capabilities_.contract.layout,capture);
        w.bytes.insert(w.bytes.end(),bytes.begin(),bytes.end());
        std::vector<uint8_t> reply;
        Clock::time_point execute_started;
        if(timing) execute_started=Clock::now();
        if(!exchange(R::Type::execute,w.bytes,R::Type::result,reply,deadline,error)) return false;
        if(timing) {
            measured.execute_us=elapsed(execute_started,Clock::now());
            measured.tx_bytes=24+w.bytes.size();measured.rx_bytes=24+reply.size();
        }
        auto decoded=decode_deepseek_layer_result(capabilities_.contract.layout,capture.header,rows,reply);
        auto acknowledgement=header_bytes(capture.header);
        Clock::time_point ack_started;
        if(timing) ack_started=Clock::now();
        if(!exchange(R::Type::acknowledge,acknowledgement,R::Type::acknowledged,reply,deadline,error)) return false;
        if(timing) {
            measured.ack_us=elapsed(ack_started,Clock::now());
            measured.tx_bytes+=24+acknowledgement.size();measured.rx_bytes+=24+reply.size();
        }
        D::require(D::equal(read_header(reply),capture.header),"layer ACKED identity differs");
        std::unique_lock lock(socket_mutex_);
        if(interrupted_) {lock.unlock();return fail("acknowledgement completed after cancellation",error);}
        result=std::move(decoded);
        if(timing) {measured.total_us=elapsed(started,Clock::now());*timing=measured;}
        return true;
    } catch(const std::exception &e) {return fail(e.what(),error);}
}
bool DeepSeekLayerClient::close(std::string &error) {
    if(fd_<0) {error="layer channel is closed";return false;}
    std::vector<uint8_t> reply;
    if(!exchange(R::Type::close,{},R::Type::closed,reply,Clock::now()+std::chrono::milliseconds(timeout_ms_),error)) return false;
    if(!reply.empty()) return fail("unexpected layer CLOSED payload",error);
    std::unique_lock lock(socket_mutex_);
    if(interrupted_) {lock.unlock();return fail("close completed after cancellation",error);}
    expert_worker_close(fd_);begun_=false;return true;
}
} // namespace strata::core
