// GPU-free protocol/executor contract tests. Mock arithmetic is independently
// calculated by the client, including sparse tokens and repeated expert inputs.
#include "strata/core/expert_worker_wire.hpp"
#include "strata/core/expert_worker_identity.hpp"
#include "strata/core/network_experts.hpp"
#include <algorithm>
#include <arpa/inet.h>
#include <chrono>
#include <cmath>
#include <cstdio>
#include <cstring>
#include <fstream>
#include <limits>
#include <memory>
#include <netinet/tcp.h>
#include <set>
#include <stdexcept>
#include <sys/socket.h>
#include <thread>
#include <unistd.h>
using namespace strata::core;
using Clock = std::chrono::steady_clock;
namespace {
constexpr uint32_t N = 2048, FF = 512, L = 40, E = 256, K = 8;
const std::array<uint8_t, 32> identity{1, 2, 3, 4};
void check(bool ok, const char *why) {
    if (!ok)
        throw std::runtime_error(why);
}
enum class Fault {
    none,
    identity,
    arithmetic,
    caps_version,
    result_version,
    caps_id,
    ready_count,
    stale,
    layer,
    count,
    coverage,
    nan,
    short_payload,
    disconnect,
    timeout
};
float oracle(float x, int layer, int expert, int col) {
    return x * (1.0f + expert * 0.01f) + (layer + col % 7) * 0.001f;
}

void mock_session(int fd, Fault fault, bool nodelay = true) {
    std::string err;
    check(expert_worker_set_socket_opts(fd, 1500, &err), err.c_str());
    if (!nodelay) {
        int zero = 0;
        setsockopt(fd, IPPROTO_TCP, TCP_NODELAY, &zero, sizeof(zero));
    }
    ExpertWorkerFrameHeader hdr{};
    std::vector<uint8_t> bytes;
    std::set<std::pair<int, int>> held;
    while (expert_worker_recv_frame(fd, hdr, bytes, &err, 1500)) {
        const auto type = static_cast<ExpertWorkerMsgType>(hdr.msg_type);
        if (type == ExpertWorkerMsgType::HELLO) {
            ExpertWorkerHelloPayload hp{};
            uint32_t arithmetic=0;
            if (hdr.version == kExpertWorkerArithmeticVersion) {
                ExpertWorkerHelloV2 hello{};
                check(expert_wire_decode(bytes,hello),"mock HELLO v2 decode");
                hp=hello.model; arithmetic=hello.arithmetic;
            } else check(expert_wire_decode(bytes, hp), "mock HELLO decode");
            ExpertWorkerCapsPayload cp{};
            cp.model_fingerprint = compute_model_fingerprint(L, E, N, FF);
            cp.model_sha256 = identity;
            cp.n_embd = N;
            cp.n_ff = FF;
            cp.n_layers = L;
            cp.n_experts = E;
            cp.top_k = K;
            cp.endianness = 0x01020304;
            cp.max_tokens = 16;
            cp.max_slots = L * E;
            std::strcpy(cp.device_name, "CPU_MOCK");
            cp.status = hp.model_sha256 == identity ? 0 : 2;
            if (fault == Fault::identity)
                cp.model_sha256[0] ^= 1;
            if (hdr.version == kExpertWorkerArithmeticVersion)
                expert_worker_send_message(fd,ExpertWorkerMsgType::CAPS,hdr.request_id,
                    ExpertWorkerCapsV2{cp,arithmetic+(fault==Fault::arithmetic)},&err,1500,hdr.version);
            else expert_worker_send_message(fd, ExpertWorkerMsgType::CAPS, hdr.request_id + (fault == Fault::caps_id), cp,
                                       &err,1500,fault==Fault::caps_version ? kExpertWorkerArithmeticVersion : hdr.version);
        } else if (type == ExpertWorkerMsgType::INSTALL_PLAN) {
            ExpertWireReader r{bytes};
            uint32_t n = 0;
            r.get(n);
            check(n <= L * E, "mock plan bound");
            for (uint32_t i = 0; i < n; ++i) {
                ExpertWorkerPair p{};
                r.get(p);
                held.emplace(p.layer, p.expert);
            }
            check(r.done(), "mock plan decode");
            ExpertWorkerReadyPayload rp{};
            rp.loaded_count = static_cast<uint32_t>(held.size()) + (fault == Fault::ready_count);
            expert_worker_send_message(fd, ExpertWorkerMsgType::READY, hdr.request_id, rp, &err,1500,hdr.version);
        } else if (type == ExpertWorkerMsgType::EXECUTE) {
            if (fault == Fault::disconnect)
                break;
            if (fault == Fault::timeout) {
                std::this_thread::sleep_for(std::chrono::milliseconds(250));
                break;
            }
            ExpertWireReader r{bytes};
            ExpertWorkerExecuteFixed ef{};
            r.get(ef);
            check(ef.n_tokens <= 16 && ef.n_embd == N && ef.n_assignments <= ef.n_tokens * K, "mock execute bound");
            std::vector<ExpertWorkerAssignment> assigns(ef.n_assignments);
            for (auto &a : assigns)
                r.get(a);
            std::vector<float> acts(ef.n_tokens * N), sums(ef.n_tokens * N, 0);
            for (auto &v : acts)
                r.get(v);
            std::vector<uint8_t> mask(ef.n_tokens, 0);
            check(r.done(), "mock execute decode");
            for (auto a : assigns) {
                check(a.token_idx < ef.n_tokens && held.count({ef.layer, a.expert_id}), "mock residency");
                mask[a.token_idx] = 1;
                for (uint32_t c = 0; c < N; ++c)
                    sums[a.token_idx * N + c] += a.weight * oracle(acts[a.token_idx * N + c], ef.layer, a.expert_id, c);
            }
            ExpertWorkerResultFixed rf{};
            rf.layer = ef.layer + (fault == Fault::layer);
            rf.n_tokens = ef.n_tokens;
            rf.n_embd = N;
            rf.n_assignments_executed = ef.n_assignments + (fault == Fault::count);
            if (fault == Fault::coverage)
                mask[0] ^= 1;
            if (fault == Fault::nan)
                sums[0] = std::numeric_limits<float>::quiet_NaN();
            ExpertWireWriter w;
            w.put(rf);
            for (auto v : mask)
                w.put(v);
            for (auto v : sums)
                w.put(v);
            if (fault == Fault::short_payload)
                w.bytes.pop_back();
            expert_worker_send_frame(fd, ExpertWorkerMsgType::RESULT, hdr.request_id + (fault == Fault::stale),
                                     w.bytes.data(), w.bytes.size(), &err,1500,
                                     fault==Fault::result_version ? (hdr.version==1?2:1) : hdr.version);
        } else if (type == ExpertWorkerMsgType::PING) {
            expert_worker_send_frame(fd, ExpertWorkerMsgType::PONG, hdr.request_id, nullptr, 0, &err,1500,hdr.version);
        } else
            break;
    }
    expert_worker_close(fd);
}
struct Server {
    int fd = -1;
    uint16_t port = 0;
    std::thread thread;
    Server(Fault fault = Fault::none) {
        std::string err;
        fd = expert_worker_listen("127.0.0.1", 0, 1, &port, &err);
        check(fd >= 0, err.c_str());
        thread = std::thread([this, fault] {
            int c = accept(fd, nullptr, nullptr);
            if (c >= 0)
                mock_session(c, fault);
        });
    }
    ~Server() {
        expert_worker_shutdown(fd);
        if (thread.joinable())
            thread.join();
        expert_worker_close(fd);
    }
    std::string endpoint() const { return "127.0.0.1:" + std::to_string(port); }
};
bool connect(NetworkExpertExecutor &net, const std::string &endpoint, int timeout, std::string &err) {
    return net.connect_and_handshake(endpoint, L, L, E, N, FF, K, identity, timeout, err);
}
std::vector<ExpertWorkerPair> plan{{0, 3}, {0, 7}, {1, 5}};
std::vector<ExpertWorkerAssignment> assignments{{0, 0, 3, 0.25f}, {0, 1, 7, 0.75f}, {2, 0, 3, 1.0f}};
void valid_execution(NetworkExpertExecutor &net, int iterations) {
    std::vector<float> acts(N * 3), got(N * 3);
    std::vector<uint8_t> mask(3);
    std::string err;
    for (size_t i = 0; i < acts.size(); ++i)
        acts[i] = static_cast<float>(static_cast<int>(i % 19) - 9) / 17;
    // Duplicate activation rows, sparse token 1, shared expert 3 across rows.
    std::copy(acts.begin(), acts.begin() + N, acts.begin() + 2 * N);
    for (int it = 0; it < iterations; ++it) {
        check(net.begin_async(0, 3, acts.data(), assignments), "begin request");
        check(!net.begin_async(0, 3, acts.data(), assignments), "overlapping request accepted");
        check(net.wait_result(got.data(), mask.data(), &err), err.c_str());
        check(mask == std::vector<uint8_t>({1, 0, 1}), "coverage");
        for (uint32_t t = 0; t < 3; ++t)
            for (uint32_t c = 0; c < N; ++c) {
                double expected = 0;
                for (auto a : assignments)
                    if (a.token_idx == t)
                        expected += a.weight * oracle(acts[t * N + c], 0, a.expert_id, c);
                check(std::abs(got[t * N + c] - expected) < 1e-6, "independent mock arithmetic");
            }
    }
}
void executor_tests() {
    std::string err;
    {
        Server server;
        NetworkExpertExecutor net;
        check(connect(net, server.endpoint(), 1000, err), err.c_str());
        check(!net.holds(0, 3) && !net.holds_layer(0) && !net.has_any_owned(), "ownership before READY");
        check(net.install_plan(plan, err), err.c_str());
        check(net.has_any_owned() && net.holds_layer(0) && net.holds_layer(1) && !net.holds_layer(2) &&
                  !net.holds_layer(-1) && !net.holds_layer(L),
              "layer ownership after READY");
        check(!net.install_plan(plan, err), "mutable plan accepted");
        valid_execution(net, 5);
        std::vector<float> acts(N, 1), out(N);
        check(net.execute_sync(1, 1, acts.data(), {{0, 0, 5, 1}}, out.data(), nullptr, &err), err.c_str());
        FILE *summary = std::tmpfile();
        check(summary != nullptr, "summary tempfile");
        net.print_summary(summary);
        std::rewind(summary);
        std::string report;
        char line[2048];
        while (std::fgets(line, sizeof(line), summary)) report += line;
        check(!std::ferror(summary), "summary read");
        std::fclose(summary);
        check(report.find("layer=0 samples=5") != std::string::npos &&
              report.find("layer=1 samples=1") != std::string::npos, "per-layer sample attribution");
        check(!net.begin_async(0, 1, acts.data(), {{0, 0, 99, 1}}), "nonresident accepted");
        check(!net.begin_async(0, 1, acts.data(), {{0, 0, 3, 1}, {0, 0, 3, 1}}), "duplicate slot accepted");
        check(!net.begin_async(0, 1, acts.data(), {{0, 8, 3, 1}}), "slot bound accepted");
        check(!net.begin_async(0, 1, acts.data(), {}), "empty request accepted");
        net.close();
    }
    {
        Server server;
        NetworkExpertExecutor net;
        check(connect(net, server.endpoint(), 1000, err), err.c_str());
        check(net.install_plan({}, err), err.c_str());
        check(!net.holds(0, 3) && !net.holds_layer(0) && !net.has_any_owned(), "zero plan owns expert");
    }
    for (auto fault : {Fault::identity, Fault::caps_id, Fault::caps_version, Fault::result_version, Fault::ready_count, Fault::stale, Fault::layer, Fault::count,
                       Fault::coverage, Fault::nan, Fault::short_payload, Fault::disconnect, Fault::timeout}) {
        Server server(fault);
        NetworkExpertExecutor net;
        const bool connected = connect(net, server.endpoint(), 100, err);
        if (fault == Fault::identity || fault == Fault::caps_id || fault == Fault::caps_version) {
            check(!connected, "bad CAPS accepted");
            continue;
        }
        check(connected, err.c_str());
        const bool installed = net.install_plan(plan, err);
        if (fault == Fault::ready_count) {
            check(!installed && !net.holds(0, 3), "incomplete READY accepted");
            continue;
        }
        check(installed, err.c_str());
        std::vector<float> acts(3 * N, 1), out(3 * N, 123);
        std::vector<uint8_t> mask(3, 99);
        const auto t0 = Clock::now();
        check(!net.execute_sync(0, 3, acts.data(), assignments, out.data(), mask.data(), &err), "bad RESULT accepted");
        check(!net.healthy() && net.stats().fallbacks == 1, "failed request not disabled");
        check(out[0] == 123 && mask[0] == 99, "failed result published");
        check(Clock::now() - t0 < std::chrono::seconds(1), "deadline exceeded");
    }
    for (const auto policy : {ExpertWorkerArithmetic::DEEPSEEK_IQ_PER32,ExpertWorkerArithmetic::DEEPSEEK_IQ_Q8K}) {
        Server server;
        NetworkExpertExecutor v2;
        check(v2.connect_and_handshake(server.endpoint(),L,L,E,N,FF,K,identity,1000,err,policy),err.c_str());
        check(v2.install_plan(plan,err),err.c_str());
        valid_execution(v2,2);
    }
    {
        Server server(Fault::arithmetic);
        NetworkExpertExecutor v2;
        check(!v2.connect_and_handshake(server.endpoint(),L,L,E,N,FF,K,identity,1000,err,
                                      ExpertWorkerArithmetic::DEEPSEEK_IQ_Q8K),"accepted mismatched arithmetic");
        check(!v2.has_any_owned(),"arithmetic mismatch published ownership");
    }
    {
        Server server(Fault::result_version);
        NetworkExpertExecutor v2;
        check(v2.connect_and_handshake(server.endpoint(),L,L,E,N,FF,K,identity,1000,err,
                                      ExpertWorkerArithmetic::DEEPSEEK_IQ_Q8K),err.c_str());
        check(v2.install_plan(plan,err),err.c_str());
        std::vector<float> acts(N*3,1), out(N*3,123);
        check(!v2.execute_sync(0,3,acts.data(),assignments,out.data(),nullptr,&err),"accepted downgraded v2 RESULT");
        check(out[0]==123,"downgraded result mutated output");
    }
    // Same executor reconnects cleanly; old ownership must never survive.
    NetworkExpertExecutor net;
    for (int i = 0; i < 2; ++i) {
        Server server;
        check(connect(net, server.endpoint(), 1000, err), err.c_str());
        check(!net.holds(0, 3), "stale ownership on reconnect");
        check(net.install_plan(plan, err), err.c_str());
        valid_execution(net, 1);
        net.close();
    }
    for (const auto policy : {ExpertWorkerArithmetic::DEEPSEEK_IQ_Q8K,ExpertWorkerArithmetic::NATIVE_V1,
                              ExpertWorkerArithmetic::DEEPSEEK_IQ_PER32}) {
        Server server;
        check(net.connect_and_handshake(server.endpoint(),L,L,E,N,FF,K,identity,1000,err,policy),err.c_str());
        check(!net.has_any_owned(),"old ownership survived a version change");
        check(net.install_plan(plan,err),err.c_str());
        valid_execution(net,1); net.close();
    }
}
void framing_tests() {
    const ExpertWorkerFrameHeader h{kExpertWorkerMagic, 1, 7, 0x0807060504030201ULL, 0, 0};
    const std::vector<uint8_t> expected{'S', 'T', 'E', 'X', 1, 0, 7, 0, 1, 2, 3, 4, 5, 6, 7, 8, 0, 0, 0, 0, 0, 0, 0, 0};
    check(expert_wire_encode(h) == expected, "wire header golden bytes");
    ExpertWorkerAssignment a{2, 5, -7, -0.0f};
    auto bytes = expert_wire_encode(a);
    check(bytes == std::vector<uint8_t>({2, 0, 5, 0, 249, 255, 255, 255, 0, 0, 0, 128}),
          "wire assignment golden bytes");
    ExpertWorkerAssignment b{};
    check(expert_wire_decode(bytes, b) && b.expert_id == -7 && std::signbit(b.weight), "float round trip");
    ExpertWorkerHelloPayload hello{};
    hello.model_fingerprint=42; hello.top_k=6;
    const auto v1=expert_wire_encode(hello);
    auto v2=expert_wire_encode(ExpertWorkerHelloV2{hello,2});
    check(v2.size()==v1.size()+4 && std::equal(v1.begin(),v1.end(),v2.begin()) &&
          v2[v1.size()]==2 && v2[v1.size()+1]==0 && v2[v1.size()+2]==0 && v2[v1.size()+3]==0,
          "v2 explicit policy envelope changed v1 fields");
    check(!expert_wire_decode(v2,hello),"v1 decoder accepted a v2 envelope");
    std::string err;
    for (int offset : {0, 4, 6, 16, 20}) {
        int f[2];
        check(socketpair(AF_UNIX, SOCK_STREAM, 0, f) == 0, "socketpair");
        auto bad = expected;
        bad[offset] = 255;
        check(send(f[0], bad.data(), bad.size(), MSG_NOSIGNAL) == 24, "malformed frame send");
        ExpertWorkerFrameHeader out{};
        std::vector<uint8_t> payload;
        check(!expert_worker_recv_frame(f[1], out, payload, &err, 50), "malformed frame accepted");
        close(f[0]);
        close(f[1]);
    }
    // One byte arriving repeatedly must not extend the whole-frame deadline.
    int f[2];
    check(socketpair(AF_UNIX, SOCK_STREAM, 0, f) == 0, "socketpair");
    std::thread trickle([&] {
        for (auto c : expected) {
            if (send(f[0], &c, 1, MSG_NOSIGNAL) != 1)
                break;
            std::this_thread::sleep_for(std::chrono::milliseconds(10));
        }
        close(f[0]);
    });
    const auto t0 = Clock::now();
    ExpertWorkerFrameHeader out{};
    std::vector<uint8_t> payload;
    check(!expert_worker_recv_frame(f[1], out, payload, &err, 50), "trickle timeout");
    close(f[1]);
    trickle.join();
    check(Clock::now() - t0 < std::chrono::milliseconds(200), "trickle extended deadline");
    const std::string path = "/tmp/strata-placement-test-" + std::to_string(getpid());
    {
        std::ofstream file(path);
        file << "# empty\n0 3,7\n1:5\n";
    }
    std::vector<ExpertWorkerPair> pairs;
    check(load_expert_placement_plan(path, L, E, pairs, &err) && pairs.size() == 3, "placement parse");
    {
        std::ofstream file(path);
        file << "garbage\n";
    }
    check(!load_expert_placement_plan(path, L, E, pairs, &err), "garbage placement accepted");
    unlink(path.c_str());
}

void idle_deadline_tests() {
    const auto frame = expert_wire_encode(ExpertWorkerFrameHeader{kExpertWorkerMagic,1,7,42,0,0});
    // A startup pause may exceed the active frame budget only when explicitly enabled.
    for (const int idle_ms : {0, 1000}) {
        int f[2];check(socketpair(AF_UNIX,SOCK_STREAM,0,f)==0,"idle socketpair");
        std::thread sender([&] {
            std::this_thread::sleep_for(std::chrono::milliseconds(200));
            (void)send(f[0],frame.data(),frame.size(),MSG_NOSIGNAL);
            close(f[0]);
        });
        ExpertWorkerFrameHeader header{};std::vector<uint8_t> payload;std::string error;
        const bool ok=expert_worker_recv_frame(f[1],header,payload,&error,100,idle_ms);
        close(f[1]);sender.join();
        check(ok==(idle_ms>0),"idle opt-in did not preserve default frame deadline");
        if(ok) check(header.request_id==42 && payload.empty(),"idle frame changed wire contents");
    }
    // Once the first byte arrives, neither a slow header nor a stalled payload
    // may consume the longer idle budget.
    for(const bool stalled_payload : {false,true}) {
        int f[2];check(socketpair(AF_UNIX,SOCK_STREAM,0,f)==0,"active socketpair");
        auto bytes=frame;if(stalled_payload) bytes[16]=1;
        const size_t sent=stalled_payload ? bytes.size():1;
        check(send(f[0],bytes.data(),sent,MSG_NOSIGNAL)==static_cast<ssize_t>(sent),"active fixture send");
        ExpertWorkerFrameHeader header{};std::vector<uint8_t> payload;std::string error;
        const auto start=Clock::now();
        const bool ok=expert_worker_recv_frame(f[1],header,payload,&error,100,1000);
        const auto elapsed=Clock::now()-start;
        close(f[0]);close(f[1]);
        check(!ok && error=="network frame deadline exceeded","partial active frame escaped deadline");
        check(elapsed<std::chrono::milliseconds(700),"active frame inherited idle deadline");
    }
    int f[2];check(socketpair(AF_UNIX,SOCK_STREAM,0,f)==0,"empty idle socketpair");
    ExpertWorkerFrameHeader header{};std::vector<uint8_t> payload;std::string error;
    const auto start=Clock::now();
    check(!expert_worker_recv_frame(f[1],header,payload,&error,1000,100),"empty idle socket never timed out");
    check(Clock::now()-start<std::chrono::milliseconds(700),"idle deadline used active budget");
    check(!expert_worker_recv_frame(f[1],header,payload,&error,0,100) && error=="invalid expert worker frame deadline",
          "invalid active deadline waited for idle data");
    close(f[0]);
    check(!expert_worker_recv_frame(f[1],header,payload,&error,100,1000) && error=="network peer disconnected",
          "idle disconnect was hidden");
    close(f[1]);
    std::puts("expert worker idle deadline: delayed startup, active header/payload, idle expiry and disconnect PASS");
}
} // namespace
// Run the actual native adapter method against this test's independent TCP
// arithmetic oracle. This minimal context supplies no GPU or alternate kernels.
struct CpuMissAdapterHarness {
    struct Layer {int il=0;bool mtp=false;};
    struct Geometry {int dim=static_cast<int>(N);} g;
    static constexpr int kPicks=8;
    std::unique_ptr<NetworkExpertExecutor> net_worker=std::make_unique<NetworkExpertExecutor>();
    bool net_warned=false;
#include "../../src/dsv41/ds4_network_adapter.inc"
};
void cpu_miss_adapter_tests() {
    constexpr int T=3;
    const std::vector<std::pair<int,int>> picks{{7,5},{2,0},{3,17},{1,20}};
    const std::vector<std::pair<int,int>> locals{{2,0},{1,20}};
    std::vector<float> weights(T*8,9.0f);weights[5]=.25f;weights[17]=.75f;
    // Both native consumers: contiguous synchronous rows and padded async rows.
    for(size_t stride:{static_cast<size_t>(N),static_cast<size_t>(N)+23}) {
        Server server;CpuMissAdapterHarness adapter;std::string error;
        check(adapter.net_worker->connect_and_handshake(server.endpoint(),L,L,E,N,FF,K,identity,1000,error,
              ExpertWorkerArithmetic::DEEPSEEK_IQ_Q8K),error.c_str());
        check(adapter.net_worker->install_plan(plan,error),error.c_str());
        std::vector<float> rows(T*stride,777.0f);
        for(int t=0;t<T;++t) for(uint32_t c=0;c<N;++c) rows[t*stride+c]=t+c*.001f;
        CpuMissAdapterHarness::NetworkCpuParts parts;auto only_local=locals;
        check(!adapter.offload_cpu_picks({},T,rows.data(),stride,weights.data(),only_local,parts) &&
              only_local==locals && parts.assignments.empty() && parts.values.empty() && adapter.net_worker->stats().layers_dispatched==0,
              "CPU-only picks emitted a remote request or changed local work");
        auto delegated=picks;
        check(adapter.offload_cpu_picks({},T,rows.data(),stride,weights.data(),delegated,parts),"native adapter did not delegate held CPU misses");
        check(delegated==locals && parts.assignments.size()==2 && parts.values.size()==2*N &&
              parts.assignments[0].expert_id==3 && parts.assignments[0].token_idx==2 &&
              parts.assignments[1].expert_id==7 && parts.assignments[1].token_idx==0,
              "native adapter removed wrong local picks or changed remote coverage");
        std::vector<float> sums(T*N,42.0f);
        adapter.combine_network_cpu_parts(parts,{},T,sums.data(),N);
        for(uint32_t c=0;c<N;++c) {
            check(std::fabs(sums[c]-.25f*oracle(rows[c],0,7,c))<1e-6f,"remote first-row slot weight/activation changed");
            check(sums[N+c]==0,"unassigned remote row acquired a contribution");
            check(std::fabs(sums[2*N+c]-.75f*oracle(rows[2*stride+c],0,3,c))<1e-6f,"remote padded-row stride/slot weight changed");
        }
        check(adapter.net_worker->stats().assignments_executed==2 && adapter.net_worker->stats().fallbacks==0,
              "native adapter executed missing/double remote assignments");
        delegated=picks;
        check(!adapter.offload_cpu_picks({0,true},T,rows.data(),stride,weights.data(),delegated,parts) && delegated==picks && parts.assignments.empty() && parts.values.empty(),
              "native adapter delegated an unsupported drafter layer");
    }
    for(Fault fault:{Fault::disconnect,Fault::nan,Fault::short_payload}) {
        Server server(fault);CpuMissAdapterHarness adapter;std::string error;
        check(adapter.net_worker->connect_and_handshake(server.endpoint(),L,L,E,N,FF,K,identity,1000,error,
              ExpertWorkerArithmetic::DEEPSEEK_IQ_Q8K),error.c_str());
        check(adapter.net_worker->install_plan(plan,error),error.c_str());
        const std::vector<float> rows(T*N,.25f);auto retained=picks;CpuMissAdapterHarness::NetworkCpuParts parts;
        check(!adapter.offload_cpu_picks({},T,rows.data(),N,weights.data(),retained,parts) && retained==picks && parts.assignments.empty() && parts.values.empty(),
              "failed expert delivery lost local fallback picks or exposed partial sums");
        check(adapter.net_warned && !adapter.net_worker->healthy() && adapter.net_worker->stats().fallbacks==1,
              "failed expert delivery remained healthy or hid its fallback");
        check(!adapter.offload_cpu_picks({},T,rows.data(),N,weights.data(),retained,parts) && retained==picks && adapter.net_worker->stats().fallbacks==1,
              "disabled expert channel was automatically retried");
    }
}
void identity_span_tests() {
    const uint8_t abc[]={0x61,0x62,0x63};
    const std::array<uint8_t,32> expected{0xba,0x78,0x16,0xbf,0x8f,0x01,0xcf,0xea,0x41,0x41,0x40,0xde,0x5d,0xae,0x22,0x23,
        0xb0,0x03,0x61,0xa3,0x96,0x17,0x7a,0x9c,0xb4,0x10,0xff,0x61,0xf2,0x00,0x15,0xad};
    check(expert_worker_spans_sha256({{abc,1},{abc+1,2}})==expected,"loaded span SHA-256 differs from standard abc vector");
    check(expert_worker_spans_sha256({{abc,3}})==expected,"span boundaries changed digest");
    bool rejected=false;try {expert_worker_spans_sha256({{nullptr,3}});} catch(const std::runtime_error &) {rejected=true;}
    check(rejected,"accepted null nonempty loaded mapping");
}
int main(int argc, char **argv) {
    try {
        std::string server, client;
        int iters = 1000, sessions = 1;
        for (int i = 1; i < argc; ++i) {
            std::string a = argv[i];
            check(i + 1 < argc, "option missing value");
            std::string v = argv[++i];
            if (a == "--server")
                server = v;
            else if (a == "--client")
                client = v;
            else if (a == "--iters")
                iters = std::stoi(v);
            else if (a == "--sessions")
                sessions = std::stoi(v);
            else
                check(false, "unknown option");
        }
        if (!server.empty()) {
            std::string ip, err;
            uint16_t port = 0;
            check(expert_worker_parse_host_port(server, ip, port), "server endpoint");
            int fd = expert_worker_listen(ip, port, 1, nullptr, &err);
            check(fd >= 0, err.c_str());
            std::printf("MOCK_SERVER_LISTENING %s\n", server.c_str());
            std::fflush(stdout);
            for (int i = 0; i < sessions; ++i) {
                int c = accept(fd, nullptr, nullptr);
                check(c >= 0, "accept");
                mock_session(c, Fault::none);
            }
            close(fd);
        } else if (!client.empty()) {
            NetworkExpertExecutor net;
            std::string err;
            check(connect(net, client, 1000, err), err.c_str());
            check(net.install_plan(plan, err), err.c_str());
            valid_execution(net, iters);
            net.print_summary(stdout);
        } else {
            identity_span_tests();
            cpu_miss_adapter_tests();
            framing_tests();
            idle_deadline_tests();
            executor_tests();
        }
        std::puts("expert_worker_proto_test: PASS");
        return 0;
    } catch (const std::exception &e) {
        std::fprintf(stderr, "expert_worker_proto_test: FAIL: %s\n", e.what());
        return 1;
    }
}
