#pragma once

#include <atomic>
#include <chrono>
#include <cerrno>
#include <condition_variable>
#include <cstdint>
#include <functional>
#include <mutex>
#include <optional>
#include <string>
#include <stdexcept>
#include <thread>
#include <utility>
#include <poll.h>
#include <unistd.h>

namespace strata::ds4::network {
/// Bounded, joined stdin reader for the one-request layer process. STOP/QUIT
/// interrupt an active channel; no reader thread outlives its callback owner.
class CommandInput {
public:
    explicit CommandInput(int fd,int request_ms=0):fd_(fd),request_ms_(request_ms>=0 ? request_ms:throw std::invalid_argument("negative request timeout")),reader_([this](std::stop_token stop){read(stop);}) {}
    ~CommandInput() {reader_.request_stop();reader_.join();}
    CommandInput(const CommandInput&)=delete;
    CommandInput &operator=(const CommandInput&)=delete;
    /// Admit one line; a STOP received while it was queued remains effective.
    bool next(std::string &line) {
        std::unique_lock lock(mutex_);changed_.wait(lock,[&]{return pending_.has_value() || quit_;});
        if(!pending_) return false;
        line=std::move(*pending_);pending_.reset();active_=true;cancelled_=pending_cancelled_ || quit_;
        deadline_=request_ms_>0 ? (Clock::now()+std::chrono::milliseconds(request_ms_)).time_since_epoch().count():0;return true;
    }
    /// Unregister an operation's callback before destroying the captured client.
    void unbind() {std::lock_guard lock(mutex_);interrupt_={};}
    /// Bind a thread-safe interrupt operation for the current request only.
    void bind(std::function<void()> callback) {
        std::lock_guard lock(mutex_);
        if(!active_ || interrupt_) throw std::logic_error("invalid layer cancellation binding");
        interrupt_=std::move(callback);if(cancelled_ && interrupt_) interrupt_();
    }
    /// End admission before publishing DONE/ERR; the next request starts cleanly.
    void finish() {std::lock_guard lock(mutex_);interrupt_={};active_=false;deadline_=0;}
    bool cancelled() const {return cancelled_.load() || expired();}
    std::string failure() const {std::lock_guard lock(mutex_);return !failure_.empty() ? failure_:expired() ? "layer request deadline exceeded":"";}
private:
    using Clock=std::chrono::steady_clock;
    bool expired() const {const auto deadline=deadline_.load();return deadline>0 && Clock::now().time_since_epoch().count()>=deadline;}
    void cancel_locked() {cancelled_=true;pending_cancelled_=true;if(interrupt_) interrupt_();changed_.notify_all();}
    void fail_locked(std::string reason) {failure_=std::move(reason);quit_=true;cancel_locked();}
    void line(std::string text) {
        if(!text.empty() && text.back()=='\r') text.pop_back();
        std::lock_guard lock(mutex_);
        if(text=="STOP") {if(active_ || pending_) cancel_locked();return;}
        if(text=="QUIT") {quit_=true;cancel_locked();return;}
        if(quit_) return;
        if(active_ || pending_) {fail_locked("layer process accepts one request at a time");return;}
        pending_=std::move(text);pending_cancelled_=false;changed_.notify_all();
    }
    void read(std::stop_token stop) {
        std::string buffered;char chunk[4096];
        while(!stop.stop_requested()) {
            if(expired()) {std::lock_guard lock(mutex_);if(active_ && expired()) {fail_locked("layer request deadline exceeded");return;}}
            pollfd input{fd_,POLLIN,0};const int ready=::poll(&input,1,100);
            if(ready<0 && errno==EINTR) continue;
            if(ready<0) {std::lock_guard lock(mutex_);fail_locked("stdin poll failed");return;}
            if(!ready) continue;
            if(input.revents&POLLNVAL) {std::lock_guard lock(mutex_);fail_locked("stdin descriptor is invalid");return;}
            if(!(input.revents&(POLLIN|POLLHUP))) {std::lock_guard lock(mutex_);fail_locked("stdin poll returned an unexpected event");return;}
            const auto size=::read(fd_,chunk,sizeof chunk);
            if(size<0 && errno==EINTR) continue;
            if(size<0) {std::lock_guard lock(mutex_);fail_locked("stdin read failed");return;}
            if(!size) {
                if(!buffered.empty()) {std::lock_guard lock(mutex_);fail_locked("stdin ended inside a command");return;}
                std::lock_guard lock(mutex_);quit_=true;cancel_locked();return;
            }
            buffered.append(chunk,static_cast<size_t>(size));
            for(;;) {
                const auto newline=buffered.find('\n');
                if(newline==std::string::npos) break;
                if(newline>32768) {std::lock_guard lock(mutex_);fail_locked("layer command exceeds 32768-byte bound");return;}
                line(buffered.substr(0,newline));buffered.erase(0,newline+1);
            }
            if(buffered.size()>32768) {std::lock_guard lock(mutex_);fail_locked("layer command exceeds 32768-byte bound");return;}
        }
    }
    int fd_;
    int request_ms_;
    mutable std::mutex mutex_;
    std::condition_variable changed_;
    std::optional<std::string> pending_;
    std::function<void()> interrupt_;
    std::atomic<bool> cancelled_{false};
    std::atomic<int64_t> deadline_{0};
    bool active_=false,pending_cancelled_=false,quit_=false;
    std::string failure_;
    std::jthread reader_;
};

/// Clear a reader callback before its captured native client leaves scope.
struct InputBinding {
    CommandInput &input;
    InputBinding(CommandInput &owner,std::function<void()> callback):input(owner) {input.bind(std::move(callback));}
    ~InputBinding() {input.unbind();}
    InputBinding(const InputBinding&)=delete;
    InputBinding &operator=(const InputBinding&)=delete;
};
} // namespace strata::ds4::network
