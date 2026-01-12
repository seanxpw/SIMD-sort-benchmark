#pragma once
#include <fcntl.h>
#include <unistd.h>
#include <cerrno>
#include <cstring>
#include <cstdlib>
#include <stdexcept>
#include <string>

class PerfFifoController {
public:
  PerfFifoController() {
    const char* ctl = std::getenv("PERF_CTL_FIFO");
    const char* ack = std::getenv("PERF_ACK_FIFO");
    if (!ctl || !ack) return; // 未配置则自动禁用

    enabled_ = true;

    // open() FIFO 可能会阻塞；perf stat 已经打开了对端，一般不会卡。
    ctl_fd_ = ::open(ctl, O_WRONLY);
    if (ctl_fd_ < 0) throw std::runtime_error(std::string("open PERF_CTL_FIFO failed: ") + std::strerror(errno));

    ack_fd_ = ::open(ack, O_RDONLY);
    if (ack_fd_ < 0) throw std::runtime_error(std::string("open PERF_ACK_FIFO failed: ") + std::strerror(errno));
  }

  ~PerfFifoController() {
    if (ctl_fd_ >= 0) ::close(ctl_fd_);
    if (ack_fd_ >= 0) ::close(ack_fd_);
  }

  bool active() const { return enabled_; }

  void enable()  { send_cmd("enable\n"); }
  void disable() { send_cmd("disable\n"); }

private:
  int ctl_fd_ = -1;
  int ack_fd_ = -1;
  bool enabled_ = false;

  void send_cmd(const char* s) {
    if (!enabled_) return;

    ssize_t n = ::write(ctl_fd_, s, std::strlen(s));
    if (n < 0) throw std::runtime_error(std::string("write ctl fifo failed: ") + std::strerror(errno));

    // perf 会回 "ack\n"
    char ack[4];
    size_t got = 0;
    while (got < sizeof(ack)) {
      ssize_t r = ::read(ack_fd_, ack + got, sizeof(ack) - got);
      if (r < 0) throw std::runtime_error(std::string("read ack fifo failed: ") + std::strerror(errno));
      if (r == 0) throw std::runtime_error("ack fifo EOF");
      got += (size_t)r;
    }
    // 可选：校验内容是否为 "ack\n"
  }
};
