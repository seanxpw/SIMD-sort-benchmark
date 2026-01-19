#include <iostream>
#include <vector>
#include <chrono>
#include <cstdlib>
#include <cstring>
#include <iomanip>

// 只包含 parlay 的通用 parallel_for
#include <parlay/parallel.h>

class Timer {
  using clock = std::chrono::steady_clock;
  clock::time_point t0;
public:
  Timer() : t0(clock::now()) {}
  void reset() { t0 = clock::now(); }
  double sec() const {
    return std::chrono::duration_cast<std::chrono::duration<double>>(clock::now() - t0).count();
  }
};

static inline void sink(double x) {
  asm volatile("" : : "r,m"(x) : "memory");
}

double checksum_sample(const std::vector<double>& a) {
  const size_t n = a.size();
  double s = 0.0;
  const size_t step = (n >= 4096 ? n / 4096 : 1);
  for (size_t i = 0; i < n; i += step) s += a[i];
  sink(s);
  return s;
}

int main(int argc, char** argv) {
  size_t N = 200ull * 1000 * 1000; // 200M doubles
  int iters = 5;

  for (int i = 1; i < argc; i++) {
    if (!std::strcmp(argv[i], "--n") && i + 1 < argc) N = std::strtoull(argv[++i], nullptr, 10);
    else if (!std::strcmp(argv[i], "--iters") && i + 1 < argc) iters = std::atoi(argv[++i]);
    else {
      std::cerr << "Usage: " << argv[0] << " [--n <elements>] [--iters <iters>]\n";
      return 1;
    }
  }

  const double alpha = 3.0;

  std::cout << "=========================================\n";
  std::cout << "STREAM Triad Benchmark (ORIG)\n";
  std::cout << "N: " << N << " doubles\n";
  std::cout << "Per-array: " << std::fixed << std::setprecision(2)
            << (N * sizeof(double) / 1024.0 / 1024.0) << " MB\n";
  std::cout << "Total(3 arrays): "
            << (3.0 * N * sizeof(double) / 1024.0 / 1024.0) << " MB\n";
  std::cout << "Iters: " << iters << "\n";
  std::cout << "Workers: " << parlay::num_workers() << "\n";
  std::cout << "=========================================\n";

  std::vector<double> a(N), b(N), c(N);

  Timer t;
  // baseline-good：并行 first-touch（很多机器上已经不错）
  parlay::parallel_for(0, N, [&](size_t i) {
    a[i] = 0.0;
    b[i] = double(i) * 0.5;
    c[i] = double(i) * 0.25;
  },N/1000000);
  double t_init = t.sec();

  t.reset();
  for (int rep = 0; rep < iters; rep++) {
    parlay::parallel_for(0, N, [&](size_t i) {
      a[i] = b[i] + alpha * c[i];
    },N/1000000);
    sink(checksum_sample(a));
  }
  double t_compute = t.sec();

  // 最小流量估算：读 b,c（16B）写 a（8B）=> 24B/elem/iter
  const double bytes = double(N) * 24.0 * double(iters);
  const double gb = bytes / 1e9;
  const double bw = gb / t_compute;

  std::cout << "Init time:    " << t_init << " s\n";
  std::cout << "Compute time: " << t_compute << " s\n";
  std::cout << "Bandwidth:    " << bw << " GB/s (24B/elem/iter)\n";
  std::cout << "Checksum:     " << checksum_sample(a) << " (ignore)\n";
  return 0;
}
