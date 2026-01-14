#include <algorithm>
#include <chrono>
#include <cstdint>
#include <cstring>
#include <iomanip>
#include <iostream>
#include <limits>
#include <random>
#include <string>
#include <type_traits>
#include <vector>

#include "x86simdsort-static-incl.h"

using Clock = std::chrono::steady_clock;

static inline double seconds_since(const Clock::time_point& t0, const Clock::time_point& t1) {
  return std::chrono::duration<double>(t1 - t0).count();
}

enum class DType { INT32, INT64, FLOAT, DOUBLE };

static bool parse_dtype(const std::string& s, DType& out) {
  if (s == "int32")  { out = DType::INT32;  return true; }
  if (s == "int64")  { out = DType::INT64;  return true; }
  if (s == "int63")  { out = DType::INT64;  return true; } // alias: accept user's "int63"
  if (s == "float")  { out = DType::FLOAT;  return true; }
  if (s == "double") { out = DType::DOUBLE; return true; }
  return false;
}

static const char* dtype_name(DType t) {
  switch (t) {
    case DType::INT32:  return "int32";
    case DType::INT64:  return "int64";
    case DType::FLOAT:  return "float";
    case DType::DOUBLE: return "double";
  }
  return "unknown";
}

struct Args {
  size_t n = 100000000;
  size_t min_block = 128;
  size_t max_block = 10000000;
  bool descending = false;
  bool check = false;
  uint64_t seed = 12345;
  DType dtype = DType::INT32;
};

static void usage(const char* prog) {
  std::cerr
    << "Usage: " << prog
    << " [--type int32|int64|int63|float|double]"
    << " [--n N] [--min_block B] [--max_block B]"
    << " [--descending 0|1] [--check 0|1] [--seed S]\n"
    << "Default: --type int32 --n 100000000 --min_block 128 --max_block 10000000"
    << " --descending 0 --check 0 --seed 12345\n";
}

static bool parse_bool(const std::string& s, bool& out) {
  if (s == "0" || s == "false" || s == "False") { out = false; return true; }
  if (s == "1" || s == "true"  || s == "True")  { out = true;  return true; }
  return false;
}

static bool parse_args(int argc, char** argv, Args& a) {
  for (int i = 1; i < argc; i++) {
    std::string k = argv[i];
    auto need_val = [&](const char* name) -> const char* {
      if (i + 1 >= argc) { std::cerr << "Missing value for " << name << "\n"; return nullptr; }
      return argv[++i];
    };

    if (k == "--type") {
      const char* v = need_val("--type"); if (!v) return false;
      DType dt;
      if (!parse_dtype(v, dt)) {
        std::cerr << "Bad --type: " << v << "\n";
        return false;
      }
      a.dtype = dt;

    } else if (k == "--n") {
      const char* v = need_val("--n"); if (!v) return false;
      a.n = std::stoull(v);

    } else if (k == "--min_block") {
      const char* v = need_val("--min_block"); if (!v) return false;
      a.min_block = std::stoull(v);

    } else if (k == "--max_block") {
      const char* v = need_val("--max_block"); if (!v) return false;
      a.max_block = std::stoull(v);

    } else if (k == "--descending") {
      const char* v = need_val("--descending"); if (!v) return false;
      std::string sv(v);
      if (!parse_bool(sv, a.descending)) { std::cerr << "Bad --descending: " << sv << "\n"; return false; }

    } else if (k == "--check") {
      const char* v = need_val("--check"); if (!v) return false;
      std::string sv(v);
      if (!parse_bool(sv, a.check)) { std::cerr << "Bad --check: " << sv << "\n"; return false; }

    } else if (k == "--seed") {
      const char* v = need_val("--seed"); if (!v) return false;
      a.seed = std::stoull(v);

    } else if (k == "--help" || k == "-h") {
      usage(argv[0]);
      return false;

    } else {
      std::cerr << "Unknown arg: " << k << "\n";
      usage(argv[0]);
      return false;
    }
  }
  return true;
}

template <class T>
static void fill_random(std::vector<T>& a, uint64_t seed) {
  std::mt19937_64 rng(seed);

  if constexpr (std::is_integral_v<T>) {
    // Use int64 distribution, cast down (works for int32/int64)
    std::uniform_int_distribution<long long> dist(
      (long long)std::numeric_limits<T>::min(),
      (long long)std::numeric_limits<T>::max()
    );
    for (auto& x : a) x = (T)dist(rng);

  } else {
    // Finite reals only: avoid NaN/Inf so hasnan=false is safe
    std::uniform_real_distribution<double> dist(-1e6, 1e6);
    for (auto& x : a) x = (T)dist(rng);
  }
}

template <class T>
static bool is_sorted_block(const T* p, size_t n, bool descending) {
  if (n <= 1) return true;
  if (!descending) {
    for (size_t i = 1; i < n; i++) if (p[i-1] > p[i]) return false;
  } else {
    for (size_t i = 1; i < n; i++) if (p[i-1] < p[i]) return false;
  }
  return true;
}

template <class T>
static int bench_blocks(const Args& args, const char* dt_name) {
  std::vector<T> master(args.n);
  std::vector<T> work(args.n);

  fill_random(master, args.seed);

  // CSV header (stdout)
  std::cout << "dtype,elem_size,block_size,num_blocks,effective_n,total_sec,avg_block_ns,throughput_Melem_s,throughput_GiB_s\n";
  std::cout << std::fixed << std::setprecision(6);

  // block sizes: min, *2, *2, ... plus max if needed
  std::vector<size_t> blocks;
  for (size_t b = args.min_block; b <= args.max_block; ) {
    blocks.push_back(b);
    if (b > (std::numeric_limits<size_t>::max() / 2)) break;
    size_t nb = b * 2;
    if (nb > args.max_block) break;
    b = nb;
  }
  if (blocks.empty() || blocks.back() != args.max_block) blocks.push_back(args.max_block);

  for (size_t block : blocks) {
    const size_t num_blocks = args.n / block;
    const size_t effective_n = num_blocks * block;
    if (num_blocks == 0) continue;

    // each block-size round uses identical input: copy from master
    std::memcpy(work.data(), master.data(), effective_n * sizeof(T));

    const auto t0 = Clock::now();
    for (size_t bi = 0; bi < num_blocks; bi++) {
      T* ptr = work.data() + bi * block;

      // hasnan=false: safe because our generator avoids NaN for float/double
      x86simdsortStatic::qsort(ptr, block, /*hasnan=*/false, /*descending=*/args.descending);

      if (args.check) {
        if ((bi & 1023ull) == 0ull) {
          if (!is_sorted_block(ptr, block, args.descending)) {
            std::cerr << "ERROR: block not sorted. dtype=" << dt_name
                      << " block_size=" << block << " bi=" << bi << "\n";
            return 3;
          }
        }
      }
    }
    const auto t1 = Clock::now();

    const double total_sec = seconds_since(t0, t1);
    const double avg_block_ns = (total_sec * 1e9) / (double)num_blocks;

    const double throughput_elem_s = (double)effective_n / total_sec;
    const double throughput_melem_s = throughput_elem_s / 1e6;

    const double bytes = (double)effective_n * (double)sizeof(T);
    const double throughput_gib_s = (bytes / total_sec) / (1024.0 * 1024.0 * 1024.0);

    std::cout << dt_name << ","
              << sizeof(T) << ","
              << block << ","
              << num_blocks << ","
              << effective_n << ","
              << total_sec << ","
              << avg_block_ns << ","
              << throughput_melem_s << ","
              << throughput_gib_s << "\n";

    std::cerr << "[OK] dtype=" << dt_name
              << " block=" << block
              << " num_blocks=" << num_blocks
              << " effective_n=" << effective_n
              << " total=" << total_sec << " s"
              << " avg=" << avg_block_ns << " ns/block"
              << " thr=" << throughput_melem_s << " Melem/s"
              << " (" << throughput_gib_s << " GiB/s)\n";
  }

  return 0;
}

int main(int argc, char** argv) {
  Args args;
  if (!parse_args(argc, argv, args)) return 1;

  if (args.min_block < 1 || args.max_block < args.min_block) {
    std::cerr << "Invalid block range.\n";
    return 2;
  }

  const char* dt_name = dtype_name(args.dtype);

  switch (args.dtype) {
    case DType::INT32:  return bench_blocks<int32_t>(args, dt_name);
    case DType::INT64:  return bench_blocks<int64_t>(args, dt_name);
    case DType::FLOAT:  return bench_blocks<float>(args, dt_name);
    case DType::DOUBLE: return bench_blocks<double>(args, dt_name);
  }

  return 0;
}
