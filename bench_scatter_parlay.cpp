#include <iostream>
#include <vector>
#include <algorithm>
#include <chrono>
#include <random>
#include <cstring>
#include <immintrin.h>
#include <limits>

// ParlayLib Headers
#include "parlay/parallel.h"
#include "parlay/sequence.h"
#include "parlay/primitives.h"
#include "parlay/slice.h"
#include "parlay/utilities.h"

#include "perf_ctl.hpp"
#include "flush_range_clflushopt.h"


// 引入你的核心 Scatter 函数 (为了保持独立性，我直接粘贴在这里，
// 实际工程中你可以 include 你的头文件)
// ============================================================
// 你的核心 Scatter 实现
// ============================================================
template <typename InIterator, typename PivotIterator, typename OutIterator>
void scatter_tile_merge_galloping_aligned(parlay::slice<InIterator, InIterator> tile,
                                          parlay::slice<PivotIterator, PivotIterator> pivots,
                                          OutIterator* bucket_ptrs,
                                          size_t num_buckets) {
    if (tile.size() == 0) return;

    const int* curr_ptr = &(*tile.begin());
    const int* end_ptr = curr_ptr + tile.size();
    auto pit = pivots.begin();
    auto pend = pivots.end();

    int* dest_ptr = (int*)bucket_ptrs[0];

    for (size_t b = 0; b < num_buckets; ++b) {
        int p_val = (pit != pend) ? *pit : std::numeric_limits<int>::max();
        __m512i v_pivot = _mm512_set1_epi32(p_val);

        while (curr_ptr + 16 <= end_ptr) {
            _mm_prefetch((const char*)(curr_ptr + 128), _MM_HINT_T0);
            __m512i v_data = _mm512_loadu_si512(curr_ptr);
            __mmask16 mask = _mm512_cmplt_epi32_mask(v_data, v_pivot);

            if (mask == 0xFFFF) {
                uintptr_t addr = (uintptr_t)dest_ptr;
                if ((addr & 63) == 0) {
                    _mm512_stream_si512((__m512i*)dest_ptr, v_data);
                    dest_ptr += 16;
                    curr_ptr += 16;
                } else {
                    if (curr_ptr + 32 > end_ptr) {
                        _mm512_storeu_si512(dest_ptr, v_data);
                        dest_ptr += 16;
                        curr_ptr += 16;
                        continue;
                    }
                    int misalignment_bytes = addr & 63;
                    int elems_to_align = (64 - misalignment_bytes) >> 2;
                    __mmask16 align_mask = (1 << elems_to_align) - 1;
                    _mm512_mask_storeu_epi32(dest_ptr, align_mask, v_data);
                    dest_ptr += elems_to_align;
                    curr_ptr += elems_to_align;

                    while (curr_ptr + 16 <= end_ptr) {
                        _mm_prefetch((const char*)(curr_ptr + 128), _MM_HINT_T0);
                        __m512i v_inner = _mm512_loadu_si512(curr_ptr);
                        __mmask16 m_inner = _mm512_cmplt_epi32_mask(v_inner, v_pivot);
                        if (m_inner == 0xFFFF) {
                            _mm512_stream_si512((__m512i*)dest_ptr, v_inner);
                            dest_ptr += 16;
                            curr_ptr += 16;
                        } else {
                            break; 
                        }
                    }
                }
            } else {
                int k = _mm_popcnt_u32(mask);
                if (k > 0) {
                    _mm512_mask_storeu_epi32(dest_ptr, mask, v_data);
                    dest_ptr += k;
                    curr_ptr += k;
                }
                bucket_ptrs[b] = (OutIterator)dest_ptr;
                goto NEXT_BUCKET;
            }
        }

        if (curr_ptr < end_ptr) {
            int rem = end_ptr - curr_ptr;
            __mmask16 load_mask = (1 << rem) - 1;
            __m512i v_data = _mm512_mask_loadu_epi32(_mm512_setzero_si512(), load_mask, curr_ptr);
            __mmask16 cmp_mask = _mm512_cmplt_epi32_mask(v_data, v_pivot);
            cmp_mask &= load_mask;

            if (cmp_mask == load_mask) {
                _mm512_mask_storeu_epi32(dest_ptr, load_mask, v_data);
                dest_ptr += rem;
                curr_ptr += rem;
                bucket_ptrs[b] = (OutIterator)dest_ptr;
                return;
            } else {
                int k = _mm_popcnt_u32(cmp_mask);
                if (k > 0) {
                    _mm512_mask_storeu_epi32(dest_ptr, cmp_mask, v_data);
                    dest_ptr += k;
                    curr_ptr += k;
                }
                bucket_ptrs[b] = (OutIterator)dest_ptr;
            }
        } else {
            bucket_ptrs[b] = (OutIterator)dest_ptr;
            return;
        }

        NEXT_BUCKET:
        if (pit != pend) ++pit;
        if (b + 1 < num_buckets) dest_ptr = (int*)bucket_ptrs[b+1];
    }
    
    // Last bucket handling
    if (curr_ptr < end_ptr) {
         uintptr_t addr = (uintptr_t)dest_ptr;
         if ((addr & 63) != 0 && curr_ptr + 32 <= end_ptr) {
             int misalignment_bytes = addr & 63;
             int elems_to_align = (64 - misalignment_bytes) >> 2;
             __mmask16 align_mask = (1 << elems_to_align) - 1;
             __m512i v_head = _mm512_loadu_si512(curr_ptr);
             _mm512_mask_storeu_epi32(dest_ptr, align_mask, v_head);
             dest_ptr += elems_to_align;
             curr_ptr += elems_to_align;
         }
         while (curr_ptr + 16 <= end_ptr) {
             _mm512_stream_si512((__m512i*)dest_ptr, _mm512_loadu_si512(curr_ptr));
             dest_ptr += 16;
             curr_ptr += 16;
         }
         if (curr_ptr < end_ptr) {
             int rem = end_ptr - curr_ptr;
             __mmask16 mask = (1 << rem) - 1;
             _mm512_mask_storeu_epi32(dest_ptr, mask, _mm512_mask_loadu_epi32(_mm512_setzero_si512(), mask, curr_ptr));
             dest_ptr += rem;
         }
         bucket_ptrs[num_buckets - 1] = (OutIterator)dest_ptr;
    }
}
template <typename InIterator, typename PivotIterator, typename OutIterator>
void scatter_tile_merge_std_move(parlay::slice<InIterator, InIterator> tile,
                                 parlay::slice<PivotIterator, PivotIterator> pivots,
                                 OutIterator* bucket_ptrs,
                                 size_t num_buckets) {
    if (tile.size() == 0) return;

    auto curr_it = tile.begin();
    auto end_it = tile.end();
    auto pit = pivots.begin();
    auto pend = pivots.end();

    // 当前写入的 bucket 指针
    // 假设 OutIterator 是 T*，我们统一转成 T* 操作
    using T = typename std::iterator_traits<InIterator>::value_type;
    T* dest_ptr = (T*)bucket_ptrs[0];

    // 遍历每一个 Pivot (即每一个 Bucket)
    for (size_t b = 0; b < num_buckets; ++b) {
        // 获取当前 Pivot 值，如果是最后一个 bucket，pivot 设为最大值
        T p_val = (pit != pend) ? *pit : std::numeric_limits<T>::max();

        // 核心逻辑：Galloping / 线性扫描
        // 只要当前元素 < pivot，就属于当前 bucket
        while (curr_it != end_it) {
            T val = *curr_it;
            if (val < p_val) {
                // 1. 写入数据 (std::move)
                // 编译器通常会生成普通 Store (mov / vmovdqu)，会污染 Cache
                *dest_ptr = std::move(val); 
                
                // 2. 指针推进
                ++dest_ptr;
                ++curr_it;
            } else {
                // 当前元素 >= pivot，说明当前 bucket 已填完
                // 跳出内层循环，进入下一个 bucket
                break;
            }
        }

        // 更新当前 bucket 的尾指针
        bucket_ptrs[b] = (OutIterator)dest_ptr;

        // 准备处理下一个 bucket
        if (pit != pend) ++pit;
        if (b + 1 < num_buckets) {
            // 切换到下一个 bucket 的写入位置
            dest_ptr = (T*)bucket_ptrs[b + 1];
        }
    }

    // 处理剩余元素 (Last Bucket logic)
    // 实际上上面的循环中，如果 b == num_buckets - 1，p_val 是 max，
    // 应该已经把剩余所有元素都收进去了。
    // 但为了逻辑严谨性（防止 pivot 没覆盖完），可以保留这个尾部检查
    while (curr_it != end_it) {
        *dest_ptr = std::move(*curr_it);
        ++dest_ptr;
        ++curr_it;
    }
    // 更新最后一个 bucket 的指针（如果是通过循环退出的，可能已经更过了，这里确保一下）
    if (num_buckets > 0) {
        bucket_ptrs[num_buckets - 1] = (OutIterator)dest_ptr;
    }
}
// ============================================================
// Benchmarking Infrastructure
// ============================================================

using T = int;

// 1. 数据生成器
enum DistType { DIST_UNIFORM, DIST_SORTED, DIST_REVERSE };

parlay::sequence<T> generate_data(size_t n, DistType type) {
    if (type == DIST_SORTED) {
        // 从 INT_MIN 开始 ++
        return parlay::tabulate(n, [](size_t i) {
            return (T)((long long)i + std::numeric_limits<int>::min());
        });
    } else if (type == DIST_REVERSE) {
        return parlay::tabulate(n, [n](size_t i) {
            return (T)((long long)(n - 1 - i) + std::numeric_limits<int>::min());
        });
    } else {
        // Uniform Random
        // 使用 Parlay 自带的哈希生成伪随机数，速度快且并行安全
        return parlay::tabulate(n, [](size_t i) {
            return (T)parlay::hash64(i);
        });
    }
}

// 2. 简易的计数器 (用于 Setup 阶段)
// 统计一个 Tile 里每个桶有多少元素，为了计算写偏移量
void count_tile_simple(const T* begin, const T* end, 
                       const T* pivots, size_t num_buckets, 
                       size_t* counts) {
    // 初始化
    for(size_t i=0; i<num_buckets; ++i) counts[i] = 0;
    
    // 因为 Tile 已经排序，我们可以直接二分查找或者线性扫描 Pivots
    // 这里用简单的线性扫描，因为只跑一次
    const T* curr = begin;
    for(size_t b=0; b<num_buckets-1; ++b) {
        T p = pivots[b];
        while(curr < end && *curr < p) {
            counts[b]++;
            curr++;
        }
    }
    // 剩下的给最后一个桶
    counts[num_buckets-1] = (end - curr);
}
// 全局 Flush Buffer
// 使用 parlay::sequence 方便并行操作，但需要注意它可能不保证 64 字节对齐
// 这里我们主要为了挤占 Cache，对齐不是 flush 的硬性要求
parlay::sequence<char> g_dummy_buffer;
const size_t FLUSH_SIZE_BYTES = 2UL * 1024 * 1024 * 1024; 
// ==========================================
// 辅助函数：Flush Cache
// ==========================================
void init_flush_buffer() {
    std::cout << "[Init] Allocating " << FLUSH_SIZE_BYTES / 1024 / 1024 << " MB for cache flushing..." << std::endl;
    g_dummy_buffer = parlay::sequence<char>(FLUSH_SIZE_BYTES, 0);
}

void flush_cache() {
    // 使用并行写入，强制所有核心参与 Cache 驱逐
    // 写入操作比读取更能有效地触发 Cache 替换策略 (RFO)
    parlay::parallel_for(0, g_dummy_buffer.size() / 64, [&](size_t i) {
        // 写入每个 Cache Line 的第一个字节
        // volatile 阻止编译器优化掉“无用”的写入
        volatile char* ptr = &g_dummy_buffer[i * 64];
        *ptr = (char)i;
    });
    // 内存屏障，防止乱序执行
    std::atomic_thread_fence(std::memory_order_seq_cst);
}
void flush_cache_readonly() {
  const size_t nlines = g_dummy_buffer.size() / 64;

  const size_t P = (size_t)parlay::num_workers();
  std::vector<uint64_t> tls(P, 0);

  parlay::parallel_for(0, nlines, [&](size_t i) {
    size_t wid = (size_t)parlay::worker_id();
    // 读每条 cache line 的一个字节，累加到线程局部，防止被优化掉
    tls[wid] += (unsigned char)g_dummy_buffer[i * 64];
  });

  // 防止编译器把 tls 优化掉
  uint64_t sink = 0;
  for (auto v : tls) sink ^= v;
  asm volatile("" :: "r"(sink) : "memory");
}


// ------------------------------------------------------------
// Helpers for correct int32 pivots covering full signed range
// ------------------------------------------------------------
static inline uint32_t to_u32_ordered(int32_t x) {
  return (uint32_t)x ^ 0x80000000u;
}
static inline int32_t to_i32_ordered(uint32_t u) {
  return (int32_t)(u ^ 0x80000000u);
}

// ---- You already have these somewhere ----
// using T = int; (or int32_t)
// enum DistType { DIST_UNIFORM, DIST_SORTED, DIST_REVERSE };
// parlay::sequence<T> generate_data(size_t n, DistType dist);
// void init_flush_buffer();
// void flush_cache();
// void count_tile_simple(T* begin, T* end, T* pivots, size_t num_buckets, size_t* out_counts);
// template <typename InIterator, typename PivotIterator, typename OutIterator>
// void scatter_tile_merge_galloping_aligned(parlay::slice<InIterator, InIterator> tile,
//                                           parlay::slice<PivotIterator, PivotIterator> pivots,
//                                           OutIterator* bucket_ptrs,
//                                           size_t num_buckets);

int main(int argc, char* argv[]) {
  // 默认参数
  size_t n = 1000000000UL;         // 10亿
  size_t num_buckets = 256;        // 256 桶
  size_t block_size = 256 * 1024;  // 256K elements (对 int32 是 1MB)
  DistType dist = DIST_UNIFORM;
  std::string dist_name = "uniform";
  printf("threads number: %d\n", parlay::num_workers());

  if (argc > 1) n = std::stoull(argv[1]);
  if (argc > 2) num_buckets = std::stoull(argv[2]);
  if (argc > 3) block_size = std::stoull(argv[3]);
  if (argc > 4) {
    dist_name = argv[4];
    if (dist_name == "sorted") dist = DIST_SORTED;
    else if (dist_name == "reverse") dist = DIST_REVERSE;
  }

  std::cout << "Config: N=" << n << ", Buckets=" << num_buckets
            << ", BlockSize=" << block_size << ", Dist=" << dist_name << "\n";

  // ----------------------------------------------------------------
  // Setup Phase (Not Timed)
  // ----------------------------------------------------------------
  std::cout << "[Setup] Generating data...\n";
  auto input = generate_data(n, dist);

  init_flush_buffer();

  // 预分配输出数组
  parlay::sequence<T> output(n + 4096, 0);

  // 强制 first-touch（计时外）：并行 touch output，避免 NUMA 偏置
  std::cout << "[Setup] First-touch output...\n";
  parlay::parallel_for(0, output.size(), [&](size_t i) {
    output[i] = 0;
  }, 4096);

  // 生成 pivots
  parlay::sequence<T> pivots(num_buckets - 1);
  if (dist == DIST_UNIFORM) {
    // 让 pivots 覆盖 int32 全域且单调：在 ordered uint32 空间均匀切分，再映射回 int32
    // step = 2^32 / num_buckets
    uint64_t step = (uint64_t(1) << 32) / (uint64_t)num_buckets;
    for (size_t i = 0; i < num_buckets - 1; ++i) {
      uint32_t u = (uint32_t)((uint64_t)(i + 1) * step);
      int32_t p = to_i32_ordered(u);
      pivots[i] = (T)p;
    }
  } else {
    // sorted/reverse：你原逻辑（按值域/范围切分）
    long long min_val = std::numeric_limits<int>::min();
    long long range = (long long)n;
    double step = (double)range / (double)num_buckets;
    for (size_t i = 0; i < num_buckets - 1; ++i) {
      pivots[i] = (T)(min_val + (long long)((i + 1) * step));
    }
  }

  // Phase 1: Sort each Block
  std::cout << "[Setup] Sorting blocks...\n";
  size_t num_blocks = (n + block_size - 1) / block_size;
  parlay::parallel_for(0, num_blocks, [&](size_t i) {
    size_t s = i * block_size;
    size_t e = std::min(s + block_size, n);
    std::sort(input.begin() + s, input.begin() + e);
  });

  // Phase 1.5: Oracle Counts
  std::cout << "[Setup] Calculating offsets (Oracle)...\n";

  parlay::sequence<size_t> block_bucket_counts(num_blocks * num_buckets);

  parlay::parallel_for(0, num_blocks, [&](size_t i) {
    size_t s = i * block_size;
    size_t e = std::min(s + block_size, n);
    count_tile_simple(input.begin() + s, input.begin() + e,
                      pivots.begin(), num_buckets,
                      &block_bucket_counts[i * num_buckets]);
  });

  parlay::sequence<T*> block_write_ptrs(num_blocks * num_buckets);

  std::vector<size_t> global_bucket_tails(num_buckets, 0);
  for (size_t b = 0; b < num_buckets; ++b) {
    size_t current_offset = (b > 0) ? global_bucket_tails[b - 1] : 0;
    for (size_t i = 0; i < num_blocks; ++i) {
      block_write_ptrs[i * num_buckets + b] = output.begin() + current_offset;
      size_t count = block_bucket_counts[i * num_buckets + b];
      current_offset += count;
    }
    global_bucket_tails[b] = current_offset;
  }

  // ----------------------------------------------------------------
  // Benchmark Phase (Timed)
  // ----------------------------------------------------------------
  std::cout << "[Benchmark] Running Scatter Phase...\n";

  // 每个 worker 一份 local_ptrs（计时外分配，ROI 内只 memcpy）
  const size_t P = (size_t)parlay::num_workers();
  std::vector<std::vector<T*>> tls_ptrs(P, std::vector<T*>(num_buckets));

  // cache flush/evict（计时外，且在 perf.enable 前）
  flush_cache_readonly();

  PerfFifoController perf; // 若没有设置 env，则 inactive

  if (perf.active()) perf.enable();
  auto t0 = std::chrono::high_resolution_clock::now();

  parlay::parallel_for(0, num_blocks, [&](size_t i) {
    size_t s = i * block_size;
    size_t e = std::min(s + block_size, n);

    auto tile_slice   = parlay::make_slice(input.begin() + s, input.begin() + e);
    auto pivots_slice = parlay::make_slice(pivots.begin(), pivots.end());

    auto& local_ptrs = tls_ptrs[(size_t)parlay::worker_id()];
    std::memcpy(local_ptrs.data(),
                &block_write_ptrs[i * num_buckets],
                num_buckets * sizeof(T*));

    // scatter_tile_merge_galloping_aligned(
    //   tile_slice,
    //   pivots_slice,
    //   local_ptrs.data(),
    //   num_buckets
    // );
    scatter_tile_merge_std_move(
      tile_slice,
      pivots_slice,
      local_ptrs.data(),
      num_buckets
    );
  });

auto t1 = std::chrono::high_resolution_clock::now();
  if (perf.active()) perf.disable();

  std::chrono::duration<double> elapsed = t1 - t0;

  // 1. 基础数据
  double gb_oneway = (double)n * sizeof(T) / 1e9;
  double gb_rw     = 2.0 * gb_oneway; // Scatter 也是读 Input 写 Output

  // 2. 打印 Key-Value 供 Python 解析
  // 注意：确保 T 在此前已定义，通常 scatter 是 int 或 uint32_t
  std::cout << "TYPE_BYTES " << sizeof(T) << "\n"; 
  std::cout << "ROI_SECONDS " << elapsed.count() << "\n";
  
  // 统一命名为 GOODPUT，方便脚本解析
  std::cout << "GOODPUT_ONEWAY_GBps " << (gb_oneway / elapsed.count()) << "\n";
  std::cout << "GOODPUT_RW_GBps "     << (gb_rw     / elapsed.count()) << "\n";

  // 3. 人类可读的结尾（脚本会忽略这行，因为没有 Key Match）
  std::cout << "RESULT: N=" << n << " Buckets=" << num_buckets 
            << " Time=" << elapsed.count() << "s\n";

  return 0;
}