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

int main(int argc, char* argv[]) {
    // 默认参数
    size_t n = 1000000000UL;      // 10亿
    size_t num_buckets = 256;     // 256 桶
    size_t block_size = 256 * 1024; // 256K (1MB)
    DistType dist = DIST_UNIFORM;
    std::string dist_name = "uniform";

    if (argc > 1) n = std::stoull(argv[1]);
    if (argc > 2) num_buckets = std::stoull(argv[2]);
    if (argc > 3) block_size = std::stoull(argv[3]);
    if (argc > 4) {
        dist_name = argv[4];
        if (dist_name == "sorted") dist = DIST_SORTED;
        else if (dist_name == "reverse") dist = DIST_REVERSE;
    }

    std::cout << "Config: N=" << n << ", Buckets=" << num_buckets 
              << ", BlockSize=" << block_size << ", Dist=" << dist_name << std::endl;

    // ----------------------------------------------------------------
    // Setup Phase (Not Timed)
    // ----------------------------------------------------------------
    std::cout << "[Setup] Generating data..." << std::endl;
    auto input = generate_data(n, dist);
    
    // 预分配输出数组 (NUMA First Touch)
    parlay::sequence<T> output(n + 4096, 0); // Padding for safety

    // 生成 Pivots (均匀采样)
    // 注意：对于 Uniform 分布，均匀采样 Pivot 是合理的。
    // 对于 Sorted/Reverse 分布，我们直接在值域上均匀切分。
    parlay::sequence<T> pivots(num_buckets - 1);
    if (dist == DIST_UNIFORM) {
        // 对于随机数，我们在 unsigned 空间均匀切分
        size_t step = (size_t)(-1) / num_buckets; // unsigned max / buckets
        for(size_t i=0; i<num_buckets-1; ++i) {
            pivots[i] = (T)((i + 1) * step);
        }
    } else {
        // 对于 Sorted，我们在实际 Range 上切分
        long long min_val = std::numeric_limits<int>::min();
        long long range = n;
        double step = (double)range / num_buckets;
        for(size_t i=0; i<num_buckets-1; ++i) {
            pivots[i] = (T)(min_val + (long long)((i+1) * step));
        }
    }

    // Phase 1: Sort each Block (模拟 Sample Sort 的第一步)
    std::cout << "[Setup] Sorting blocks..." << std::endl;
    size_t num_blocks = (n + block_size - 1) / block_size;
    parlay::parallel_for(0, num_blocks, [&](size_t i) {
        size_t start = i * block_size;
        size_t end = std::min(start + block_size, n);
        std::sort(input.begin() + start, input.begin() + end);
    });

    // Phase 1.5: Oracle Counts (计算所有写指针)
    // 为了 benchmark scatter，我们需要知道每个 block 往每个 bucket 写到全局数组的哪个位置
    std::cout << "[Setup] Calculating offsets (Oracle)..." << std::endl;
    
    // 存储每个 Block 每个 Bucket 的 Count
    // 大小: num_blocks * num_buckets
    parlay::sequence<size_t> block_bucket_counts(num_blocks * num_buckets);
    
    parlay::parallel_for(0, num_blocks, [&](size_t i) {
        size_t start = i * block_size;
        size_t end = std::min(start + block_size, n);
        count_tile_simple(input.begin() + start, input.begin() + end, 
                          pivots.begin(), num_buckets, 
                          &block_bucket_counts[i * num_buckets]);
    });

    // 存储每个 Block 每个 Bucket 的 Write Offset (Transpose 后的全局偏移)
    parlay::sequence<T*> block_write_ptrs(num_blocks * num_buckets);
    
    // 计算前缀和 (Sequential for buckets, Parallel logic is complex, keep simple here)
    // 真正的 Sample Sort 会用 parallel scan，这里我们简单做
    // 按 Bucket 优先遍历 (Transpose)
    std::vector<size_t> global_bucket_tails(num_buckets, 0);
    
    // 这是一个串行的 Setup，可能会慢，但保证正确性
    // 对于 10亿数据，这个过程也就几秒
    for(size_t b=0; b<num_buckets; ++b) {
        size_t current_offset = 0;
        // 累加所有前面的桶的总大小 (前一个桶的结束位置)
        if (b > 0) current_offset = global_bucket_tails[b-1];
        
        for(size_t i=0; i<num_blocks; ++i) {
            block_write_ptrs[i * num_buckets + b] = output.begin() + current_offset;
            size_t count = block_bucket_counts[i * num_buckets + b];
            current_offset += count;
        }
        global_bucket_tails[b] = current_offset;
    }

    // ----------------------------------------------------------------
    // Benchmark Phase (Timed)
    // ----------------------------------------------------------------
    std::cout << "[Benchmark] Running Scatter Phase..." << std::endl;
    
    // 每个线程需要一个局部的指针缓存数组，避免 false sharing
    // 我们的 scatter 函数需要 OutIterator* bucket_ptrs
    // 我们在 parallel_for 内部从 block_write_ptrs拷贝过来
    
    auto start = std::chrono::high_resolution_clock::now();

    parlay::parallel_for(0, num_blocks, [&](size_t i) {
        size_t start = i * block_size;
        size_t end = std::min(start + block_size, n);
        
        // 准备当前 Block 的 input slice
        auto tile_slice = parlay::make_slice(input.begin() + start, input.begin() + end);
        auto pivots_slice = parlay::make_slice(pivots.begin(), pivots.end());
        
        // 准备当前 Block 的 output pointers
        // 这是一个放在栈上的小数组 (num_buckets * 8 bytes)
        // 假设 buckets <= 1024，栈空间足够 (8KB)
        // 如果 buckets 很大，需要换成 std::vector 或 heap alloc
        std::vector<T*> local_ptrs(num_buckets);
        for(size_t b=0; b<num_buckets; ++b) {
            local_ptrs[b] = block_write_ptrs[i * num_buckets + b];
        }

        // 调用核心函数
        scatter_tile_merge_galloping_aligned(
            tile_slice, 
            pivots_slice, 
            local_ptrs.data(), 
            num_buckets
        );
    });

    auto end = std::chrono::high_resolution_clock::now();
    std::chrono::duration<double> elapsed = end - start;

    double gb = (double)n * sizeof(T) / 1e9;
    double bw = gb / elapsed.count();

    std::cout << "RESULT: N=" << n << " Buckets=" << num_buckets 
              << " Time=" << elapsed.count() << "s" 
              << " Bandwidth=" << bw << " GB/s" << std::endl;

    return 0;
}