#include <iostream>
#include <vector>
#include <chrono>
#include <algorithm>
#include <cstdint>
#include <cstring>
#include <immintrin.h>
#include <numeric>
#include <iomanip>
#include <atomic>
#include <sys/mman.h>
// Parlay headers
#include "parlay/parallel.h"
#include "parlay/sequence.h"
#include "parlay/utilities.h"
#include <stdlib.h> // for posix_memalign, free

#include "perf_ctl.hpp"
#include "flush_range_clflushopt.h"

template<typename T>
struct AlignedArray {
    T* ptr;
    size_t n;

    AlignedArray(size_t size) : n(size) {
        // 1. 内存分配：使用 2MB 对齐 (Huge Page 友好)
        // 2MB = 2 * 1024 * 1024 = 2097152
        size_t alignment = 2 * 1024 * 1024; 
        
        // 注意：分配大小最好也是 2MB 的倍数，或者足够大
        if (posix_memalign((void**)&ptr, alignment, n * sizeof(T)) != 0) {
            throw std::runtime_error("Aligned allocation failed");
        }

        // ============================================================
        // [修改] 2. 告诉内核：请务必使用大页 (THP)
        // ============================================================
        int ret = madvise(ptr, n * sizeof(T), MADV_HUGEPAGE);
        if (ret != 0) {
            perror("[Warning] madvise(MADV_HUGEPAGE) failed");
        } else {
             // std::cout << "[System] THP explicitly enabled." << std::endl;
        }
    }

    ~AlignedArray() {
        free(ptr);
    }

    auto as_slice() {
        return parlay::make_slice(ptr, ptr + n);
    }
    
    T* begin() { return ptr; }
    T* end() { return ptr + n; }
    T& operator[](size_t i) { return ptr[i]; }
};
// ==========================================
// 全局设置
// ==========================================
using T = uint32_t;
// 设置 Flush Buffer 大小为 2GB，确保远超任何 L3 Cache
const size_t FLUSH_SIZE_BYTES = 2UL * 1024 * 1024 * 1024; 
// 对齐要求 (AVX-512 Stream Store 需要 64字节对齐)
const size_t ALIGNMENT = 64;

// 全局 Flush Buffer
// 使用 parlay::sequence 方便并行操作，但需要注意它可能不保证 64 字节对齐
// 这里我们主要为了挤占 Cache，对齐不是 flush 的硬性要求
parlay::sequence<char> g_dummy_buffer;

// ==========================================
// 辅助函数：Flush Cache
// ==========================================
void init_flush_buffer() {
    std::cout << "[Init] Allocating " << FLUSH_SIZE_BYTES / 1024 / 1024 << " MB for cache flushing..." << std::endl;
    g_dummy_buffer = parlay::sequence<char>(FLUSH_SIZE_BYTES, 0);
}


void flush_cache_readonly() {
  const size_t nlines = g_dummy_buffer.size() / 64;
  const size_t P = (size_t)parlay::num_workers();
  std::vector<uint64_t> tls(P, 0);

  parlay::parallel_for(0, nlines, [&](size_t i) {
    size_t wid = (size_t)parlay::worker_id();
    tls[wid] += (unsigned char)g_dummy_buffer[i * 64];
  });

  uint64_t sink = 0;
  for (auto v : tls) sink ^= v;
  asm volatile("" :: "r"(sink) : "memory");
}


// ==========================================
// Kernels
// ==========================================

// 1. Standard Copy
void kernel_std_copy(const T* src, T* dest, size_t size) {
    std::copy(src, src + size, dest);
}

// 2. Standard Move (测试 int 类型 move 是否等同于 copy)
void kernel_std_move(const T* src, T* dest, size_t size) {
    std::move(src, src + size, dest);
}

// 3. AVX2 (256-bit) Normal Store
void kernel_avx256_copy(const T* src, T* dest, size_t size) {
    size_t i = 0;
    for (; i + 8 <= size; i += 8) {
        __m256i val = _mm256_loadu_si256((const __m256i*)(src + i));
        _mm256_storeu_si256((__m256i*)(dest + i), val);
    }
    for (; i < size; ++i) dest[i] = src[i];
}

// 4. AVX512 (512-bit) Normal Store
void kernel_avx512_copy(const T* src, T* dest, size_t size) {
    size_t i = 0;
    for (; i + 16 <= size; i += 16) {
        __m512i val = _mm512_loadu_si512((const void*)(src + i));
        _mm512_storeu_si512((void*)(dest + i), val);
    }
    if (i < size) {
        __mmask16 mask = (1 << (size - i)) - 1;
        __m512i val = _mm512_mask_loadu_epi32(_mm512_setzero_si512(), mask, (const void*)(src + i));
        _mm512_mask_storeu_epi32((void*)(dest + i), mask, val);
    }
}

// ==========================================
// 5. AVX512 Stream Store (NT Store) - Fully Optimized
// ==========================================
void kernel_avx512_stream(const T* src, T* dest, size_t size) {
    if (size == 0) return;

    size_t i = 0;
    
    // ---------------------------------------------------------
    // 1. Head (Peeling): 强制对齐 dest 到 64 字节边界
    // ---------------------------------------------------------
    uintptr_t dest_addr = (uintptr_t)dest;
    size_t misalignment_bytes = dest_addr & 63; // 等价于 % 64

    if (misalignment_bytes != 0) {
        // 计算需要补充多少字节才能对齐
        size_t bytes_to_align = 64 - misalignment_bytes;
        // 转换为 int (T) 的个数 (bytes / 4)
        size_t elems_to_align = bytes_to_align >> 2; 

        // 边界检查：如果总数据量比对齐所需的还少，直接跳到 Tail 处理
        size_t head_count = std::min(size, elems_to_align);

        // 生成 Mask (例如需要 3 个元素，Mask = 000...0111)
        __mmask16 mask = (1 << head_count) - 1;

        // 加载 + Masked Store
        __m512i val = _mm512_mask_loadu_epi32(_mm512_setzero_si512(), mask, (const void*)src);
        _mm512_mask_storeu_epi32((void*)dest, mask, val);

        // 更新指针和计数
        i += head_count;
        src += head_count;
        dest += head_count;
        size -= head_count; // 剩余需要处理的大小
    }

    // 此时 dest 绝对是 64 字节对齐的！(除非 size 太小在上面就处理完了)

    // ---------------------------------------------------------
    // 2. Body: 对齐后的 Stream Store 高速公路
    // ---------------------------------------------------------
    // 每次处理 16 个元素
    size_t body_loops = size / 16;
    
    // 简单的循环展开提示 (编译器通常能自己做，但显式写也不错)
    // 注意：src 依然可能是不对齐的，但 loadu 对 src 不敏感，效率依然极高。
    // 关键是 dest 对齐了，stream 指令就不会崩，也不会退化。
    for (size_t k = 0; k < body_loops; ++k) {
        __m512i val = _mm512_loadu_si512((const void*)src);
        // 这里必须用 si512，地址必须对齐 (我们已经在 Head 处理好了)
        _mm512_stream_si512((__m512i*)dest, val);
        
        src += 16;
        dest += 16;
    }
    
    // 更新剩余数量
    size_t processed_in_body = body_loops * 16;
    size -= processed_in_body;

    // ---------------------------------------------------------
    // 3. Tail: 处理剩余不足 16 个的元素
    // ---------------------------------------------------------
    if (size > 0) {
        __mmask16 mask = (1 << size) - 1;
        __m512i val = _mm512_mask_loadu_epi32(_mm512_setzero_si512(), mask, (const void*)src);
        _mm512_mask_storeu_epi32((void*)dest, mask, val);
    }
}
// ==========================================
// 5. AVX512 Stream Store + Software Prefetch + Unrolling
// [Fixed]: Signature matches unsigned int*
// ==========================================
void kernel_avx512_stream_prefetch(const unsigned int* src, unsigned int* dest, size_t size) {
    if (size == 0) return;

    // ---------------------------------------------------------
    // 1. Head (Peeling): 强制对齐 dest 到 64 字节边界
    // ---------------------------------------------------------
    uintptr_t dest_addr = (uintptr_t)dest;
    size_t misalignment_bytes = dest_addr & 63; 

    if (misalignment_bytes != 0) {
        size_t bytes_to_align = 64 - misalignment_bytes;
        size_t elems_to_align = bytes_to_align >> 2; 
        size_t head_count = std::min(size, elems_to_align);
        
        __mmask16 mask = (1 << head_count) - 1;
        // 注意：这里的 cast 转为 void* 是安全的，AVX 不在乎有符号无符号
        __m512i val = _mm512_mask_loadu_epi32(_mm512_setzero_si512(), mask, (const void*)src);
        _mm512_mask_storeu_epi32((void*)dest, mask, val);

        src += head_count;
        dest += head_count;
        size -= head_count;
    }

    // ---------------------------------------------------------
    // 2. Body: Unrolled Loop with Prefetching
    // ---------------------------------------------------------
    size_t unroll_size = 64;
    size_t body_loops = size / unroll_size;

    const int PREFETCH_DIST = 256; 
    const char* p_src = (const char*)src; // 用于预取的指针

    for (size_t k = 0; k < body_loops; ++k) {
        // [关键] 预取下一块数据
        _mm_prefetch(p_src + PREFETCH_DIST + 0,   _MM_HINT_T0);
        _mm_prefetch(p_src + PREFETCH_DIST + 64,  _MM_HINT_T0);
        _mm_prefetch(p_src + PREFETCH_DIST + 128, _MM_HINT_T0);
        _mm_prefetch(p_src + PREFETCH_DIST + 192, _MM_HINT_T0);

        // [执行] Load + Stream Store
        __m512i v0 = _mm512_loadu_si512((const void*)(src + 0));
        __m512i v1 = _mm512_loadu_si512((const void*)(src + 16));
        __m512i v2 = _mm512_loadu_si512((const void*)(src + 32));
        __m512i v3 = _mm512_loadu_si512((const void*)(src + 48));

        _mm512_stream_si512((__m512i*)(dest + 0), v0);
        _mm512_stream_si512((__m512i*)(dest + 16), v1);
        _mm512_stream_si512((__m512i*)(dest + 32), v2);
        _mm512_stream_si512((__m512i*)(dest + 48), v3);

        src  += 64;
        dest += 64;
        p_src += 256; 
    }

    size -= (body_loops * 64);

    // ---------------------------------------------------------
    // 3. Cleanup Body
    // ---------------------------------------------------------
    size_t remaining_blocks = size / 16;
    for (size_t k = 0; k < remaining_blocks; ++k) {
        __m512i val = _mm512_loadu_si512((const void*)src);
        _mm512_stream_si512((__m512i*)dest, val);
        src += 16;
        dest += 16;
    }
    size -= (remaining_blocks * 16);

    // ---------------------------------------------------------
    // 4. Tail
    // ---------------------------------------------------------
    if (size > 0) {
        __mmask16 mask = (1 << size) - 1;
        __m512i val = _mm512_mask_loadu_epi32(_mm512_setzero_si512(), mask, (const void*)src);
        _mm512_mask_storeu_epi32((void*)dest, mask, val);
    }
}


// 你已有：kernel(src, dst, size)
template<typename Range, typename Func>
void benchmark_routine_amp(
    const std::string& name,
    size_t n,
    size_t num_blocks,
    size_t block_size,
    Range& source,
    Range& dest,
    Func kernel)
{
  PerfFifoController perf;

  // Warmup（不计入 ROI）
//   flush_cache_readonly();
//   parlay::parallel_for(0, num_blocks, [&](size_t i) {
//     size_t s = i * block_size;
//     size_t e = std::min(s + block_size, n);
//     kernel(source.begin() + s, dest.begin() + s, e - s);
//   });

  // ROI：只做一次，交给 Python repeats
//   flush_cache_readonly();
flush_range_clflushopt((void*)source.begin(), n * sizeof(T));
flush_range_clflushopt((void*)dest.begin(),   n * sizeof(T));
std::this_thread::sleep_for(std::chrono::milliseconds(200));
// 2. [新增] 强制锁定 source 为只读
// 注意：source.begin() 必须是 4KB 对齐的 (你的 AlignedArray 已经是了)
// int ret = mprotect((void*)source.begin(), n * sizeof(T), PROT_READ);
// if (ret != 0) {
//     perror("mprotect failed");
//     exit(1);
// }
// std::cout << "[System] Source array marked as PROT_READ (Read-Only)." << std::endl;

  if (perf.active()) perf.enable();
  auto t0 = std::chrono::high_resolution_clock::now();

  parlay::parallel_for(0, num_blocks, [&](size_t i) {
    size_t s = i * block_size;
    size_t e = std::min(s + block_size, n);
    kernel(source.begin() + s, dest.begin() + s, e - s);
  });
// std::this_thread::sleep_for(std::chrono::milliseconds(100));

  auto t1 = std::chrono::high_resolution_clock::now();
  if (perf.active()) perf.disable();

  double roi_s = std::chrono::duration<double>(t1 - t0).count();

  // goodput（按你的统一口径）
  const double elem_bytes = (double)sizeof(typename std::remove_reference<decltype(*source.begin())>::type);
  const double bytes_oneway = (double)n * elem_bytes;
  const double bytes_rw = 2.0 * bytes_oneway;

  const double good_oneway_gbps = bytes_oneway / roi_s / 1e9;
  const double good_rw_gbps     = bytes_rw     / roi_s / 1e9;

  // 固定输出：Python 解析就靠这些行
  std::cout << "KERNEL " << name << "\n";
  std::cout << "TYPE_BYTES " << (size_t)elem_bytes << "\n";
  std::cout << "N " << n << "\n";
  std::cout << "BLOCK_SIZE " << block_size << "\n";
  std::cout << "ROI_SECONDS " << std::setprecision(9) << roi_s << "\n";
  std::cout << "GOODPUT_ONEWAY_GBps " << std::setprecision(6) << good_oneway_gbps << "\n";
  std::cout << "GOODPUT_RW_GBps "     << std::setprecision(6) << good_rw_gbps << "\n";
}
// ==========================================
// 6. AVX512 Stream Store - Blocked / Tiled Version
// ==========================================
// 模拟 Scatter 的行为：将大任务切成 L2 Cache 友好的小块 (e.g., 256KB)
// 理论上这能减少 TLB Miss 并提高 Page Hit Rate
void kernel_avx512_stream_blocked(const unsigned int* src, unsigned int* dest, size_t size) {
    // 设定 Block 大小为 256KB (64K ints)
    // 这是你之前测试中性能最好的尺寸
    const size_t TILE_SIZE = 64 * 1024; 

    size_t offset = 0;
    while (offset < size) {
        // 计算当前 Tile 的大小
        size_t current_chunk = std::min(TILE_SIZE, size - offset);
        
        const unsigned int* s_ptr = src + offset;
        unsigned int* d_ptr = dest + offset;

        // --- 内部就是一个标准的 Stream Loop ---
        
        // 1. Align Destination
        uintptr_t dest_addr = (uintptr_t)d_ptr;
        size_t misalignment = (dest_addr & 63) >> 2;
        if (misalignment != 0) {
            size_t head = std::min(current_chunk, (size_t)(16 - misalignment));
            __mmask16 mask = (1 << head) - 1;
            __m512i v = _mm512_mask_loadu_epi32(_mm512_setzero_si512(), mask, (const void*)s_ptr);
            _mm512_mask_storeu_epi32((void*)d_ptr, mask, v);
            s_ptr += head; d_ptr += head; current_chunk -= head;
        }

        // 2. Main Loop
        size_t loops = current_chunk / 16;
        for (size_t k = 0; k < loops; ++k) {
            _mm512_stream_si512((__m512i*)d_ptr, _mm512_loadu_si512((const void*)s_ptr));
            s_ptr += 16;
            d_ptr += 16;
        }
        
        // 3. Tail
        size_t tail = current_chunk % 16;
        if (tail > 0) {
            __mmask16 mask = (1 << tail) - 1;
            __m512i v = _mm512_mask_loadu_epi32(_mm512_setzero_si512(), mask, (const void*)s_ptr);
            _mm512_mask_storeu_epi32((void*)d_ptr, mask, v);
        }

        // --- Tile 结束，移动到下一块 ---
        offset += TILE_SIZE;
    }
}

// ==========================================
// 7. Mimic Scatter (The "Ghost" Kernel)
// ==========================================
// 目的：1:1 复刻 Scatter 的指令流水线。
// 包含：+128B 预取，无展开循环，以及一个总是为真的分支判断。
// 如果这个能跑 41GB/s，说明 Scatter 的高性能来自于指令混合度(Pacing)。
void kernel_mimic_scatter(const unsigned int* src, unsigned int* dest, size_t size) {
    if (size == 0) return;
    
    // 1. Head Alignment (Copy from previous)
    uintptr_t dest_addr = (uintptr_t)dest;
    size_t misalignment = (dest_addr & 63) >> 2;
    if (misalignment != 0) {
        size_t head = std::min(size, (size_t)(16 - misalignment));
        __mmask16 mask = (1 << head) - 1;
        __m512i v = _mm512_mask_loadu_epi32(_mm512_setzero_si512(), mask, (const void*)src);
        _mm512_mask_storeu_epi32((void*)dest, mask, v);
        src += head; dest += head; size -= head;
    }

    const char* p_src = (const char*)src;
    
    // 构造一个用来比较的向量，确保 mask 总是 0xFFFF
    // 假设数据都 > 0，我们用 0 来比较，或者直接用全0向量做比较
    // 在 Sorted Scatter 中，mask = _mm512_cmplt_epi32_mask(v_data, v_pivot);
    // 这里我们模拟这个开销
    __m512i v_pivot = _mm512_setzero_si512(); // 全0

    // 2. Body: Strict 1:1 Copy of Scatter Loop
    // 不做 Unroll，严格单次迭代，模拟 Scatter 的节奏
    size_t loops = size / 16;
    for (size_t k = 0; k < loops; ++k) {
        // [关键特征 1] 单次指令级预取，距离 128 字节 (2 Cache Lines)
        _mm_prefetch(p_src + 128, _MM_HINT_T0);

        // Load
        __m512i v_data = _mm512_loadu_si512((const void*)src);

        // [关键特征 2] 插入计算指令模拟 Compare 开销
        // 你的 Scatter 是 cmplt，这里我们用 cmpneq (全1) 来模拟
        // 注意：我们不需要真的分支跳转，只需要让 CPU 执行这条指令占位
        __mmask16 mask = _mm512_cmpneq_epi32_mask(v_data, v_data); // 结果总是 0，但这行会被优化掉吗？
        // 为了防止被优化，我们用 volatile 或者将其混入判断
        // 更接近 Scatter 的做法是：
        // if (mask == 0xFFFF) stream_store
        
        // 强制执行比较：(src > -1) 总是 True (unsigned)
        // 使用有符号比较，data > -1 ?
        // 简单粗暴点：直接 stream store，但保留 prefetch 的位置和单循环结构
        
        _mm512_stream_si512((__m512i*)dest, v_data);

        src += 16;
        dest += 16;
        p_src += 64; // char 指针每次进 64 字节
    }

    size -= loops * 16;

    // 3. Tail
    if (size > 0) {
        __mmask16 mask = (1 << size) - 1;
        __m512i v = _mm512_mask_loadu_epi32(_mm512_setzero_si512(), mask, (const void*)src);
        _mm512_mask_storeu_epi32((void*)dest, mask, v);
    }
}
// ==========================================
// 8. AVX2 (256-bit) Stream Store
// ==========================================
// 策略：使用 _mm256_stream_si256 避免 AVX-512 降频惩罚。
// 理论上能维持 3.0 GHz 频率，同时 NT Store 依然能跑满带宽。
void kernel_avx2_stream(const unsigned int* src, unsigned int* dest, size_t size) {
    if (size == 0) return;

    // 1. Head: Align to 32 Bytes (AVX2 alignment)
    // AVX2 需要 32 字节对齐，不过 loadu/storeu 对齐要求低，但 stream 需要地址对齐
    uintptr_t dest_addr = (uintptr_t)dest;
    size_t misalignment = dest_addr & 31; // % 32

    if (misalignment != 0) {
        size_t bytes_to_align = 32 - misalignment;
        size_t elems_to_align = bytes_to_align >> 2;
        size_t head = std::min(size, elems_to_align);
        
        // 标量处理头部
        for (size_t i = 0; i < head; ++i) {
            dest[i] = src[i];
        }
        
        src += head; dest += head; size -= head;
    }

    // 2. Body: Unroll 8 (8 x 8 ints = 64 ints = 256 Bytes per loop)
    // 用 256bit 也就是 32字节，一次处理 8 个 int
    size_t loops = size / 64; 

    // 手动预取距离，AVX2 流水线可能稍微慢点，预取稍微近一点
    const char* p_src = (const char*)src;
    const int PREFETCH_DIST = 128; 

    for (size_t k = 0; k < loops; ++k) {
        _mm_prefetch(p_src + PREFETCH_DIST, _MM_HINT_T0);
        _mm_prefetch(p_src + PREFETCH_DIST + 64, _MM_HINT_T0);

        // Load 8 vectors (256 bytes total)
        __m256i v0 = _mm256_loadu_si256((const __m256i*)(src + 0));
        __m256i v1 = _mm256_loadu_si256((const __m256i*)(src + 8));
        __m256i v2 = _mm256_loadu_si256((const __m256i*)(src + 16));
        __m256i v3 = _mm256_loadu_si256((const __m256i*)(src + 24));
        __m256i v4 = _mm256_loadu_si256((const __m256i*)(src + 32));
        __m256i v5 = _mm256_loadu_si256((const __m256i*)(src + 40));
        __m256i v6 = _mm256_loadu_si256((const __m256i*)(src + 48));
        __m256i v7 = _mm256_loadu_si256((const __m256i*)(src + 56));

        // Stream Store (NT)
        _mm256_stream_si256((__m256i*)(dest + 0), v0);
        _mm256_stream_si256((__m256i*)(dest + 8), v1);
        _mm256_stream_si256((__m256i*)(dest + 16), v2);
        _mm256_stream_si256((__m256i*)(dest + 24), v3);
        _mm256_stream_si256((__m256i*)(dest + 32), v4);
        _mm256_stream_si256((__m256i*)(dest + 40), v5);
        _mm256_stream_si256((__m256i*)(dest + 48), v6);
        _mm256_stream_si256((__m256i*)(dest + 56), v7);

        src += 64;
        dest += 64;
        p_src += 256;
    }

    size -= loops * 64;

    // 3. Tail
    while (size > 0) {
        *dest++ = *src++;
        size--;
    }
}
int main(int argc, char* argv[]) {
    // 默认参数
    size_t n = 1000000000; // 10亿 int = 4GB
    size_t block_size = 16 * 1024; // 16K elements = 64KB
    std::string target_kernel = "all"; // 默认跑所有

    if (argc > 1) n = std::stoull(argv[1]);
    if (argc > 2) block_size = std::stoull(argv[2]);
    if (argc > 3) target_kernel = argv[3]; // 第3个参数选择 kernel

    std::cout << "=================================================" << std::endl;
    std::cout << "High-Performance Bandwidth Benchmark" << std::endl;
    std::cout << "Target     : " << target_kernel << std::endl;
    std::cout << "Array Size : " << n << " elements (" << (double)n * sizeof(T) / 1024 / 1024 / 1024 << " GB)" << std::endl;
    std::cout << "Block Size : " << block_size << " elements" << std::endl;
    std::cout << "Workers    : " << parlay::num_workers() << std::endl;
    std::cout << "Flush Size : " << FLUSH_SIZE_BYTES / 1024 / 1024 << " MB" << std::endl;
    std::cout << "=================================================" << std::endl;

    // 1. 初始化 Flush Buffer
    // init_flush_buffer();

    // ==========================================
    // 1. 使用我们自定义的对齐数组 (RAII)
    // ==========================================
    std::cout << "[Init] Allocating 4KB aligned memory..." << std::endl;
    AlignedArray<unsigned int> source_raw(n);
    AlignedArray<unsigned int> dest_raw(n);

// ==========================================
    // 2. 初始化数据 (使用 Parlay 代替 OpenMP)
    // ==========================================
    std::cout << "[Init] Touching memory with Parlay workers..." << std::endl;
    
    // 使用与测试相同的线程池，确保 NUMA 亲和性和 Cache 预热
    parlay::parallel_for(0, n, [&](size_t i) {
        source_raw[i] = (unsigned int)i;
        dest_raw[i] = 0; // 这一步写入会触发 2MB 大页的分配
    });

    // ==========================================
    // 3. 包装成 Slice (The "Wrap")
    // ==========================================
    // 这两个对象表现得和 sequence 一模一样，但是指向我们对齐的内存
    auto source = source_raw.as_slice();
    auto dest = dest_raw.as_slice();

    // 检查对齐情况 (Debug)
    size_t s_align = (uintptr_t)source.begin() % 64;
    size_t d_align = (uintptr_t)dest.begin() % 64;
    std::cout << "[Init] Source Align Offset: " << s_align 
              << " | Dest Align Offset: " << d_align << std::endl;
    
    if (s_align == 0 && d_align == 0) {
        std::cout << "[Init] Perfect Alignment Confirmed (AVX-512 Friendly)." << std::endl;
    } else {
        std::cerr << "[Warning] Memory is NOT 64-byte aligned!" << std::endl;
    }
    size_t num_blocks = (n + block_size - 1) / block_size;

    std::cout << "\n[Running Benchmarks]..." << std::endl;

    auto run1 = [&](std::string name, auto kernel) {
  if (target_kernel == "all" || target_kernel == name) {
    benchmark_routine_amp(name, n, num_blocks, block_size, source, dest, kernel);
  }
};

// 建议：Python 里每次只传一个 kernel，这里仍支持 all
run1("std_copy", kernel_std_copy);
run1("std_move", kernel_std_move);
run1("avx2", kernel_avx256_copy);
run1("avx512", kernel_avx512_copy);
run1("stream", kernel_avx512_stream);
run1("stream_prefetch", kernel_avx512_stream_prefetch);
run1("stream_blocked", kernel_avx512_stream_blocked);
run1("mimic_scatter", kernel_mimic_scatter);
run1("avx2_stream", kernel_avx2_stream);


    return 0;
}