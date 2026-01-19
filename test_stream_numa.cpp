#include <iostream>
#include <vector>
#include <chrono>
#include <cstdlib>
#include <cstring>
#include <iomanip>
#include <numa.h>
#include <unistd.h>
#include <memory>
#include <fstream>
#include <sstream>

#include "parlay/parallel.h" // 确保包含你的调度器头文件

// ==========================================
// 1. 内存包装器：只分配虚拟内存，绝不触碰！
// ==========================================
template <typename T>
struct UninitVector {
    T* data;
    size_t size;

    UninitVector(size_t n) : size(n) {
        // 4096 字节对齐，利用大页优势（如果有），且避免 False Sharing
        // aligned_alloc 分配的内存是未初始化的，OS 尚未分配物理页
        data = static_cast<T*>(aligned_alloc(4096, n * sizeof(T)));
        if (!data) {
            std::cerr << "Memory allocation failed!" << std::endl;
            exit(1);
        }
    }

    ~UninitVector() {
        free(data);
    }

    T& operator[](size_t i) { return data[i]; }
    const T& operator[](size_t i) const { return data[i]; }
    
    // 禁止拷贝
    UninitVector(const UninitVector&) = delete;
    UninitVector& operator=(const UninitVector&) = delete;
};

// ==========================================
// 2. 工具函数
// ==========================================

static void sink(double x) {
    asm volatile("" : : "r,m"(x) : "memory");
}

static void run_cmd(const std::string& cmd) {
    int rc = std::system(cmd.c_str());
    (void)rc;
}

static void print_numa_distribution(pid_t pid, const char* label) {
    std::cout << "\n--- [" << label << "] Memory Distribution ---\n";
    std::stringstream ss;
    ss << "numastat -p " << pid << " | grep 'Private'";
    run_cmd(ss.str());
}

// ==========================================
// 3. 核心测试逻辑
// ==========================================

template <typename InitFunc, typename ComputeFunc>
void run_benchmark(const char* name, size_t N, int iters, 
                  InitFunc&& init_f, ComputeFunc&& compute_f) {
    
    std::cout << "\n=============================================\n";
    std::cout << "Running Case: " << name << "\n";
    std::cout << "=============================================\n";

    // 1. 分配 (此时没有物理内存占用)
    UninitVector<double> A(N), B(N), C(N);

    // 2. 初始化 (这一步决定了物理内存落在哪个 NUMA 节点！)
    auto t0 = std::chrono::high_resolution_clock::now();
    init_f(A, B, C);
    auto t1 = std::chrono::high_resolution_clock::now();
    
    // 打印初始化后的内存分布
    print_numa_distribution(getpid(), "After Init");

    // 3. 计算带宽
    double total_time = 0;
    double check_sum = 0;

    for(int i=0; i<iters; i++) {
        auto ts = std::chrono::high_resolution_clock::now();
        compute_f(A, B, C);
        auto te = std::chrono::high_resolution_clock::now();
        total_time += std::chrono::duration_cast<std::chrono::duration<double>>(te - ts).count();
        
        // 简单采样防止优化
        check_sum += A[N/2]; 
    }

    sink(check_sum);

    double avg_time = total_time / iters;
    // STREAM Triad: 3 memory ops (2 reads, 1 write) per element
    double bytes = 3.0 * N * sizeof(double); 
    double gb_s = (bytes / 1e9) / avg_time;

    std::cout << "Compute Time: " << avg_time << " s\n";
    std::cout << "Bandwidth:    " << gb_s << " GB/s\n";
}

int main(int argc, char** argv) {
    // if (numa_available() < 0) {
    //     std::cerr << "NUMA not supported.\n";
    //     return 1;
    // }

    // 默认参数：1亿个double (约 2.4GB)
    size_t N = 100000000; 
    if (argc > 1) N = std::atol(argv[1]);
    
    // 强制使用你的 Scheduler (单例模式或者传入)
    // 假设 parlay 内部已经初始化好了 scheduler
    // 如果你的 parfor 需要 scheduler 对象，请在这里获取
    // 这里的 scheduler 获取方式取决于你的实现，假设 parlay::scheduler 是全局可见或可构造的
    // auto& sched = *parlay::scheduler<parlay::WorkStealingJob>::get_current_scheduler(); 
    // ^ 上面这行如果不好获取，就用下面的静态调用方式，或者在 main 里构造一个 scheduler
    
    // 注意：根据你之前的头文件，我们需要一个 scheduler 实例
    // parlay::fork_join_scheduler::scheduler_t my_scheduler(std::thread::hardware_concurrency());

    std::cout << "Elements: " << N << " (" << (N * sizeof(double) * 3 / 1024 / 1024) << " MB per array set)\n";

    // ---------------------------------------------------------
    // Case 1: 错误示范 (Baseline)
    // 主线程初始化 (全都落在 Node 0) -> 并行计算
    // ---------------------------------------------------------
    run_benchmark("Baseline: this is compiler with original parlaylib scheduler \n \
      parlay::parallel_for init -> Parallel Compute", N, 5, 
        [&](auto& a, auto& b, auto& c) {
            // 模拟 std::vector 的默认行为，主线程顺序写入
            parlay::parallel_for(0, N, [&](size_t i) {
                a[i] = 0.0; b[i] = 1.0; c[i] = 2.0;
            });
        },
        [&](auto& a, const auto& b, const auto& c) {
            // 普通 parfor
            parlay::parallel_for(0, N, [&](size_t i) {
                a[i] = b[i] + 3.0 * c[i];
            });
        }
    );

    // ---------------------------------------------------------
    // Case 2: 你的 NUMA 优化
    // Numa-aware Init (分散在 Node 0/1) -> Numa-aware Compute
    // ---------------------------------------------------------
    // run_benchmark("Optimized: Numa-Aware Init -> Numa-Aware Compute", N, 5,
    //     [&](auto& a, auto& b, auto& c) {
    //         // 使用你的 numa_aware_parfor 进行初始化！
    //         // 这会触发 First-Touch，将物理页分配给执行线程所在的 Node
    //         parlay::numa_aware_parallel_for(0, N, [&](size_t i) {
    //             a[i] = 0.0; b[i] = 1.0; c[i] = 2.0;
    //         });
    //     },
    //     [&](auto& a, const auto& b, const auto& c) {
    //         // 使用同样的划分进行计算，保证线程访问的是本地内存
    //         parlay::numa_aware_parallel_for( 0, N, [&](size_t i) {
    //             a[i] = b[i] + 3.0 * c[i];
    //         });
    //     }
    // );

    return 0;
}