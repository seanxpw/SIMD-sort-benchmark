import subprocess
import os
import csv
import re
import time

# ================= 配置区域 =================
EXE = "./build/bench_scatter_parlay"  # 你的可执行文件路径
OUTPUT_CSV = "scatter_results.csv"    # 结果保存文件

# 测试参数网格
N = 1000000000  # 10亿元素 (约 4GB 数据)

# 1. 桶数量列表 (从 2 到 1024)
BUCKETS_LIST = [2, 4, 8, 16, 32, 64, 128, 256, 512, 1024]

# 2. Block 大小列表 (单位: 元素个数)
# 16K   ints = 64KB  (L1/L2 boundary)
# 64K   ints = 256KB (L2 typical)
# 256K  ints = 1MB   (L3 slice)
# 1024K ints = 4MB   (L3 capacity)
BLOCK_SIZES = [16*1024, 64*1024, 256*1024, 1024*1024]

# 3. 数据分布
DISTS = ["uniform", "sorted", "reverse"]
# ===========================================

def run_test(buckets, block_size, dist):
    """
    运行单个测试并解析结果
    返回: (time_sec, bw_gbs)
    """
    cmd = [
        "numactl", "-i", "all",
        EXE, str(N), str(buckets), str(block_size), dist
    ]
    
    print(f"Running: Dist={dist}, Block={block_size/1024:.0f}K, Buckets={buckets} ... ", end="", flush=True)

    try:
        # 运行命令
        process = subprocess.run(
            cmd, 
            stdout=subprocess.PIPE, 
            stderr=subprocess.PIPE, 
            universal_newlines=True,
            check=True # 如果 crash 则抛出异常
        )
        
        output = process.stdout
        
        # 使用正则解析结果行
        # 格式参考 C++: RESULT: N=... Buckets=... Time=0.123s Bandwidth=35.5 GB/s
        # 我们这里放宽匹配，只要找到数值即可
        match_time = re.search(r"Time=([\d\.]+)", output)
        match_bw = re.search(r"Bandwidth=([\d\.]+)", output)
        
        if match_time and match_bw:
            time_val = float(match_time.group(1))
            bw_val = float(match_bw.group(1))
            print(f"Done. {bw_val} GB/s")
            return time_val, bw_val
        else:
            print("Parse Error")
            return None, None
            
    except subprocess.CalledProcessError as e:
        print(f"CRASHED! Return code: {e.returncode}")
        return None, None
    except Exception as e:
        print(f"Error: {e}")
        return None, None

def main():
    # 检查可执行文件是否存在
    if not os.path.exists(EXE):
        print(f"Error: Executable {EXE} not found. Did you run cmake & make?")
        return

    # 初始化 CSV 文件
    file_exists = os.path.exists(OUTPUT_CSV)
    
    with open(OUTPUT_CSV, mode='w', newline='') as csvfile:
        fieldnames = ['Distribution', 'BlockSize_Elements', 'BlockSize_KB', 'Buckets', 'Time_s', 'Bandwidth_GBs']
        writer = csv.DictWriter(csvfile, fieldnames=fieldnames)
        
        writer.writeheader()
        
        # 开始网格搜索
        total_tests = len(DISTS) * len(BLOCK_SIZES) * len(BUCKETS_LIST)
        current_test = 0
        
        for dist in DISTS:
            for blk in BLOCK_SIZES:
                for b in BUCKETS_LIST:
                    current_test += 1
                    
                    time_s, bw = run_test(b, blk, dist)
                    
                    if time_s is not None:
                        writer.writerow({
                            'Distribution': dist,
                            'BlockSize_Elements': blk,
                            'BlockSize_KB': blk * 4 / 1024, # assuming int is 4 bytes
                            'Buckets': b,
                            'Time_s': time_s,
                            'Bandwidth_GBs': bw
                        })
                        csvfile.flush() # 确保立即写入磁盘，防止中断丢失数据
                    
                    # 稍微歇一下，防止过热导致降频（可选）
                    # time.sleep(0.5) 

    print(f"\nBenchmark finished. Results saved to {OUTPUT_CSV}")

if __name__ == "__main__":
    main()