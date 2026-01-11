import os
import re
import subprocess

def get_cpu_base_frequency():
    """
    尝试自动获取 CPU 基准频率 (GHz)。
    方法：
    1. 尝试解析 lscpu 的输出 (最准确)。
    2. 回退到读取 /proc/cpuinfo 的 model name。
    """
    try:
        # 方法 1: 使用 lscpu 命令
        # lscpu 输出中通常有一行 "Model name: ... @ 2.40GHz"
        output = subprocess.check_output(["lscpu"], universal_newlines=True)
        
        # 匹配 "Model name" 行中的 "@ x.xxGHz"
        match = re.search(r"Model name:.*@\s+(\d+\.\d+)\s*GHz", output, re.IGNORECASE)
        if match:
            return float(match.match.group(1))
            
        # 某些较新的 lscpu 版本可能有 "CPU max MHz" 但那通常是 Turbo 频率
        # 我们还是优先找 model name 里的标称频率
        
    except Exception:
        pass

    try:
        # 方法 2: 读取 /proc/cpuinfo
        with open("/proc/cpuinfo", "r") as f:
            for line in f:
                if "model name" in line:
                    # Intel(R) Xeon(R) Silver 4214R CPU @ 2.40GHz
                    match = re.search(r"@\s+(\d+\.\d+)\s*GHz", line)
                    if match:
                        return float(match.group(1))
                    break
    except Exception:
        pass

    # 如果都失败了，打印警告并返回一个默认值 (防止除以0)
    print("Warning: Could not detect CPU base frequency. Defaulting to 1.0 GHz (ratios will be raw).")
    return 1.0

# ================= 使用示例 =================
# 在你的脚本配置区域直接调用：
BASE_FREQ_GHZ = get_cpu_base_frequency()

print(f"Detected Base Frequency: {BASE_FREQ_GHZ} GHz")
# ================= 配置区域 =================
EXE_PATH = "./bench_bandwidth"  # 可执行文件路径
# 确保 array size 和 block size 与你手动运行的一致
ARRAY_SIZE = 1000 * 1000 * 1000  
BLOCK_SIZE = 16 * 1024           
# 你的 CPU 基准频率 (Xeon 4214R @ 2.40GHz)
# BASE_FREQ_GHZ = 2.40 
KERNELS = ["std_copy", "std_move", "avx2", "avx512", "stream", "stream_prefetch", "stream_blocked", "mimic_scatter", "avx2_stream"]
# ===========================================

def clean_number(num_str):
    """去除逗号并转为 float"""
    return float(num_str.replace(",", "").strip())

def get_bandwidth_clean(kernel):
    """Run 1: 不带 perf，只测带宽"""
    cmd = [
        "numactl", "-i", "all",
        EXE_PATH, str(ARRAY_SIZE), str(BLOCK_SIZE), kernel
    ]
    try:
        result = subprocess.run(cmd, stdout=subprocess.PIPE, stderr=subprocess.PIPE, universal_newlines=True)
        # 匹配: | Max: 36.33 GB/s
        match = re.search(r"\|\s*Max:\s*(\d+\.\d+)\s*GB/s", result.stdout)
        if match:
            return float(match.group(1))
    except Exception as e:
        print(f"Error running clean bench for {kernel}: {e}")
    return 0.0

def get_frequency_perf(kernel):
    """Run 2: 带 perf，只测频率"""
    cmd = [
        "perf", "stat", 
        "-e", "cycles,ref-cycles", # 只需要这两个核心指标
        "numactl", "-i", "all",
        EXE_PATH, str(ARRAY_SIZE), str(BLOCK_SIZE), kernel
    ]
    try:
        result = subprocess.run(cmd, stdout=subprocess.PIPE, stderr=subprocess.PIPE, universal_newlines=True)
        stderr_output = result.stderr
        
        # 提取 cycles (兼容 cycles:u)
        # 匹配: 159,482,776,363      cycles:u
        cycles_match = re.search(r"(\d[\d,]*)\s+cycles", stderr_output)
        
        # 提取 ref-cycles
        ref_match = re.search(r"(\d[\d,]*)\s+ref-cycles", stderr_output)
        
        if cycles_match and ref_match:
            cycles = clean_number(cycles_match.group(1))
            ref_cycles = clean_number(ref_match.group(1))
            
            if ref_cycles > 0:
                # 公式: 真实频率 = 基准频率 * (cycles / ref-cycles)
                real_freq = BASE_FREQ_GHZ * (cycles / ref_cycles)
                return real_freq
    except Exception as e:
        print(f"Error running perf for {kernel}: {e}")
    return 0.0

def main():
    if not os.path.exists(EXE_PATH):
        print(f"Error: {EXE_PATH} not found.")
        sys.exit(1)

    print(f"{'Kernel Name':<15} | {'Clean BW (GB/s)':<18} | {'Real Freq (GHz)':<18} | {'Efficiency (GB/s/GHz)'}")
    print("-" * 80)

    for kernel in KERNELS:
        # 1. 获取纯净带宽
        bw = get_bandwidth_clean(kernel)
        
        # 2. 获取真实频率
        freq = get_frequency_perf(kernel)
        
        # 3. 计算效率
        eff = 0.0
        if freq > 0:
            eff = bw / freq
            
        print(f"{kernel:<15} | {bw:<18.2f} | {freq:<18.2f} | {eff:.2f}")

    print("-" * 80)
    print(f"Base Frequency used for calc: {BASE_FREQ_GHZ} GHz")
    print("Real Freq = Base_Freq * (cycles / ref-cycles)")

if __name__ == "__main__":
    main()