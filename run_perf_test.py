#!/usr/bin/env python3
import os
import re
import sys
import subprocess
from typing import Tuple, Optional, List

def get_cpu_base_frequency_ghz() -> float:
    try:
        out = subprocess.check_output(["lscpu"], text=True)
        m = re.search(r"Model name:.*@\s+(\d+\.\d+)\s*GHz", out, re.IGNORECASE)
        if m:
            return float(m.group(1))
    except Exception:
        pass

    try:
        with open("/proc/cpuinfo", "r") as f:
            for line in f:
                if "model name" in line:
                    m = re.search(r"@\s+(\d+\.\d+)\s*GHz", line)
                    if m:
                        return float(m.group(1))
                    break
    except Exception:
        pass

    print("Warning: Could not detect CPU base frequency. Defaulting to 1.0 GHz.")
    return 1.0

def parse_kv(stdout: str, key: str) -> Optional[str]:
    for line in stdout.splitlines():
        if line.startswith(key + " "):
            return line.split(None, 1)[1].strip()
    return None

def run_cmd_capture(cmd: List[str]) -> Tuple[int, str, str]:
    p = subprocess.run(cmd, stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True)
    return p.returncode, p.stdout, p.stderr

def clean_number(num_str: str) -> float:
    return float(num_str.replace(",", "").strip())

def get_goodput_from_app(exe: str, n: int, block: int, kernel: str, numactl: bool = True) -> Tuple[float, float, float]:
    """
    Returns: (roi_seconds, good_oneway_gbps, good_rw_gbps) from app stdout.
    Assumes your latest C++ prints:
      ROI_SECONDS ...
      GOODPUT_ONEWAY_GBps ...
      GOODPUT_RW_GBps ...
    """
    cmd = []
    if numactl:
        cmd += ["numactl", "-i", "all"]
    cmd += [exe, str(n), str(block), kernel]

    rc, out, err = run_cmd_capture(cmd)
    if rc != 0:
        raise RuntimeError(f"App failed: {' '.join(cmd)}\n{err}")

    roi_s = float(parse_kv(out, "ROI_SECONDS"))
    good_one = float(parse_kv(out, "GOODPUT_ONEWAY_GBps"))
    good_rw = float(parse_kv(out, "GOODPUT_RW_GBps"))
    return roi_s, good_one, good_rw

def get_real_freq_ghz(exe: str, n: int, block: int, kernel: str, base_freq_ghz: float, numactl: bool = True) -> float:
    """
    perf stat -e cycles,ref-cycles ... (process-scoped)
    real_freq = base_freq * (cycles / ref-cycles)
    """
    cmd = ["perf", "stat", "-e", "cycles,ref-cycles"]
    if numactl:
        cmd += ["numactl", "-i", "all"]
    cmd += [exe, str(n), str(block), kernel]

    rc, out, err = run_cmd_capture(cmd)
    if rc != 0:
        raise RuntimeError(f"perf stat failed: {' '.join(cmd)}\n{err}")

    cycles_m = re.search(r"(\d[\d,]*)\s+cycles", err)
    ref_m = re.search(r"(\d[\d,]*)\s+ref-cycles", err)
    if not cycles_m or not ref_m:
        raise RuntimeError(f"Could not parse cycles/ref-cycles from perf output:\n{err}")

    cycles = clean_number(cycles_m.group(1))
    ref = clean_number(ref_m.group(1))
    if ref <= 0:
        return 0.0
    return base_freq_ghz * (cycles / ref)

def main():
    BASE_FREQ = get_cpu_base_frequency_ghz()
    print(f"Detected Base Frequency: {BASE_FREQ:.2f} GHz")

    EXE_PATH = "./bench_bandwidth"
    N = 1_000_000_000

    # 你可以按需要改成列表
    BLOCK_SIZES = [4096, 16384, 65536]
    KERNELS = ["std_copy", "std_move", "avx2", "avx512",
               "stream", "stream_prefetch", "stream_blocked",
               "mimic_scatter", "avx2_stream"]

    REPEATS = 3

    if not os.path.exists(EXE_PATH):
        print(f"Error: {EXE_PATH} not found.")
        sys.exit(1)

    print(f"{'Kernel':<15} {'Block':>8} | {'GoodRW Max':>11} | {'Freq Avg':>8} | {'Eff (GB/s/GHz)':>14}")
    print("-" * 70)

    for blk in BLOCK_SIZES:
        for ker in KERNELS:
            good_rw_vals = []
            freqs = []
            for _ in range(REPEATS):
                # run app once (it already does warmup+ROI inside)
                _, _, good_rw = get_goodput_from_app(EXE_PATH, N, blk, ker, numactl=True)
                good_rw_vals.append(good_rw)

                # run perf cycles/ref-cycles once
                fghz = get_real_freq_ghz(EXE_PATH, N, blk, ker, BASE_FREQ, numactl=True)
                freqs.append(fghz)

            good_rw_max = max(good_rw_vals) if good_rw_vals else 0.0
            freq_avg = sum(freqs) / len(freqs) if freqs else 0.0
            eff = (good_rw_max / freq_avg) if freq_avg > 0 else 0.0

            print(f"{ker:<15} {blk:>8} | {good_rw_max:>11.2f} | {freq_avg:>8.2f} | {eff:>14.2f}")

    print("-" * 70)
    print("Real Freq = Base_Freq * (cycles / ref-cycles)")

if __name__ == "__main__":
    main()
