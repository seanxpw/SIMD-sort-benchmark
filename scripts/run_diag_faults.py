#!/usr/bin/env python3
import argparse, csv, itertools, os, re, subprocess, sys, time
from pathlib import Path

# =========================================================
# 定义我们要监控的诊断性事件
# =========================================================
DIAG_EVENTS = [
    # --- 核心干扰项 ---
    "page-faults",          # 总缺页 (Soft + Hard)
    "minor-faults",         # 软缺页 (不需要读盘，但需要OS介入)
    "major-faults",         # 硬缺页 (需要读盘，性能杀手)
    "context-switches",     # 进程/线程切换 (调度器干扰)
    "cpu-migrations",       # 跨核迁移 (NUMA亲和性破坏者)
    
    # --- TLB (Translation Lookaside Buffer) ---
    # 如果 Huge Pages 生效，dTLB-load-misses 应该极低
    "dTLB-loads",
    "dTLB-load-misses",
    "dTLB-stores",
    "dTLB-store-misses",
    
    # --- Cache ---
    # 检查是否有意外的 Cache 颠簸
    "LLC-loads",
    "LLC-load-misses",
    "L1-dcache-load-misses",
]

def run(cmd, **kwargs):
    return subprocess.run(cmd, **kwargs)

def parse_list(s, cast=str):
    parts = []
    for tok in re.split(r"[,\s]+", (s or "").strip()):
        if tok:
            parts.append(cast(tok))
    return parts

# 简单的 CSV 解析器，提取 perf stat 的计数值
def parse_perf_val(text: str, event_name: str) -> float:
    # 格式通常是: 1,234,567,,event_name,100.00,,
    # 我们需要模糊匹配 event_name
    for line in text.splitlines():
        if event_name in line:
            parts = line.split(',')
            if len(parts) > 0:
                try:
                    val_str = parts[0].replace('<not supported>', '0').replace('<not counted>', '0')
                    return float(val_str)
                except ValueError:
                    return 0.0
    return 0.0

def parse_kv(stdout: str, key: str):
    for line in stdout.splitlines():
        if line.startswith(key + " "):
            return line.split(None, 1)[1].strip()
    return None

def main():
    ap = argparse.ArgumentParser("Diagnostic runner for Page Faults, TLB & Scheduler noise")
    ap.add_argument("-o", "--outdir", default="./diag_runs")
    ap.add_argument("--repeats", type=int, default=1)
    ap.add_argument("--n", default="1000000000")
    
    # 默认只测大 Block，因为小 Block 本身调度开销就大，测不准
    ap.add_argument("--block", default="2097152", help="Block sizes") 
    
    # 重点对比 stream (慢的) 和 scatter (快的)
    ap.add_argument("--kernels", default="std_copy,std_move,avx2,avx512,stream,stream_prefetch,stream_blocked,mimic_scatter,avx2_stream")
    ap.add_argument("--csv-name", default="diag_results.csv")
    
    args, rest = ap.parse_known_args()
    if rest and rest[0] == "--":
        rest = rest[1:]
    if not rest:
        print("Usage: python run_diag_faults.py [options] -- ./bench_bandwidth", file=sys.stderr)
        sys.exit(1)
    
    base_cmd = rest
    out_root = Path(args.outdir)
    ts = time.strftime("%Y-%m-%d_%H_%M_%S")
    root = out_root / f"diag_faults_{ts}"
    root.mkdir(parents=True, exist_ok=True)

    n_list = [int(x) for x in parse_list(args.n, int)]
    blk_list = [int(x) for x in parse_list(args.block, int)]
    kernels = [k.strip() for k in parse_list(args.kernels, str)]

    # 构建 perf -e 参数
    # 注意：某些机器可能不支持所有事件，perf 会自动报错或忽略，这里简单处理
    valid_events = []
    # 预检 perf list (可选，这里直接尝试运行)
    event_str = ",".join(DIAG_EVENTS)

    out_csv = root / args.csv_name
    print(f"Logging to: {out_csv}")
    
    with out_csv.open("w", newline="") as fcsv:
        w = csv.writer(fcsv)
        header = [
            "kernel", "n", "block", "rep", "roi_sec",
            "page_faults", "minor_faults", "major_faults",
            "ctx_switches", "cpu_migrations",
            "dtlb_misses", "dtlb_miss_ratio_pct",
            "llc_misses", "llc_miss_ratio_pct",
            "path"
        ]
        w.writerow(header)

        total_runs = len(n_list) * len(blk_list) * len(kernels) * args.repeats
        run_idx = 0

        for n, blk, ker in itertools.product(n_list, blk_list, kernels):
            for rep in range(args.repeats):
                run_idx += 1
                tag = f"{ker}_n{n}_blk{blk}_rep{rep}"
                run_dir = root / tag
                run_dir.mkdir(parents=True, exist_ok=True)

                # FIFO Control Setup
                fifo_ctl = run_dir / "perf_ctl.fifo"
                fifo_ack = run_dir / "perf_ack.fifo"
                for pth in (fifo_ctl, fifo_ack):
                    if pth.exists(): os.unlink(pth)
                os.mkfifo(fifo_ctl)
                os.mkfifo(fifo_ack)

                env = os.environ.copy()
                env["PERF_CTL_FIFO"] = str(fifo_ctl)
                env["PERF_ACK_FIFO"] = str(fifo_ack)

                perf_out = run_dir / "perf_stat.txt"
                
                # App command
                app_cmd_args = base_cmd + [str(n), str(blk), ker]
                
                # Perf command
                perf_cmd = [
                    "perf", "stat", "-a",
                    "--no-big-num", "--no-scale",
                    "-x", ",",
                    "-e", event_str,
                    "-o", str(perf_out),
                    "--delay=-1", 
                    "--control", f"fifo:{fifo_ctl},{fifo_ack}",
                    "--"
                ] + app_cmd_args

                print(f"[{run_idx}/{total_runs}] Running {ker}...", end=" ", flush=True)

                try:
                    p = run(perf_cmd, stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True, env=env)
                    
                    if p.returncode != 0:
                        print("FAIL")
                        (run_dir / "error.log").write_text(p.stderr)
                        continue

                    # Parse App Output
                    roi_s = float(parse_kv(p.stdout, "ROI_SECONDS") or 0.0)
                    
                    # Parse Perf Output
                    p_txt = perf_out.read_text()
                    
                    pf = parse_perf_val(p_txt, "page-faults")
                    min_f = parse_perf_val(p_txt, "minor-faults")
                    maj_f = parse_perf_val(p_txt, "major-faults")
                    cs = parse_perf_val(p_txt, "context-switches")
                    migrations = parse_perf_val(p_txt, "cpu-migrations")
                    
                    dtlb_load = parse_perf_val(p_txt, "dTLB-loads")
                    dtlb_miss = parse_perf_val(p_txt, "dTLB-load-misses")
                    dtlb_ratio = (dtlb_miss / dtlb_load * 100.0) if dtlb_load > 0 else 0.0
                    
                    llc_load = parse_perf_val(p_txt, "LLC-loads")
                    llc_miss = parse_perf_val(p_txt, "LLC-load-misses")
                    llc_ratio = (llc_miss / llc_load * 100.0) if llc_load > 0 else 0.0

                    w.writerow([
                        ker, n, blk, rep, f"{roi_s:.6f}",
                        int(pf), int(min_f), int(maj_f),
                        int(cs), int(migrations),
                        int(dtlb_miss), f"{dtlb_ratio:.4f}",
                        int(llc_miss), f"{llc_ratio:.4f}",
                        str(run_dir)
                    ])
                    fcsv.flush() # Ensure write

                    print(f"OK (Faults: {int(pf)}, TLB Miss%: {dtlb_ratio:.2f}%)")

                except Exception as e:
                    print(f"CRASH: {e}")
                    continue
                finally:
                    # Cleanup FIFOs
                    for pth in (fifo_ctl, fifo_ack):
                        if pth.exists(): os.unlink(pth)

    print(f"\nDone. Results saved to {out_csv}")

if __name__ == "__main__":
    main()