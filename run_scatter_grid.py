#!/usr/bin/env python3
import argparse, csv, itertools, os, re, subprocess, sys, time
from pathlib import Path

# =========================================================
# 通用工具函数 (和 bandwidth 脚本一样，无需改动)
# =========================================================
def run(cmd, **kwargs):
    return subprocess.run(cmd, **kwargs)

def parse_list(s, cast=str):
    parts = []
    for tok in re.split(r"[,\s]+", (s or "").strip()):
        if tok: parts.append(cast(tok))
    return parts

def get_imc_events():
    # 自动探测 perf uncore imc 事件
    p = run(["perf", "list"], stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True)
    if p.returncode != 0: return ["imc/cas_count_read/", "imc/cas_count_write/"]
    pat = re.compile(r"^\s*(uncore_imc_(\d+))/cas_count_read/")
    bases = []
    seen = set()
    for line in p.stdout.splitlines():
        m = pat.search(line)
        if m:
            base, idx = m.group(1), int(m.group(2))
            if base not in seen:
                seen.add(base)
                bases.append((idx, base))
    if bases:
        bases.sort()
        ev = []
        for _, b in bases:
            ev.append(f"{b}/cas_count_read/")
            ev.append(f"{b}/cas_count_write/")
        return ev
    return ["imc/cas_count_read/", "imc/cas_count_write/"]

def parse_perf_stat_csv(text: str):
    def to_bytes(val: float, unit: str) -> float:
        unit = (unit or "").strip()
        if unit == "": return val * 64.0
        m = re.match(r"^([KMGTP]?)(i)?B$", unit)
        if m:
            pfx, ib = m.group(1), m.group(2)
            base = 1024.0 if ib else 1000.0
            exp = {"":0, "K":1, "M":2, "G":3, "T":4, "P":5}[pfx]
            return val * (base ** exp)
        return val * 64.0
    r = w = 0.0
    for line in text.splitlines():
        line = line.strip()
        if not line or line.startswith("#"): continue
        cols = line.split(",")
        if len(cols) < 3: continue
        raw, unit, event = cols[0].strip(), cols[1].strip(), cols[2].strip()
        if raw in ("<not supported>", "<not counted>"): continue
        try: val = float(raw)
        except: continue
        b = to_bytes(val, unit)
        if "cas_count_read" in event: r += b
        elif "cas_count_write" in event: w += b
    return r, w

def parse_kv(stdout: str, key: str):
    for line in stdout.splitlines():
        if line.startswith(key + " "):
            return line.split(None, 1)[1].strip()
    return None

def perf_noise(event_str: str, sleep_s: float, out_csv: Path):
    cmd = ["perf", "stat", "-a", "--no-big-num", "--no-scale", "-x", ",", "-e", event_str, "-o", str(out_csv), "--", "sleep", str(sleep_s)]
    run(cmd, stdout=subprocess.PIPE, stderr=subprocess.PIPE)
    rb, wb = parse_perf_stat_csv(out_csv.read_text())
    return (rb+wb)/sleep_s/1e9

# =========================================================
# Scatter 专用主逻辑
# =========================================================
def main():
    ap = argparse.ArgumentParser("Scatter AMP Grid")
    ap.add_argument("-o", "--outdir", default="./scatter_results")
    ap.add_argument("--repeats", type=int, default=3)
    ap.add_argument("--noise-s", type=float, default=0.2)
    
    # 核心参数：Scatter 特有的 buckets
    ap.add_argument("--n", default="1000000000")
    ap.add_argument("--buckets", default="256") 
    ap.add_argument("--block", default="262144")
    ap.add_argument("--csv-name", default="scatter_grid.csv")
    
    args, rest = ap.parse_known_args()
    if rest and rest[0] == "--": rest = rest[1:]
    if not rest:
        print("Usage: python run_scatter_grid.py [options] -- <cmd_path>", file=sys.stderr)
        sys.exit(1)
    base_cmd = rest

    out_root = Path(args.outdir)
    ts = time.strftime("%Y-%m-%d_%H_%M_%S")
    root = out_root / f"scatter_{ts}"
    root.mkdir(parents=True, exist_ok=True)

    n_list = [int(x) for x in parse_list(args.n, int)]
    bkt_list = [int(x) for x in parse_list(args.buckets, int)]
    blk_list = [int(x) for x in parse_list(args.block, int)]

    events = get_imc_events()
    event_str = ",".join(events)

    out_csv = root / args.csv_name
    with out_csv.open("w", newline="") as fcsv:
        w = csv.writer(fcsv)
        # 表头：增加了 buckets
        w.writerow([
            "n", "buckets", "block", "rep",
            "roi_s", "type_bytes",
            "logical_gb",          # 理论总量 (2N)
            "imc_read_gb",         # 物理读总量
            "imc_write_gb",        # 物理写总量
            "amp_read",            # 读放大
            "amp_write",           # 写放大
            "amp_total",           # 总放大
            "goodput_rw_gbps",     # 逻辑带宽
            "imc_total_gbps",      # 物理带宽
            "noise_total_gbps",
            "run_dir"
        ])

        total = len(n_list)*len(bkt_list)*len(blk_list)*args.repeats
        idx = 0

        # Grid Search: N x Buckets x Block
        for n, bkt, blk in itertools.product(n_list, bkt_list, blk_list):
            for rep in range(args.repeats):
                idx += 1
                tag = f"n{n}_bkt{bkt}_blk{blk}_rep{rep}"
                run_dir = root / tag
                run_dir.mkdir(parents=True, exist_ok=True)

                # 1. 测底噪
                noise_val = 0.0
                try:
                    noise_val = perf_noise(event_str, args.noise_s, run_dir / "noise.csv")
                except: pass

                # 2. 准备 FIFO
                fifo_ctl = run_dir / "perf_ctl.fifo"
                fifo_ack = run_dir / "perf_ack.fifo"
                for p in (fifo_ctl, fifo_ack):
                    if p.exists(): p.unlink()
                os.mkfifo(fifo_ctl)
                os.mkfifo(fifo_ack)

                env = os.environ.copy()
                env["PERF_CTL_FIFO"] = str(fifo_ctl)
                env["PERF_ACK_FIFO"] = str(fifo_ack)

                # 3. 构造命令：注意这里是 Scatter 的参数顺序
                # 假设 C++ 接收: <N> <Buckets> <Block>
                cmd_args = [str(n), str(bkt), str(blk)]
                
                full_cmd = [
                    "perf", "stat", "-a", "--no-big-num", "--no-scale", "-x", ",",
                    "-e", event_str,
                    "-o", str(run_dir / "perf_stat.csv"),
                    "--delay=-1", "--control", f"fifo:{fifo_ctl},{fifo_ack}",
                    "--"
                ] + base_cmd + cmd_args

                # 4. 执行
                p = run(full_cmd, stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True, env=env)
                (run_dir / "stdout.txt").write_text(p.stdout)
                (run_dir / "stderr.txt").write_text(p.stderr)

                if p.returncode != 0:
                    print(f"[{idx}/{total}] FAIL {tag}")
                    (run_dir / "run.err").write_text(p.stderr)
                    continue

                # 5. 解析结果
                try:
                    roi_s = float(parse_kv(p.stdout, "ROI_SECONDS"))
                    type_bytes = int(parse_kv(p.stdout, "TYPE_BYTES"))
                    # 读取 C++ 输出的带宽
                    good_rw_gbps = float(parse_kv(p.stdout, "GOODPUT_RW_GBps"))
                except (TypeError, ValueError):
                    print(f"[{idx}/{total}] PARSE ERROR {tag}")
                    continue

                # 6. 计算放大倍数
                rb, wb = parse_perf_stat_csv((run_dir / "perf_stat.csv").read_text())
                
                # Scatter 逻辑数据量: 读 N + 写 N = 2 * N
                # 注意：中间的 pivots 读取相比 N 极小，忽略不计
                logical_bytes = float(n) * type_bytes * 2.0
                
                logical_gb = logical_bytes / 1e9
                imc_read_gb = rb / 1e9
                imc_write_gb = wb / 1e9
                
                # 读放大：IMC实际读 / 理论读(N)
                amp_r = rb / (float(n) * type_bytes)
                # 写放大：IMC实际写 / 理论写(N)
                amp_w = wb / (float(n) * type_bytes)
                # 总放大
                amp_t = (rb + wb) / logical_bytes

                imc_total_gbps = (rb + wb) / roi_s / 1e9

                w.writerow([n, bkt, blk, rep,
                            f"{roi_s:.6f}", type_bytes,
                            f"{logical_gb:.4f}",
                            f"{imc_read_gb:.4f}", f"{imc_write_gb:.4f}",
                            f"{amp_r:.3f}", f"{amp_w:.3f}", f"{amp_t:.3f}",
                            f"{good_rw_gbps:.3f}", f"{imc_total_gbps:.3f}",
                            f"{noise_val:.3f}", str(run_dir)])

                print(f"[{idx}/{total}] OK {tag} "
                      f"Log:{logical_gb:.2f}G IMC_R:{imc_read_gb:.2f}G({amp_r:.2f}x) "
                      f"IMC_W:{imc_write_gb:.2f}G({amp_w:.2f}x)")

    print(f"\nDONE. Results: {out_csv}")

if __name__ == "__main__":
    main()