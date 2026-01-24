#!/usr/bin/env python3
import argparse, csv, itertools, os, re, subprocess, sys, time
from pathlib import Path

def run(cmd, **kwargs):
    return subprocess.run(cmd, **kwargs)

def parse_list(s, cast=str):
    parts = []
    for tok in re.split(r"[,\s]+", (s or "").strip()):
        if tok:
            parts.append(cast(tok))
    return parts

def get_imc_events():
    p = run(["perf", "list"], stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True)
    if p.returncode != 0:
        raise RuntimeError("perf list failed:\n" + p.stderr)

    pat = re.compile(r"^\s*(uncore_imc_(\d+))/cas_count_read/")
    bases = []
    seen = set()
    for line in p.stdout.splitlines():
        m = pat.search(line)
        if m:
            base = m.group(1)
            idx = int(m.group(2))
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
        if unit == "":
            return val * 64.0 
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
        if not line or line.startswith("#"):
            continue
        cols = line.split(",")
        if len(cols) < 3:
            continue
        raw, unit, event = cols[0].strip(), cols[1].strip(), cols[2].strip()
        if raw in ("<not supported>", "<not counted>"):
            continue
        try:
            val = float(raw)
        except ValueError:
            continue
        b = to_bytes(val, unit)
        if "cas_count_read" in event:
            r += b
        elif "cas_count_write" in event:
            w += b
    return r, w

def parse_kv(stdout: str, key: str):
    for line in stdout.splitlines():
        if line.startswith(key + " "):
            return line.split(None, 1)[1].strip()
    return None

def perf_noise(event_str: str, sleep_s: float, out_csv: Path):
    cmd = [
        "perf", "stat", "-a",
        "--no-big-num", "--no-scale",
        "-x", ",",
        "-e", event_str,
        "-o", str(out_csv),
        "--", "sleep", str(sleep_s)
    ]
    p = run(cmd, stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True)
    if p.returncode != 0:
        raise RuntimeError(p.stderr)
    txt = out_csv.read_text()
    rb, wb = parse_perf_stat_csv(txt)
    return (rb/sleep_s/1e9, wb/sleep_s/1e9, (rb+wb)/sleep_s/1e9)

def main():
    # === [关键修改] 生成默认 Block 列表 (512 .. 1M) ===
    # 2^9=512, 2^20=1048576
    default_blocks = ",".join(str(2**i) for i in range(9, 21))

    ap = argparse.ArgumentParser("Bandwidth-kernel amp grid using perf IMC + FIFO ROI gating")
    ap.add_argument("-o", "--outdir", default="./bw_runs")
    ap.add_argument("--repeats", type=int, default=5)
    ap.add_argument("--noise-s", type=float, default=0.1)
    ap.add_argument("--n", default="1000000000")
    
    # === [关键修改] 使用生成的默认值 ===
    ap.add_argument("--block", default=default_blocks, 
                    help=f"Block sizes (powers of 2). Default: {default_blocks}")
    
    # ap.add_argument("--kernels", default="std_copy,std_move,avx2,avx512,stream,stream_prefetch,stream_blocked,mimic_scatter,avx2_stream")
    ap.add_argument("--kernels", default="std_move,avx2,avx512,stream")
    ap.add_argument("--csv-name", default="grid_results.csv")
    args, rest = ap.parse_known_args()
    
    if rest and rest[0] == "--":
        rest = rest[1:]
    if not rest:
        print("Usage: run_bw_amp_grid.py [options] -- <base_cmd_exe_path>", file=sys.stderr)
        print("Example: python run_bw_amp_grid.py -o out -- -- numactl -i all ./bench_bandwidth", file=sys.stderr)
        sys.exit(1)
    base_cmd = rest

    out_root = Path(args.outdir)
    ts = time.strftime("%Y-%m-%d_%H_%M_%S")
    root = out_root / f"bw_amp_grid_{ts}"
    root.mkdir(parents=True, exist_ok=True)

    n_list = [int(x) for x in parse_list(args.n, int)]
    blk_list = [int(x) for x in parse_list(args.block, int)]
    kernels = [k.strip() for k in parse_list(args.kernels, str)]

    events = get_imc_events()
    event_str = ",".join(events)

    (root / "meta.txt").write_text(
        f"BASE_CMD: {' '.join(base_cmd)}\nEVENTS: {event_str}\n"
        f"N: {n_list}\nBLOCK: {blk_list}\nKERNELS: {kernels}\n"
        f"REPEATS: {args.repeats}\nNOISE_S: {args.noise_s}\n"
    )

    out_csv = root / args.csv_name
    with out_csv.open("w", newline="") as fcsv:
        w = csv.writer(fcsv)
        # Header
        w.writerow([
            "n", "block", "kernel", "rep",
            "roi_s", "type_bytes",
            "logical_rw_gb",   
            "imc_read_gb",     
            "imc_write_gb",    
            "amp_read",        
            "amp_write",       
            "amp_total",       
            "goodput_rw_gbps", 
            "imc_total_gbps",  
            "noise_total_gbps",
            "run_dir"
        ])

        total = len(n_list)*len(blk_list)*len(kernels)*args.repeats
        idx = 0
        for n, blk, ker in itertools.product(n_list, blk_list, kernels):
            for rep in range(args.repeats):
                idx += 1
                tag = f"n{n}_blk{blk}_{ker}_rep{rep}"
                run_dir = root / tag
                run_dir.mkdir(parents=True, exist_ok=True)

                # baseline noise
                noise_csv = run_dir / "noise.csv"
                try:
                    _, _, noise_t = perf_noise(event_str, args.noise_s, noise_csv)
                except Exception as e:
                    noise_t = float("nan")
                    (run_dir / "noise.err").write_text(str(e) + "\n")

                # FIFOs
                fifo_ctl = run_dir / "perf_ctl.fifo"
                fifo_ack = run_dir / "perf_ack.fifo"
                for pth in (fifo_ctl, fifo_ack):
                    try:
                        os.unlink(pth)
                    except FileNotFoundError:
                        pass
                os.mkfifo(fifo_ctl)
                os.mkfifo(fifo_ack)

                env = os.environ.copy()
                env["PERF_CTL_FIFO"] = str(fifo_ctl)
                env["PERF_ACK_FIFO"] = str(fifo_ack)

                perf_csv = run_dir / "perf_stat.csv"
                app_out = run_dir / "app_stdout.txt"
                app_err = run_dir / "app_stderr.txt"

                cmd = base_cmd + [str(n), str(blk), ker]

                perf_cmd = [
                    "perf", "stat", "-a",
                    "--no-big-num", "--no-scale",
                    "-x", ",",
                    "-e", event_str,
                    "-o", str(perf_csv),
                    "--delay=-1", "--control", f"fifo:{fifo_ctl},{fifo_ack}",
                    "--"
                ] + cmd

                p = run(perf_cmd, stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True, env=env)
                app_out.write_text(p.stdout)
                app_err.write_text(p.stderr)

                if p.returncode != 0:
                    (run_dir / "run.err").write_text(p.stderr + "\n")
                    print(f"[{idx}/{total}] FAIL {tag}")
                    continue

                roi_s = float(parse_kv(p.stdout, "ROI_SECONDS"))
                type_bytes = int(parse_kv(p.stdout, "TYPE_BYTES"))
                good_rw_gbps = float(parse_kv(p.stdout, "GOODPUT_RW_GBps"))

                rb, wb = parse_perf_stat_csv(perf_csv.read_text())
                
                logical_bytes_oneway = n * type_bytes
                logical_bytes_rw = logical_bytes_oneway * 2.0
                
                logical_rw_gb = logical_bytes_rw / 1e9
                imc_read_gb = rb / 1e9
                imc_write_gb = wb / 1e9

                if logical_bytes_oneway > 0:
                    amp_r = rb / logical_bytes_oneway
                    amp_w = wb / logical_bytes_oneway
                    amp_t = (rb + wb) / logical_bytes_rw
                else:
                    amp_r = amp_w = amp_t = float("nan")

                imc_total_gbps = (rb + wb) / roi_s / 1e9

                w.writerow([n, blk, ker, rep,
                            f"{roi_s:.9f}", type_bytes,
                            f"{logical_rw_gb:.6f}",    
                            f"{imc_read_gb:.6f}", 
                            f"{imc_write_gb:.6f}",
                            f"{amp_r:.3f}", 
                            f"{amp_w:.3f}", 
                            f"{amp_t:.3f}", 
                            f"{good_rw_gbps:.3f}",
                            f"{imc_total_gbps:.3f}",
                            f"{noise_t:.6f}",
                            str(run_dir)])

                print(f"[{idx}/{total}] OK {tag} "
                      f"LogRW:{logical_rw_gb:.2f}G IMC_R:{imc_read_gb:.2f}G({amp_r:.2f}x) "
                      f"IMC_W:{imc_write_gb:.2f}G({amp_w:.2f}x)")

    print(f"\nDONE. Results: {out_csv}")
    print(f"Root dir: {root}")

if __name__ == "__main__":
    main()