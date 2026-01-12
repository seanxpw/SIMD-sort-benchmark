#!/usr/bin/env python3
import argparse
import csv
import itertools
import os
import re
import subprocess
import sys
import time
from dataclasses import dataclass
from pathlib import Path
from typing import List, Tuple, Optional

def run(cmd, **kwargs):
    return subprocess.run(cmd, **kwargs)

def parse_list_arg(s: str, cast_fn):
    """
    Accept:
      - single value: "256"
      - comma-separated: "2,4,8,16"
      - space-separated (rare): "2 4 8"
    """
    if s is None:
        return []
    parts = []
    for token in re.split(r"[,\s]+", s.strip()):
        if token == "":
            continue
        parts.append(cast_fn(token))
    return parts

def get_imc_events() -> List[str]:
    """
    Prefer explicit uncore_imc_<id>/cas_count_{read,write}/ events (robust across machines).
    Fall back to imc/cas_count_{read,write}/ only if uncore_imc not found.
    """
    p = run(["perf", "list"], stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True)
    if p.returncode != 0:
        raise RuntimeError("perf list failed:\n" + p.stderr)

    # Lines look like:
    #   uncore_imc_0/cas_count_read/ [Kernel PMU event]
    pat = re.compile(r"^\s*(uncore_imc_(\d+))/cas_count_read/")
    bases: List[Tuple[int, str]] = []
    seen = set()

    for line in p.stdout.splitlines():
        m = pat.search(line)
        if m:
            imc_id = int(m.group(2))
            base = m.group(1)
            if base not in seen:
                seen.add(base)
                bases.append((imc_id, base))

    if bases:
        bases.sort(key=lambda x: x[0])
        events = []
        for _, b in bases:
            events.append(f"{b}/cas_count_read/")
            events.append(f"{b}/cas_count_write/")
        return events

    # fallback (some systems expose alias)
    return ["imc/cas_count_read/", "imc/cas_count_write/"]

def parse_roi_seconds(app_stdout: str) -> float:
    for line in app_stdout.splitlines():
        if line.startswith("ROI_SECONDS"):
            parts = line.split()
            if len(parts) >= 2:
                return float(parts[1])
    raise RuntimeError("ROI_SECONDS not found in app stdout")

def parse_perf_stat_csv(stat_text: str) -> Tuple[float, float]:
    """
    Parse perf stat -x , output and return (read_bytes, write_bytes).
    Handles two cases:
      - unit is KiB/MiB/GiB => treat as bytes-like and convert.
      - unit empty => raw CAS count => CAS*64B.
    """
    def to_bytes(val: float, unit: str) -> float:
        unit = (unit or "").strip()
        if unit == "":
            return val * 64.0  # CAS * 64B (common assumption)

        # KiB/MiB/GiB or KB/MB/GB
        m = re.match(r"^([KMGTP]?)(i)?B$", unit)
        if m:
            p, ib = m.group(1), m.group(2)
            base = 1024.0 if ib else 1000.0
            exp = {"":0, "K":1, "M":2, "G":3, "T":4, "P":5}[p]
            return val * (base ** exp)

        # Some perf builds may print "MiB" already (handled above), but keep safe fallback
        return val * 64.0

    read_bytes = 0.0
    write_bytes = 0.0

    for line in stat_text.splitlines():
        line = line.strip()
        if not line or line.startswith("#"):
            continue
        cols = line.split(",")
        if len(cols) < 3:
            continue

        raw = cols[0].strip()
        unit = cols[1].strip()
        event = cols[2].strip()

        if raw in ("<not supported>", "<not counted>"):
            continue

        try:
            val = float(raw)
        except ValueError:
            continue

        b = to_bytes(val, unit)
        if "cas_count_read" in event:
            read_bytes += b
        elif "cas_count_write" in event:
            write_bytes += b

    return read_bytes, write_bytes

@dataclass
class OneRunResult:
    n: int
    buckets: int
    block: int
    dist: str
    rep: int
    roi_s: float
    imc_read_bytes: float
    imc_write_bytes: float
    imc_read_gbps: float
    imc_write_gbps: float
    imc_total_gbps: float
    goodput_oneway_gbps: float
    goodput_rw_gbps: float
    amplification: float
    noise_read_gbps: float
    noise_write_gbps: float
    noise_total_gbps: float
    run_dir: str

def perf_stat_sleep(event_str: str, sleep_s: float, out_csv: Path) -> Tuple[float, float, float]:
    """
    Measure baseline DRAM traffic for sleep_s seconds (system-wide) and return (read_gbps, write_gbps, total_gbps).
    """
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
        raise RuntimeError("perf stat sleep failed:\n" + p.stderr)

    stat_text = out_csv.read_text()
    rbytes, wbytes = parse_perf_stat_csv(stat_text)

    r_gbps = rbytes / sleep_s / 1e9
    w_gbps = wbytes / sleep_s / 1e9
    t_gbps = (rbytes + wbytes) / sleep_s / 1e9
    return r_gbps, w_gbps, t_gbps

def run_one(event_str: str,
            base_cmd: List[str],
            n: int, buckets: int, block: int, dist: str,
            use_fifo: bool,
            run_dir: Path) -> Tuple[str, str, str]:
    """
    Run perf stat around the app. Returns (app_stdout, app_stderr, perf_stat_csv_text).
    """
    app_out_path = run_dir / "app_stdout.txt"
    app_err_path = run_dir / "app_stderr.txt"
    stat_out_path = run_dir / "perf_stat.csv"

    env = os.environ.copy()

    fifo_ctl = None
    fifo_ack = None
    perf_cmd = [
        "perf", "stat", "-a",
        "--no-big-num", "--no-scale",
        "-x", ",",
        "-e", event_str,
        "-o", str(stat_out_path),
    ]

    if use_fifo:
        fifo_ctl = run_dir / "perf_ctl.fifo"
        fifo_ack = run_dir / "perf_ack.fifo"
        for f in (fifo_ctl, fifo_ack):
            try:
                os.unlink(f)
            except FileNotFoundError:
                pass
        os.mkfifo(fifo_ctl)
        os.mkfifo(fifo_ack)
        env["PERF_CTL_FIFO"] = str(fifo_ctl)
        env["PERF_ACK_FIFO"] = str(fifo_ack)
        perf_cmd += ["--delay=-1", "--control", f"fifo:{fifo_ctl},{fifo_ack}"]

    # Append the 4 positional args expected by your C++ main:
    cmd = base_cmd + [str(n), str(buckets), str(block), dist]

    perf_cmd += ["--"] + cmd

    p = run(perf_cmd, stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True, env=env)
    app_out_path.write_text(p.stdout)
    app_err_path.write_text(p.stderr)

    if p.returncode != 0:
        raise RuntimeError(f"Command failed (see {app_err_path}):\n{p.stderr}")

    return p.stdout, p.stderr, stat_out_path.read_text()

def main():
    ap = argparse.ArgumentParser(
        description="Grid-sweep IMC DRAM bandwidth via perf stat (uncore_imc cas_count_{read,write}) with optional ROI FIFO gating."
    )
    ap.add_argument("-o", "--outdir", default="./bw_runs", help="Output root directory")
    ap.add_argument("--no-fifo", action="store_true", help="Disable perf --control FIFO gating (measures whole program)")
    ap.add_argument("--repeats", type=int, default=3, help="Repeats per configuration")
    ap.add_argument("--noise-s", type=float, default=0.2, help="Baseline noise measurement duration (seconds) before each run")
    ap.add_argument("--n", default="1000000000", help="N list (e.g. '1e9' not allowed; use int; comma-separated)")
    ap.add_argument("--buckets", default="256", help="Buckets list (comma-separated)")
    ap.add_argument("--block", default="262144", help="BlockSize list in ELEMENTS (comma-separated)")
    ap.add_argument("--dist", default="uniform", help="Dist list: uniform,sorted,reverse (comma-separated)")
    ap.add_argument("--csv-name", default="grid_results.csv", help="Summary CSV file name")
    # parse_known_args: remaining tokens are the base command
    args, rest = ap.parse_known_args()
    if rest and rest[0] == "--":
        rest = rest[1:]
    if not rest:
        print("Usage: run_imc_bw.py [grid options...] -- <base_command_without_4_params>", file=sys.stderr)
        print("Example: python run_imc_bw.py -o out --buckets 2,256 -- -- numactl -i all ./build/bench_scatter_parlay", file=sys.stderr)
        sys.exit(1)

    base_cmd = rest

    out_root = Path(args.outdir)
    ts = time.strftime("%Y-%m-%d_%H_%M_%S")
    root = out_root / f"imc_grid_{ts}"
    root.mkdir(parents=True, exist_ok=True)

    # parse lists
    n_list = parse_list_arg(args.n, int)
    b_list = parse_list_arg(args.buckets, int)
    blk_list = parse_list_arg(args.block, int)
    dist_list = parse_list_arg(args.dist, str)

    # normalize dist tokens
    dist_list = [d.strip().lower() for d in dist_list]
    for d in dist_list:
        if d not in ("uniform", "sorted", "reverse"):
            raise ValueError(f"Unknown dist '{d}'. Use uniform,sorted,reverse.")

    # events
    events = get_imc_events()
    event_str = ",".join(events)

    # write a top-level metadata file
    (root / "meta.txt").write_text(
        f"BASE_CMD: {' '.join(base_cmd)}\n"
        f"EVENTS: {event_str}\n"
        f"N_LIST: {n_list}\nBUCKETS_LIST: {b_list}\nBLOCK_LIST: {blk_list}\nDIST_LIST: {dist_list}\n"
        f"REPEATS: {args.repeats}\nNOISE_S: {args.noise_s}\nFIFO: {not args.no_fifo}\n"
    )

    results: List[OneRunResult] = []

    # optional: measure a global baseline once at start
    global_noise_csv = root / "noise_global_start.csv"
    try:
        gnr, gnw, gnt = perf_stat_sleep(event_str, args.noise_s, global_noise_csv)
    except Exception as e:
        # If sleep noise fails, continue but record NaNs
        gnr = gnw = gnt = float("nan")
        (root / "noise_global_start.err").write_text(str(e) + "\n")

    cfgs = list(itertools.product(n_list, b_list, blk_list, dist_list))
    total_runs = len(cfgs) * args.repeats
    print(f"Grid configs: {len(cfgs)}; total runs: {total_runs}")
    print(f"Using events: {event_str}")

    run_idx = 0
    for (n, buckets, block, dist) in cfgs:
        for rep in range(args.repeats):
            run_idx += 1
            tag = f"n{n}_b{buckets}_blk{block}_{dist}_rep{rep}"
            run_dir = root / tag
            run_dir.mkdir(parents=True, exist_ok=True)

            # noise measurement before each run (so you can see drift)
            noise_csv = run_dir / "noise.csv"
            try:
                nr, nw, nt = perf_stat_sleep(event_str, args.noise_s, noise_csv)
            except Exception as e:
                nr = nw = nt = float("nan")
                (run_dir / "noise.err").write_text(str(e) + "\n")

            # actual run
            try:
                app_out, app_err, stat_text = run_one(
                    event_str=event_str,
                    base_cmd=base_cmd,
                    n=n, buckets=buckets, block=block, dist=dist,
                    use_fifo=(not args.no_fifo),
                    run_dir=run_dir,
                )
            except Exception as e:
                (run_dir / "run.err").write_text(str(e) + "\n")
                print(f"[{run_idx}/{total_runs}] FAILED {tag}: {e}")
                continue

            # parse
            try:
                roi_s = parse_roi_seconds(app_out)
            except Exception as e:
                (run_dir / "parse.err").write_text(str(e) + "\n")
                print(f"[{run_idx}/{total_runs}] PARSE FAILED {tag}: {e}")
                continue

            rbytes, wbytes = parse_perf_stat_csv(stat_text)

            imc_r_gbps = rbytes / roi_s / 1e9
            imc_w_gbps = wbytes / roi_s / 1e9
            imc_t_gbps = (rbytes + wbytes) / roi_s / 1e9

            # goodput (T assumed 4 bytes; if you use different type, adjust here)
            elem_bytes = 4.0
            good_oneway_gbps = (n * elem_bytes) / roi_s / 1e9
            good_rw_gbps = (2.0 * n * elem_bytes) / roi_s / 1e9

            # amplification vs ideal RW traffic
            ideal_bytes = 2.0 * n * elem_bytes
            amplification = (rbytes + wbytes) / ideal_bytes if ideal_bytes > 0 else float("nan")

            results.append(OneRunResult(
                n=n, buckets=buckets, block=block, dist=dist, rep=rep,
                roi_s=roi_s,
                imc_read_bytes=rbytes,
                imc_write_bytes=wbytes,
                imc_read_gbps=imc_r_gbps,
                imc_write_gbps=imc_w_gbps,
                imc_total_gbps=imc_t_gbps,
                goodput_oneway_gbps=good_oneway_gbps,
                goodput_rw_gbps=good_rw_gbps,
                amplification=amplification,
                noise_read_gbps=nr,
                noise_write_gbps=nw,
                noise_total_gbps=nt,
                run_dir=str(run_dir),
            ))

            # per-run quick summary
            (run_dir / "summary.txt").write_text(
                f"RUN_DIR: {run_dir}\n"
                f"CMD: {' '.join(base_cmd + [str(n), str(buckets), str(block), dist])}\n"
                f"ROI_SECONDS {roi_s:.9f}\n"
                f"IMC_READ_BYTES {rbytes:.0f}\nIMC_WRITE_BYTES {wbytes:.0f}\n"
                f"IMC_BW_READ_GBps {imc_r_gbps:.3f}\nIMC_BW_WRITE_GBps {imc_w_gbps:.3f}\nIMC_BW_TOTAL_GBps {imc_t_gbps:.3f}\n"
                f"GOODPUT_ONEWAY_GBps {good_oneway_gbps:.3f}\nGOODPUT_RW_GBps {good_rw_gbps:.3f}\n"
                f"AMPLIFICATION {amplification:.3f}\n"
                f"NOISE_BW_TOTAL_GBps {nt:.6f}\n"
                f"GLOBAL_NOISE_START_TOTAL_GBps {gnt:.6f}\n"
            )

            print(f"[{run_idx}/{total_runs}] OK {tag} "
                  f"ROI={roi_s:.4f}s IMC={imc_t_gbps:.1f}GB/s "
                  f"goodRW={good_rw_gbps:.1f}GB/s amp={amplification:.2f} "
                  f"noise={nt:.3f}GB/s")

    # write summary CSV
    out_csv = root / args.csv_name
    with out_csv.open("w", newline="") as f:
        w = csv.writer(f)
        w.writerow([
            "n", "buckets", "block", "dist", "rep",
            "roi_s",
            "imc_read_gbps", "imc_write_gbps", "imc_total_gbps",
            "goodput_oneway_gbps", "goodput_rw_gbps",
            "amplification",
            "noise_read_gbps", "noise_write_gbps", "noise_total_gbps",
            "run_dir",
        ])
        for r in results:
            w.writerow([
                r.n, r.buckets, r.block, r.dist, r.rep,
                f"{r.roi_s:.9f}",
                f"{r.imc_read_gbps:.6f}", f"{r.imc_write_gbps:.6f}", f"{r.imc_total_gbps:.6f}",
                f"{r.goodput_oneway_gbps:.6f}", f"{r.goodput_rw_gbps:.6f}",
                f"{r.amplification:.6f}",
                f"{r.noise_read_gbps:.6f}", f"{r.noise_write_gbps:.6f}", f"{r.noise_total_gbps:.6f}",
                r.run_dir
            ])

    print(f"\nDONE. Results: {out_csv}")
    print(f"Root dir: {root}")

if __name__ == "__main__":
    main()
