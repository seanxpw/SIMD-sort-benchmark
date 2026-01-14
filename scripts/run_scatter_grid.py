#!/usr/bin/env python3
import argparse, csv, itertools, os, re, subprocess, sys, time
from pathlib import Path

# =========================================================
# Utilities
# =========================================================
def run(cmd, **kwargs):
    return subprocess.run(cmd, **kwargs)

def parse_list(s, cast=str):
    parts = []
    for tok in re.split(r"[,\s]+", (s or "").strip()):
        if tok:
            parts.append(cast(tok))
    return parts

def pow2_range(min_v: int, max_v: int):
    """Return [min_v..max_v] in powers of two, inclusive, min_v must be power-of-two."""
    out = []
    x = min_v
    while x <= max_v:
        out.append(x)
        x <<= 1
    return out

def get_imc_events():
    # Auto-detect perf uncore IMC events; fallback to generic names
    p = run(["perf", "list"], stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True)
    if p.returncode != 0:
        return ["imc/cas_count_read/", "imc/cas_count_write/"]

    pat = re.compile(r"^\s*(uncore_imc_(\d+))/cas_count_read/")
    bases, seen = [], set()
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
    """
    Parse perf stat -x, CSV. For imc cas_count_* we convert counts to bytes with 64B per CAS line as an approximation.
    """
    def to_bytes(val: float, unit: str) -> float:
        unit = (unit or "").strip()
        if unit == "":  # counts -> assume 64B cacheline
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
        except:
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
        "--no-big-num", "--no-scale", "-x", ",",
        "-e", event_str,
        "-o", str(out_csv),
        "--", "sleep", str(sleep_s)
    ]
    run(cmd, stdout=subprocess.PIPE, stderr=subprocess.PIPE)
    rb, wb = parse_perf_stat_csv(out_csv.read_text())
    return (rb + wb) / sleep_s / 1e9

# =========================================================
# Scatter Grid Runner
# =========================================================
def main():
    ap = argparse.ArgumentParser("Scatter Grid (buckets/tile defaults + pruning)")

    ap.add_argument("-o", "--outdir", default="./scatter_results")
    ap.add_argument("--csv-name", default="scatter_grid.csv")
    ap.add_argument("--repeats", type=int, default=3)
    ap.add_argument("--noise-s", type=float, default=0.2)

    # Core dims
    ap.add_argument("--n", default="1000000000",
                    help="N list, e.g. 1e9 or 1000000000,2000000000")
    ap.add_argument("--dist", default="uniform",
                    help="dist list: uniform,sorted,reverse (comma/space separated)")
    ap.add_argument("--modes", default="std_move,avx512_galloping_aligned",
                    help="mode list: std_move,avx512_galloping_aligned,...")

    # Buckets/tile with smart defaults
    ap.add_argument("--buckets", default="__default__",
                    help="bucket list; default is powers of two: 2..4096")
    ap.add_argument("--tile", default="__default__",
                    help="tile(block_size) list; default is powers of two: 512..1048576")

    # Prune rule
    ap.add_argument("--min-tile-per-bucket", type=int, default=2,
                    help="Skip if tile < (min_tile_per_bucket * buckets). Default=2.")

    # Perf event selection
    ap.add_argument("--events", default="__auto__",
                    help="perf events string; default auto-detect IMC cas_count_*")
    ap.add_argument("--no-perf", action="store_true",
                    help="Run without perf stat (disables IMC/amp metrics, still logs ROI & goodput).")

    args, rest = ap.parse_known_args()
    if rest and rest[0] == "--":
        rest = rest[1:]
    if not rest:
        print("Usage: python run_scatter_grid.py [options] -- <cmd_path>", file=sys.stderr)
        sys.exit(1)
    base_cmd = rest

    out_root = Path(args.outdir)
    ts = time.strftime("%Y-%m-%d_%H_%M_%S")
    root = out_root / f"scatter_grid_{ts}"
    root.mkdir(parents=True, exist_ok=True)

    # Parse N list (allow 1e9 style)
    n_list = []
    for tok in parse_list(args.n, str):
        try:
            n_list.append(int(float(tok)))
        except:
            n_list.append(int(tok))

    dist_list = parse_list(args.dist, str)
    mode_list = parse_list(args.modes, str)

    if args.buckets == "__default__":
        bkt_list = pow2_range(2, 4096)
    else:
        bkt_list = [int(x) for x in parse_list(args.buckets, int)]

    if args.tile == "__default__":
        # Reasonable default sweep: 512 .. 1,048,576 elements (2^9 .. 2^20)
        tile_list = pow2_range(512, 1_048_576)
    else:
        tile_list = [int(x) for x in parse_list(args.tile, int)]

    # Events
    if args.no_perf:
        events = []
        event_str = ""
    else:
        if args.events == "__auto__":
            events = get_imc_events()
            event_str = ",".join(events)
        else:
            event_str = args.events

    out_csv = root / args.csv_name
    with out_csv.open("w", newline="") as fcsv:
        w = csv.writer(fcsv)
        w.writerow([
            "n", "buckets", "tile", "dist", "mode", "rep",
            "roi_s", "type_bytes",
            "logical_oneway_gb", "logical_rw_gb",
            "goodput_oneway_gbps", "goodput_rw_gbps",
            "imc_read_gb", "imc_write_gb", "imc_total_gbps",
            "amp_read", "amp_write", "amp_total",
            "noise_total_gbps",
            "returncode",
            "run_dir"
        ])

        total = len(n_list) * len(bkt_list) * len(tile_list) * len(dist_list) * len(mode_list) * args.repeats
        idx = 0
        skipped = 0

        for n, bkt, tile, dist_name, mode in itertools.product(n_list, bkt_list, tile_list, dist_list, mode_list):
            # prune: tile must be >= min_tile_per_bucket * buckets
            if tile < args.min_tile_per_bucket * bkt:
                skipped += args.repeats
                continue

            for rep in range(args.repeats):
                idx += 1
                tag = f"n{n}_bkt{bkt}_tile{tile}_{dist_name}_{mode}_rep{rep}"
                run_dir = root / tag
                run_dir.mkdir(parents=True, exist_ok=True)

                noise_val = 0.0
                if (not args.no_perf) and args.noise_s > 0:
                    try:
                        noise_val = perf_noise(event_str, args.noise_s, run_dir / "noise.csv")
                    except:
                        noise_val = 0.0

                # Build command
                cmd_args = [str(n), str(bkt), str(tile), dist_name, mode]

                # Run with or without perf
                if args.no_perf:
                    full_cmd = base_cmd + cmd_args
                    p = run(full_cmd, stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True, env=os.environ.copy())
                    (run_dir / "perf_stat.csv").write_text("")  # placeholder
                else:
                    fifo_ctl = run_dir / "perf_ctl.fifo"
                    fifo_ack = run_dir / "perf_ack.fifo"
                    for pp in (fifo_ctl, fifo_ack):
                        if pp.exists():
                            pp.unlink()
                    os.mkfifo(fifo_ctl)
                    os.mkfifo(fifo_ack)

                    env = os.environ.copy()
                    env["PERF_CTL_FIFO"] = str(fifo_ctl)
                    env["PERF_ACK_FIFO"] = str(fifo_ack)

                    full_cmd = [
                        "perf", "stat", "-a",
                        "--no-big-num", "--no-scale", "-x", ",",
                        "-e", event_str,
                        "-o", str(run_dir / "perf_stat.csv"),
                        "--delay=-1", "--control", f"fifo:{fifo_ctl},{fifo_ack}",
                        "--"
                    ] + base_cmd + cmd_args

                    p = run(full_cmd, stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True, env=env)

                (run_dir / "stdout.txt").write_text(p.stdout)
                (run_dir / "stderr.txt").write_text(p.stderr)

                # Parse C++ keys
                roi_s = parse_kv(p.stdout, "ROI_SECONDS")
                type_bytes = parse_kv(p.stdout, "TYPE_BYTES")
                gp1 = parse_kv(p.stdout, "GOODPUT_ONEWAY_GBps")
                gprw = parse_kv(p.stdout, "GOODPUT_RW_GBps")

                # Defaults if parsing fails
                roi_s_f = float(roi_s) if roi_s is not None else float("nan")
                tb_i = int(type_bytes) if type_bytes is not None else -1
                gp1_f = float(gp1) if gp1 is not None else float("nan")
                gprw_f = float(gprw) if gprw is not None else float("nan")

                # Compute logical GB (both one-way and RW)
                if tb_i > 0:
                    logical_oneway_gb = (float(n) * tb_i) / 1e9
                    logical_rw_gb = 2.0 * logical_oneway_gb
                else:
                    logical_oneway_gb = float("nan")
                    logical_rw_gb = float("nan")

                # IMC metrics
                if args.no_perf or p.returncode != 0:
                    imc_read_gb = imc_write_gb = imc_total_gbps = float("nan")
                    amp_r = amp_w = amp_t = float("nan")
                else:
                    try:
                        rb, wb = parse_perf_stat_csv((run_dir / "perf_stat.csv").read_text())
                        imc_read_gb = rb / 1e9
                        imc_write_gb = wb / 1e9
                        imc_total_gbps = (rb + wb) / roi_s_f / 1e9 if roi_s_f == roi_s_f else float("nan")

                        if tb_i > 0:
                            logical_oneway_bytes = float(n) * tb_i
                            logical_rw_bytes = 2.0 * logical_oneway_bytes
                            amp_r = rb / logical_oneway_bytes
                            amp_w = wb / logical_oneway_bytes
                            amp_t = (rb + wb) / logical_rw_bytes
                        else:
                            amp_r = amp_w = amp_t = float("nan")
                    except:
                        imc_read_gb = imc_write_gb = imc_total_gbps = float("nan")
                        amp_r = amp_w = amp_t = float("nan")

                w.writerow([
                    n, bkt, tile, dist_name, mode, rep,
                    f"{roi_s_f:.6f}" if roi_s_f == roi_s_f else "",
                    tb_i if tb_i > 0 else "",
                    f"{logical_oneway_gb:.6f}" if logical_oneway_gb == logical_oneway_gb else "",
                    f"{logical_rw_gb:.6f}" if logical_rw_gb == logical_rw_gb else "",
                    f"{gp1_f:.6f}" if gp1_f == gp1_f else "",
                    f"{gprw_f:.6f}" if gprw_f == gprw_f else "",
                    f"{imc_read_gb:.6f}" if imc_read_gb == imc_read_gb else "",
                    f"{imc_write_gb:.6f}" if imc_write_gb == imc_write_gb else "",
                    f"{imc_total_gbps:.6f}" if imc_total_gbps == imc_total_gbps else "",
                    f"{amp_r:.3f}" if amp_r == amp_r else "",
                    f"{amp_w:.3f}" if amp_w == amp_w else "",
                    f"{amp_t:.3f}" if amp_t == amp_t else "",
                    f"{noise_val:.6f}" if noise_val == noise_val else "",
                    p.returncode,
                    str(run_dir)
                ])

                status = "OK" if p.returncode == 0 else "FAIL"
                print(f"[{idx}/{total}] {status} n={n} bkt={bkt} tile={tile} dist={dist_name} mode={mode} rep={rep} "
                      f"roi={roi_s_f if roi_s_f==roi_s_f else 'NA'}")

        print(f"\nDONE. Results: {out_csv}")
        if skipped:
            print(f"Skipped groups (tile < {args.min_tile_per_bucket}*buckets): {skipped}")

if __name__ == "__main__":
    main()
