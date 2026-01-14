import argparse
import subprocess
import sys
import os
import re
import socket
import csv
import math
from pathlib import Path

import matplotlib.pyplot as plt


def run(cmd):
    p = subprocess.run(cmd, stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True)
    if p.returncode != 0:
        sys.stderr.write("Command failed:\n")
        sys.stderr.write("STDERR:\n" + p.stderr + "\n")
        sys.stderr.write("STDOUT:\n" + p.stdout + "\n")
        raise SystemExit(p.returncode)
    return p.stdout, p.stderr


def identify_cpu_architecture(vendor_id, model_name):
    arch = "Unknown Arch"
    year = "????"

    # AMD
    if "AMD" in vendor_id or "AMD" in model_name:
        if "EPYC" in model_name:
            match = re.search(r"EPYC (\d{4})", model_name)
            if match:
                model_num = int(match.group(1))
                last_digit = model_num % 10
                if last_digit == 1: return "Zen 1 (Naples)", "2017"
                if last_digit == 2: return "Zen 2 (Rome)", "2019"
                if last_digit == 3: return "Zen 3 (Milan)", "2021"
                if last_digit == 4: return "Zen 4 (Genoa)", "2022"
                if last_digit == 5: return "Zen 5 (Turin)", "2024"
            if "7551" in model_name: return "Zen 1 (Naples)", "2017"

        if "Ryzen" in model_name:
            if "79" in model_name or "77" in model_name: return "Zen 4/5 (Consumer)", "2022+"
            return "Zen 1/2/3 (Consumer)", "Pre-2022"

    # Intel (heuristic)
    else:
        match = re.search(r" (\d{4}[A-Z]?)", model_name)
        if match:
            try:
                model_num_str = match.group(1)
                gen_code = int(model_num_str[:2])
                if gen_code in [85, 65, 55, 45, 35]: return "Emerald Rapids (Gen 5)", "2023/24"
                if gen_code in [84, 64, 54, 44, 34]: return "Sapphire Rapids (Gen 4)", "2023"
                if gen_code in [83, 63, 53, 43]: return "Ice Lake (Gen 3)", "2021"
                if gen_code in [82, 62, 52, 42, 32]: return "Cascade Lake (Gen 2)", "2019"
                if gen_code in [81, 61, 51, 41, 31]: return "Skylake-SP (Gen 1)", "2017"
            except:
                pass
        if "12900" in model_name: return "Alder Lake", "2021"
        if "13900" in model_name: return "Raptor Lake", "2022"

    return arch, year


def get_cpu_info():
    info = {
        "model": "Unknown CPU", "vendor": "",
        "l1d": 0, "l2": 0, "l3": 0,
        "arch": "", "year": "",
        "raw_lscpu": ""
    }

    try:
        info["raw_lscpu"] = subprocess.check_output("lscpu", shell=True, universal_newlines=True)
        for line in info["raw_lscpu"].splitlines():
            if "Model name:" in line:
                info["model"] = line.split(":", 1)[1].strip()
            if "Vendor ID:" in line:
                info["vendor"] = line.split(":", 1)[1].strip()
    except:
        info["raw_lscpu"] = "Error capturing lscpu"

    info["arch"], info["year"] = identify_cpu_architecture(info["vendor"], info["model"])

    def read_cache_size(index):
        try:
            path = f"/sys/devices/system/cpu/cpu0/cache/index{index}/size"
            if not os.path.exists(path):
                return 0
            with open(path, "r") as f:
                val = f.read().strip()
            if val.endswith("K"): return int(val[:-1]) * 1024
            if val.endswith("M"): return int(val[:-1]) * 1024 * 1024
            return int(val)
        except:
            return 0

    # keep your mapping convention
    info["l1d"] = read_cache_size(0)
    info["l2"]  = read_cache_size(2)
    info["l3"]  = read_cache_size(3)

    # Fallback (conservative)
    if info["l1d"] == 0:
        if "Zen" in info["arch"]:
            info["l1d"] = 32 * 1024
        elif "Ice Lake" in info["arch"] or "Sapphire" in info["arch"]:
            info["l1d"] = 48 * 1024
        else:
            info["l1d"] = 32 * 1024

    if info["l2"] == 0:
        info["l2"] = 1024 * 1024
    if info["l3"] == 0:
        info["l3"] = 16 * 1024 * 1024

    return info


def human_bytes(n):
    if n >= 1024 * 1024 * 1024:
        return f"{n / (1024**3):.2f} GiB"
    if n >= 1024 * 1024:
        return f"{n / (1024**2):.2f} MiB"
    if n >= 1024:
        return f"{n / 1024:.0f} KiB"
    return f"{n} B"


def add_suffix(path: Path, suffix: str) -> Path:
    return path.with_name(f"{path.stem}_{suffix}{path.suffix}")


def read_csv_for_plot(csv_path: Path):
    """
    Returns:
      dtype (str), elem_size (int), block_sizes(list[int]), thr_gib_s(list[float])
    """
    rows = []
    with open(csv_path, "r", newline="") as f:
        r = csv.DictReader(f)
        for row in r:
            rows.append(row)
    if not rows:
        return None

    dtype = rows[0]["dtype"]
    elem_size = int(rows[0]["elem_size"])
    block_sizes = [int(r["block_size"]) for r in rows]
    thr_gib_s = [float(r["throughput_GiB_s"]) for r in rows]
    return dtype, elem_size, block_sizes, thr_gib_s


def log_mid(a, b):
    # geometric mean in log space; handle edge cases
    if a <= 0 or b <= 0:
        return (a + b) / 2.0
    return math.exp((math.log(a) + math.log(b)) / 2.0)

def plot_4up(results_by_dtype, cpu_info, out_png: Path):
    """
    2x2 subplot:
      int32 | float
      int64 | double

    X-axis: block_size (elements), log2 scale with ticks formatted as 2^k (like your single-plot).
    Y-axis: throughput_GiB_s (GiB/s).

    Fixes the previous issue where axvspan(0, ...) on log axis broke tick formatting and squeezed curves.
    """
    import matplotlib.ticker as mticker

    host = socket.gethostname()

    # ---- helpers (local) ----
    def _pow2_ticks(min_x, max_x):
        pmin = int(math.floor(math.log2(min_x)))
        pmax = int(math.ceil(math.log2(max_x)))
        return [2 ** p for p in range(pmin, pmax + 1)]

    def apply_pow2_xaxis(ax, xs):
        min_x = min(xs)
        max_x = max(xs)

        ax.set_xscale("log", base=2)
        ax.set_xlim(min_x, max_x)

        ticks = _pow2_ticks(min_x, max_x)
        ax.set_xticks(ticks)

        def _fmt(v, pos):
            if v <= 0:
                return ""
            k = math.log2(v)
            if abs(k - round(k)) < 1e-10:
                return rf"$2^{{{int(round(k))}}}$"
            return ""

        ax.xaxis.set_major_formatter(mticker.FuncFormatter(_fmt))
        ax.xaxis.set_minor_locator(mticker.NullLocator())  # critical: avoid clutter / squeeze

    # ---- figure ----
    fig, axs = plt.subplots(2, 2, figsize=(16, 12))

    title_str = (
        "x86simdsortStatic::qsort Block Benchmark (GiB/s)\n"
        f"{cpu_info['model']}\n"
        f"[{cpu_info['arch']} - {cpu_info['year']}]   Host: {host}"
    )
    fig.suptitle(title_str, fontsize=16, fontweight="bold")

    layout = [
        ("int32",  axs[0, 0]),
        ("float",  axs[0, 1]),
        ("int64",  axs[1, 0]),
        ("double", axs[1, 1]),
    ]

    # Cache region colors (span fills)
    span_l1 = ("#cccccc", 0.20)   # L1: gray
    span_l2 = ("#2ca02c", 0.15)   # L2: green
    span_l3 = ("#ff7f0e", 0.15)   # L3: orange
    span_mem = ("#d62728", 0.10)  # beyond: red

    # Cache boundary line colors (distinct)
    line_l1 = "red"
    line_l2 = "orange"
    line_l3 = "green"

    props = dict(boxstyle="round,pad=0.3", facecolor="white", alpha=0.9, edgecolor="gray")

    for dtype, ax in layout:
        if dtype not in results_by_dtype:
            ax.set_visible(False)
            continue

        elem_size = results_by_dtype[dtype]["elem_size"]
        xs = results_by_dtype[dtype]["block_sizes"]   # elements
        ys = results_by_dtype[dtype]["thr_gib_s"]     # GiB/s

        if not xs:
            ax.set_visible(False)
            continue

        # curve
        ax.plot(xs, ys, marker="o", markersize=4, linewidth=2)
        apply_pow2_xaxis(ax, xs)
        ax.set_title(f"{dtype} ({elem_size} Bytes)", fontsize=13, fontweight="bold")
        ax.set_xlabel("Block size (elements)")
        ax.set_ylabel("Throughput (GiB/s)")
        ax.grid(True, which="major", alpha=0.3)

        min_x = min(xs)
        max_x = max(xs)

        # cache thresholds in elements for this dtype
        l1_count = cpu_info["l1d"] / elem_size
        l2_count = cpu_info["l2"]  / elem_size
        l3_count = cpu_info["l3"]  / elem_size

        # clamp span boundaries into [min_x, max_x] for clean visuals
        b1 = max(min_x, min(l1_count, max_x))
        b2 = max(min_x, min(l2_count, max_x))
        b3 = max(min_x, min(l3_count, max_x))

        # spans (IMPORTANT: never start from 0 on log axis)
        if b1 > min_x:
            ax.axvspan(min_x, b1, color=span_l1[0], alpha=span_l1[1])
        if b2 > b1:
            ax.axvspan(b1, b2, color=span_l2[0], alpha=span_l2[1])
        if b3 > b2:
            ax.axvspan(b2, b3, color=span_l3[0], alpha=span_l3[1])
        if max_x > b3:
            ax.axvspan(b3, max_x, color=span_mem[0], alpha=span_mem[1])

        # boundary lines (different colors)
        ax.axvline(l1_count, color=line_l1, linestyle="--", alpha=0.8, linewidth=1.2)
        ax.axvline(l2_count, color=line_l2, linestyle="--", alpha=0.8, linewidth=1.2)
        ax.axvline(l3_count, color=line_l3, linestyle="--", alpha=0.8, linewidth=1.2)

        # cache label boxes near top (axis transform)
        trans = ax.get_xaxis_transform()

        def log_mid(a, b):
            if a <= 0 or b <= 0:
                return (a + b) / 2.0
            return math.exp((math.log(a) + math.log(b)) / 2.0)

        # mids in log space, clamped to plotting range
        l1_mid = log_mid(min_x, b1) if b1 > min_x else min_x
        l2_mid = log_mid(b1, b2) if b2 > b1 else b1
        l3_mid = log_mid(b2, b3) if b3 > b2 else b2

        ax.text(l1_mid, 0.94, "L1", transform=trans, ha="center", va="top",
                bbox=props, fontsize=9, color="gray")
        ax.text(l2_mid, 0.94, "L2", transform=trans, ha="center", va="top",
                bbox=props, fontsize=9, color="#2ca02c", fontweight="bold")
        ax.text(l3_mid, 0.94, "L3", transform=trans, ha="center", va="top",
                bbox=props, fontsize=9, color="#d35400")

        # peak annotation
        max_tp = max(ys)
        peak_x = xs[ys.index(max_tp)]
        ax.annotate(
            f"Peak: {max_tp:.2f} GiB/s\n@ {peak_x} el",
            xy=(peak_x, max_tp),
            xytext=(peak_x, max_tp * 1.05),
            arrowprops=dict(facecolor="black", arrowstyle="->", alpha=0.8),
            ha="center",
            fontweight="bold",
            zorder=10
        )

        # per-panel small box: cache thresholds numeric (elements)
        ax.text(
            0.02, 0.02,
            f"L1~{int(l1_count)} el | L2~{int(l2_count)} el | L3~{int(l3_count)} el",
            transform=ax.transAxes, fontsize=9, va="bottom", ha="left",
            bbox=dict(boxstyle="round,pad=0.25", facecolor="white", alpha=0.85, edgecolor="gray")
        )

    # Global system info box (keep your detailed annotation)
    sys_box = (
        "HOST SYSTEM INFORMATION\n"
        f"Host: {host}\n"
        f"CPU: {cpu_info['model']}\n"
        f"Architecture: {cpu_info['arch']} ({cpu_info['year']})\n"
        f"L1d: {human_bytes(cpu_info['l1d'])} | L2: {human_bytes(cpu_info['l2'])} | L3: {human_bytes(cpu_info['l3'])}\n"
        "Y-axis: throughput_GiB_s (GiB/s); X-axis: block_size (elements)"
    )
    fig.text(
        0.02, 0.02, sys_box,
        fontsize=10, va="bottom", ha="left",
        bbox=dict(boxstyle="round,pad=0.4", facecolor="white", alpha=0.92, edgecolor="gray")
    )

    plt.tight_layout(rect=[0, 0.08, 1, 0.90])
    out_png = Path(out_png)
    out_png.parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(out_png, dpi=300)
    plt.close()



def _toggle_alpha():
    # Slightly higher opacity for readability on dense figures
    return 0.92


def main():
    # Defaults per your request
    default_bin = "build/bench_x86simdsort_blocks"
    default_out = "results/x86simdsort_blocks.csv"
    default_png = "results/x86simdsort_blocks_thr.png"

    ap = argparse.ArgumentParser()
    ap.add_argument("--bin", default=default_bin,
                    help=f"path to C++ benchmark binary (default: {default_bin})")
    ap.add_argument("--out", default=default_out,
                    help=f"output csv base path (default: {default_out})")
    ap.add_argument("--plot_out", default=default_png,
                    help=f"output png path (default: {default_png})")
    ap.add_argument("--plot", type=int, default=1,
                    help="whether to generate plot (1=on, 0=off). default=1")

    # NEW: plot-only mode
    ap.add_argument("--plot_only", type=int, default=0,
                    help="1 = only plot from existing CSVs (skip running benchmark). default=0")

    ap.add_argument("--n", type=int, default=100000000)
    ap.add_argument("--min_block", type=int, default=8)
    ap.add_argument("--max_block", type=int, default=10000000)
    ap.add_argument("--descending", type=int, default=0)
    ap.add_argument("--check", type=int, default=0)
    ap.add_argument("--seed", type=int, default=12345)

    args = ap.parse_args()

    bin_path = Path(args.bin).expanduser().resolve()
    out_base = Path(args.out).expanduser().resolve()
    png_path = Path(args.plot_out).expanduser().resolve()

    out_base.parent.mkdir(parents=True, exist_ok=True)
    png_path.parent.mkdir(parents=True, exist_ok=True)

    cpu_info = get_cpu_info()

    # dtype order matches subplot layout
    dtypes = ["int32", "float", "int64", "double"]

    results_by_dtype = {}

    if args.plot_only == 0:
        # Normal mode: run benchmarks then parse CSVs
        if not bin_path.exists():
            raise SystemExit(
                f"ERROR: benchmark binary not found: {bin_path}\n"
                f"Hint: build it and/or pass --bin /path/to/binary, or use --plot_only 1"
            )

        for dt in dtypes:
            out_csv = add_suffix(out_base, dt)

            cmd = [
                str(bin_path),
                "--type", dt,
                "--n", str(args.n),
                "--min_block", str(args.min_block),
                "--max_block", str(args.max_block),
                "--descending", str(args.descending),
                "--check", str(args.check),
                "--seed", str(args.seed),
            ]

            stdout, stderr = run(cmd)
            out_csv.write_text(stdout)
            sys.stderr.write(stderr)
            print(f"Wrote CSV: {out_csv}")

            parsed = read_csv_for_plot(out_csv)
            if parsed is None:
                continue
            dtype, elem_size, block_sizes, thr_gib_s = parsed
            results_by_dtype[dtype] = {
                "elem_size": elem_size,
                "block_sizes": block_sizes,
                "thr_gib_s": thr_gib_s,
            }

    else:
        # Plot-only mode: read existing CSVs
        any_found = False
        for dt in dtypes:
            out_csv = add_suffix(out_base, dt)
            if not out_csv.exists():
                print(f"[PlotOnly] Missing CSV, skip: {out_csv}")
                continue

            parsed = read_csv_for_plot(out_csv)
            if parsed is None:
                print(f"[PlotOnly] Empty/invalid CSV, skip: {out_csv}")
                continue

            any_found = True
            dtype, elem_size, block_sizes, thr_gib_s = parsed
            results_by_dtype[dtype] = {
                "elem_size": elem_size,
                "block_sizes": block_sizes,
                "thr_gib_s": thr_gib_s,
            }
            print(f"[PlotOnly] Loaded CSV: {out_csv}")

        if not any_found:
            raise SystemExit(
                "ERROR: --plot_only 1 but no per-dtype CSV files were found.\n"
                f"Expected files like: {add_suffix(out_base,'int32')} (and float/int64/double)\n"
                "Hint: either run once without --plot_only, or set --out to the correct base path."
            )

    if args.plot != 0:
        plot_4up(results_by_dtype, cpu_info, png_path)
        print(f"Wrote plot: {png_path}")

if __name__ == "__main__":
    main()