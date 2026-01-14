import os
import glob
import socket
import argparse
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import sys

# === 设置绘图风格 ===
sns.set_theme(style="whitegrid", context="talk")
plt.rcParams.update({
    'font.size': 11,
    'axes.titlesize': 14,
    'axes.labelsize': 12,
    'xtick.labelsize': 10,
    'ytick.labelsize': 10,
    'figure.titlesize': 20
})

def get_latest_result_file(base_dir):
    """ 自动寻找最新的 bw_amp_grid_* 目录 """
    search_pattern = os.path.join(base_dir, "bw_amp_grid_*")
    dirs = sorted(glob.glob(search_pattern))

    if not dirs:
        print(f"Error: No 'bw_amp_grid_*' directories found in {base_dir}")
        sys.exit(1)

    latest_dir = dirs[-1]
    csv_path = os.path.join(latest_dir, "grid_results.csv")

    if not os.path.exists(csv_path):
        print(f"Error: 'grid_results.csv' not found in {latest_dir}")
        sys.exit(1)

    print(f"Found latest result directory: {latest_dir}")
    print(f"Processing file: {csv_path}")
    return latest_dir, csv_path

def add_bar_labels(ax, fmt='%.1f', rotation=90, fontsize=9, max_total_bars=40):
    """
    辅助函数：添加数值标签（类别太多时自动不加，避免挤成一团）
    max_total_bars: 当前 Axes 上柱子的总数超过阈值则不加 label
    """
    total_bars = 0
    for c in ax.containers:
        try:
            total_bars += len(c.datavalues)
        except Exception:
            pass
    if total_bars > max_total_bars:
        return

    for container in ax.containers:
        try:
            labels = [(fmt % v) if (v is not None and v > 0) else '' for v in container.datavalues]
            ax.bar_label(container, labels=labels, padding=2, fontsize=fontsize, rotation=rotation)
        except Exception:
            pass

def compute_figsize(n_cats, base_w=20, base_h=15, per_cat_w=0.55, min_w=20, max_w=90):
    """
    根据“右上角 full_label 类别数量”自适应图宽
    n_cats: full_label 类别数（kernel×block 的组合数）
    """
    # n_cats <= 18 基本不挤；超过开始线性加宽
    extra = max(0, n_cats - 18)
    w = base_w + extra * per_cat_w
    w = max(min_w, min(max_w, w))
    return w, base_h

def process_and_plot(csv_path, output_dir, dpi=150, max_width=90):
    # 1. 读取数据
    try:
        df = pd.read_csv(csv_path)
    except Exception as e:
        print(f"Error reading CSV: {e}")
        return

    hostname = socket.gethostname()

    # === 确保使用 RW (Read+Write) ===
    if 'goodput_rw_gbps' not in df.columns:
        if 'goodput_oneway_gbps' in df.columns:
            print("Notice: 'goodput_rw_gbps' not found, deriving from 'goodput_oneway_gbps' * 2")
            df['goodput_rw_gbps'] = df['goodput_oneway_gbps'] * 2.0
        else:
            print("Error: Neither 'goodput_rw_gbps' nor 'goodput_oneway_gbps' found in CSV.")
            return

    # 2. 数据预处理
    numeric_cols = ['roi_s', 'goodput_rw_gbps', 'imc_total_gbps', 'amp_read', 'amp_write']
    group_keys = ['kernel', 'block']

    for col in numeric_cols:
        if col not in df.columns:
            print(f"Warning: Column '{col}' missing, filling with 0")
            df[col] = 0

    df_avg = df.groupby(group_keys)[numeric_cols].mean().reset_index()

    # 排序：让 kernel 内部按 block 递增（画出来更有序）
    try:
        df_avg['block_num'] = df_avg['block'].astype(int)
    except Exception:
        df_avg['block_num'] = df_avg['block']
    df_avg = df_avg.sort_values(by=['kernel', 'block_num']).reset_index(drop=True)

    # 将 Block 转为 String 以便分组颜色
    df_avg['block_str'] = df_avg['block_num'].apply(lambda x: f"{int(x)}")

    # 右上角的 x 轴 label（最挤的就是这里）
    df_avg['full_label'] = df_avg.apply(lambda x: f"{x['kernel']}\n(blk={int(x['block_num'])})", axis=1)
    n_labels = df_avg['full_label'].nunique()

    # === 关键：根据 n_labels 自适应 figsize（图变宽）===
    fig_w, fig_h = compute_figsize(n_labels, base_w=20, base_h=15, per_cat_w=0.60, min_w=20, max_w=max_width)

    fig, axes = plt.subplots(2, 2, figsize=(fig_w, fig_h))

    filename_short = os.path.basename(os.path.dirname(csv_path))
    fig.suptitle(
        f"Bandwidth & Amplification Report @ {hostname}\nSource: {filename_short}",
        fontweight='bold',
        y=0.98
    )

    # === 动态调参：类别多就更激进地旋转、缩小字号、减少 bar label ===
    if n_labels <= 18:
        xrot = 45
        xtick_fs = 10
        bw_label_maxbars = 60
        bw_label_rot = 0
        bw_label_fs = 9
    elif n_labels <= 30:
        xrot = 60
        xtick_fs = 9
        bw_label_maxbars = 40
        bw_label_rot = 90
        bw_label_fs = 8
    else:
        xrot = 75
        xtick_fs = 8
        bw_label_maxbars = 0   # 直接关闭右上 bar label（否则必糊）
        bw_label_rot = 90
        bw_label_fs = 7

    # === 左上: 执行时间 ===
    ax1 = axes[0, 0]
    sns.barplot(
        x='kernel', y='roi_s', hue='block_str', data=df_avg,
        ax=ax1, palette='viridis', edgecolor='black'
    )
    ax1.set_title("1. Execution Time (Lower is Better)", fontweight='bold')
    ax1.set_ylabel("Time (seconds)")
    ax1.set_xlabel("")
    ax1.legend(title='Block Size', loc='upper right')
    # 左上柱子数量不大，可以保留 label，但也加个阈值防爆炸
    add_bar_labels(ax1, fmt='%.3fs', rotation=90, fontsize=9, max_total_bars=80)

    # === 右上: 带宽对比 (RW) ===
    ax2 = axes[0, 1]
    df_melt = df_avg.melt(
        id_vars=['full_label'],
        value_vars=['goodput_rw_gbps', 'imc_total_gbps'],
        var_name='Bandwidth Type',
        value_name='GB/s'
    )
    df_melt['Bandwidth Type'] = df_melt['Bandwidth Type'].replace({
        'goodput_rw_gbps': 'Logical (RW)',
        'imc_total_gbps': 'Physical (IMC Total)'
    })

    # 固定 full_label 顺序（避免 seaborn 自动重排导致视觉混乱）
    order_full = df_avg['full_label'].tolist()

    sns.barplot(
        x='full_label', y='GB/s', hue='Bandwidth Type',
        data=df_melt, ax=ax2,
        order=order_full,
        edgecolor='black'
    )

    ax2.set_title("2. Bandwidth Efficiency (Logical RW vs Physical)", fontweight='bold')
    ax2.set_ylabel("Bandwidth (GB/s)")
    ax2.set_xlabel("")
    ax2.set_xticklabels(ax2.get_xticklabels(), rotation=xrot, ha='right', fontsize=xtick_fs)
    ax2.legend(title=None, loc='upper right')

    # 类别多时不加 label（否则挤爆）
    if bw_label_maxbars > 0:
        add_bar_labels(ax2, fmt='%.1f', rotation=bw_label_rot, fontsize=bw_label_fs, max_total_bars=bw_label_maxbars)

    # === 左下: 读放大 (Read Amp) ===
    ax3 = axes[1, 0]
    sns.barplot(
        x='kernel', y='amp_read', hue='block_str', data=df_avg,
        ax=ax3, palette='magma', edgecolor='black'
    )
    ax3.axhline(1.0, color='gray', linestyle='--', linewidth=2, label='Ideal (1.0x)')
    ax3.set_title("3. Read Amplification (RFO Check)", fontweight='bold')
    ax3.set_ylabel("Factor (Physical / Logical)")
    ax3.set_xlabel("Kernel")
    ax3.legend(title='Block Size', loc='upper right')
    add_bar_labels(ax3, fmt='%.2fx', rotation=90, fontsize=9, max_total_bars=80)

    # === 右下: 写放大 (Write Amp) ===
    ax4 = axes[1, 1]
    sns.barplot(
        x='kernel', y='amp_write', hue='block_str', data=df_avg,
        ax=ax4, palette='rocket', edgecolor='black'
    )
    ax4.axhline(1.0, color='gray', linestyle='--', linewidth=2, label='Ideal (1.0x)')
    ax4.set_title("4. Write Amplification (Thrashing Check)", fontweight='bold')
    ax4.set_ylabel("Factor (Physical / Logical)")
    ax4.set_xlabel("Kernel")
    if ax4.get_legend():
        ax4.get_legend().remove()
    add_bar_labels(ax4, fmt='%.2fx', rotation=90, fontsize=9, max_total_bars=80)

    # === 保存 ===
    # 给 suptitle 留空间：顶部留得更大一点
    plt.tight_layout(rect=[0, 0, 1, 0.95])

    output_filename = os.path.join(output_dir, f"benchmark_4grid_{hostname}.png")
    plt.savefig(output_filename, dpi=dpi)
    print(f"\n[Success] Plot saved to: {output_filename}")

    local_filename = f"summary_4grid_{hostname}.png"
    plt.savefig(local_filename, dpi=dpi)
    print(f"[Success] Copy saved to current dir: {local_filename}")

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("dir", nargs="?", default=".", help="Directory")
    parser.add_argument("--dpi", type=int, default=150, help="Output DPI")
    parser.add_argument("--max-width", type=int, default=90, help="Max figure width (inches) when auto-expanding")
    args = parser.parse_args()

    target_dir = args.dir

    # 判断输入是直接的 result 目录还是父目录
    if os.path.exists(os.path.join(target_dir, "grid_results.csv")):
        csv_file = os.path.join(target_dir, "grid_results.csv")
        plot_dir = target_dir
        print(f"Processing specific directory: {target_dir}")
    else:
        plot_dir, csv_file = get_latest_result_file(target_dir)

    process_and_plot(csv_file, plot_dir, dpi=args.dpi, max_width=args.max_width)

if __name__ == "__main__":
    main()
