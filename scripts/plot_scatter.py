import os
import glob
import argparse
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import sys
import numpy as np
import socket  # 用于获取 Hostname

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

def get_target_file(input_path):
    """ 智能解析路径 """
    input_path = os.path.abspath(input_path)
    
    if os.path.isfile(input_path) and input_path.endswith('.csv'):
        return os.path.dirname(input_path), input_path
    
    direct_csv = os.path.join(input_path, "scatter_grid.csv")
    if os.path.isdir(input_path) and os.path.exists(direct_csv):
        return input_path, direct_csv

    print(f"[Mode] Searching for latest result in {input_path}...")
    pattern = os.path.join(input_path, "**", "scatter_grid.csv")
    candidates = glob.glob(pattern, recursive=True)
    
    if not candidates:
        pattern = os.path.join(input_path, "scatter_grid_*", "scatter_grid.csv")
        candidates = glob.glob(pattern)
    
    if not candidates:
        print(f"Error: No scatter_grid.csv found in {input_path}")
        sys.exit(1)

    latest_csv = max(candidates, key=os.path.getmtime)
    print(f"Processing latest file: {latest_csv}")
    return os.path.dirname(latest_csv), latest_csv

def add_bar_labels(ax, fmt='%.1f', rotation=90):
    """ 辅助函数：添加数值标签 (兼容旧版 Python 格式化) """
    for container in ax.containers:
        labels = [(fmt % v) if v > 0 else '' for v in container.datavalues]
        ax.bar_label(container, labels=labels, padding=3, fontsize=9, rotation=rotation)

def process_and_plot(csv_path, output_dir):
    try:
        df = pd.read_csv(csv_path)
    except Exception as e:
        print(f"Error reading CSV: {e}")
        return

    # === [配置] 获取 Hostname ===
    hostname = socket.gethostname()

    # === 数据预处理 ===
    df = df.sort_values(by=['buckets', 'tile'])
    df['buckets_cat'] = df['buckets'].astype(str)
    
    def format_tile(x):
        if x >= 1048576: return f"{x//1048576}M"
        if x >= 1024: return f"{x//1024}k"
        return str(x)
    df['tile_label'] = df['tile'].apply(format_tile)

    # === [恢复逻辑] 使用 RW (Read+Write) Goodput ===
    # 优先找 goodput_rw_gbps，如果没有则用 oneway * 2
    if 'goodput_rw_gbps' in df.columns:
        df['plot_goodput'] = df['goodput_rw_gbps']
    elif 'goodput_oneway_gbps' in df.columns:
        df['plot_goodput'] = df['goodput_oneway_gbps'] * 2.0
    else:
        print("Error: No goodput columns found.")
        return

    # === 计算效率 (Bus Efficiency) ===
    # Logic: Goodput(RW) / Physical_Total
    df['Efficiency'] = df.apply(lambda row: row['plot_goodput'] / row['imc_total_gbps'] 
                                if row['imc_total_gbps'] > 0 else 0, axis=1)
    
    # 选取表现最好的 Tile
    df_sorted = df.sort_values('plot_goodput', ascending=False)
    df_best_tile = df_sorted.drop_duplicates(subset=['buckets', 'mode']).sort_values('buckets')

    # 确定 Heatmap 目标 Mode
    target_mode = 'avx512_galloping_aligned'
    if target_mode not in df['mode'].values and len(df['mode'].unique()) > 0:
        target_mode = df['mode'].unique()[0]
    
    # === 布局设置: 4行 x 2列 ===
    fig, axes = plt.subplots(4, 2, figsize=(24, 28))
    
    filename_short = os.path.basename(os.path.dirname(csv_path))
    # 标题包含 Hostname
    fig.suptitle(f"Scatter Benchmark Report @ {hostname}\nSource: {filename_short}", 
                 fontweight='bold', y=0.995)

    # ==========================================================================
    # Row 1: 宏观趋势 (RW)
    # ==========================================================================
    
    # 图 1: Scalability
    ax1 = axes[0, 0]
    sns.lineplot(x='buckets', y='plot_goodput', hue='mode', style='mode', 
                 markers=True, dashes=False, data=df_best_tile, ax=ax1, linewidth=3, markersize=10)
    ax1.set_xscale('log', base=2)
    ax1.set_title("1. Scalability: Goodput (RW) vs Buckets", fontweight='bold')
    ax1.set_ylabel("Goodput (RW GB/s)")
    ax1.grid(True, which="minor", ls="--", alpha=0.5)

    # 图 2: Tile Sensitivity
    ax2 = axes[0, 1]
    max_bkt = df['buckets'].max()
    df_tile_sensitivity = df[df['buckets'] == max_bkt].copy()
    sns.barplot(x='tile_label', y='plot_goodput', hue='mode', 
                data=df_tile_sensitivity, ax=ax2, palette='viridis', edgecolor='black')
    ax2.set_title(f"2. Tuning: Tile Sensitivity (at Buckets={max_bkt})", fontweight='bold')
    ax2.set_ylabel("Goodput (RW GB/s)")
    ax2.legend(loc='upper left', bbox_to_anchor=(1, 1))
    add_bar_labels(ax2, fmt='%.1f')

    # ==========================================================================
    # Row 2: 诊断指标 (Amplification 保持不变)
    # ==========================================================================
    
    # 图 3: Read Amplification
    ax3 = axes[1, 0]
    sns.barplot(x='buckets_cat', y='amp_read', hue='mode', data=df_best_tile, 
                ax=ax3, palette='magma', edgecolor='black')
    ax3.axhline(1.0, color='gray', linestyle='--', label='Ideal (1.0x)')
    ax3.set_title("3. Read Amplification (RFO Analysis)", fontweight='bold')
    ax3.set_ylabel("Read Amp Factor")
    ax3.legend(loc='upper left')
    add_bar_labels(ax3, fmt='%.2fx')

    # 图 4: Write Amplification
    ax4 = axes[1, 1]
    sns.barplot(x='buckets_cat', y='amp_write', hue='mode', data=df_best_tile, 
                ax=ax4, palette='rocket', edgecolor='black')
    ax4.axhline(1.0, color='red', linestyle='--', label='Ideal (1.0x)')
    ax4.set_title("4. Write Amplification (Thrashing Analysis)", fontweight='bold')
    ax4.set_ylabel("Write Amp Factor")
    ax4.legend(loc='upper right')
    add_bar_labels(ax4, fmt='%.2fx')

    # ==========================================================================
    # Row 3: 效率分析 (RW vs Physical)
    # ==========================================================================

    # 图 5: Efficiency Gap
    ax5 = axes[2, 0]
    df_melt = df_best_tile.melt(
        id_vars=['buckets', 'buckets_cat', 'mode'], 
        value_vars=['plot_goodput', 'imc_total_gbps'],
        var_name='Metric', value_name='GB/s'
    )
    df_melt['Metric'] = df_melt['Metric'].replace({
        'plot_goodput': 'Logical (RW)', 
        'imc_total_gbps': 'Physical (Total)'
    })
    sns.barplot(x='buckets_cat', y='GB/s', hue='Metric', data=df_melt, 
                ax=ax5, palette=['#2ecc71', '#e74c3c'], alpha=0.9, edgecolor='black')
    ax5.set_title("5. Efficiency Gap: Logical (RW) vs Physical", fontweight='bold')
    ax5.legend(loc='upper left')
    add_bar_labels(ax5, fmt='%.0f')

    # 图 6: Bus Efficiency
    ax6 = axes[2, 1]
    sns.lineplot(x='buckets', y='Efficiency', hue='mode', data=df_best_tile, 
                 ax=ax6, marker='o', linewidth=3)
    ax6.set_xscale('log', base=2)
    max_eff = df_best_tile['Efficiency'].max()
    ax6.set_ylim(bottom=0, top=max(1.0, max_eff * 1.1))
    ax6.axhline(0.5, color='gray', linestyle='--', label='50% (RFO Limit)')
    ax6.set_title("6. Bus Utilization (RW / Physical Total)", fontweight='bold')
    ax6.set_ylabel("Efficiency Ratio")
    ax6.legend(loc='lower left')

    # ==========================================================================
    # Row 4: 全景热力图 (RW)
    # ==========================================================================
    
    df_target = df[df['mode'] == target_mode]
    sorted_tiles = df.sort_values('tile')['tile_label'].unique()
    
    # Heatmap 1: Goodput (RW)
    ax7 = axes[3, 0]
    hm_bw = df_target.pivot_table(index='buckets', columns='tile_label', values='plot_goodput')
    hm_bw = hm_bw.reindex(columns=[t for t in sorted_tiles if t in hm_bw.columns])
    
    sns.heatmap(hm_bw, annot=True, fmt=".0f", cmap="YlGnBu", ax=ax7, 
                cbar_kws={'label': 'GB/s'})
    ax7.set_title(f"7. Bandwidth Heatmap (RW) - {target_mode}", fontweight='bold')
    ax7.invert_yaxis()
    ax7.set_ylabel("Buckets")
    ax7.set_xlabel("Tile Size")

    # Heatmap 2: ROI Time
    ax8 = axes[3, 1]
    hm_time = df_target.pivot_table(index='buckets', columns='tile_label', values='roi_s')
    hm_time = hm_time.reindex(columns=[t for t in sorted_tiles if t in hm_time.columns])
    
    sns.heatmap(hm_time, annot=True, fmt=".2g", cmap="YlOrRd", ax=ax8, 
                cbar_kws={'label': 'Seconds'})
    ax8.set_title(f"8. ROI Time Heatmap - {target_mode}", fontweight='bold')
    ax8.invert_yaxis()
    ax8.set_ylabel("Buckets")
    ax8.set_xlabel("Tile Size")

    # ==========================================================================
    # 保存
    # ==========================================================================
    plt.tight_layout(rect=[0, 0, 1, 0.96])
    plt.subplots_adjust(hspace=0.4, wspace=0.2)
    
    out_file = os.path.join(output_dir, f"report_{hostname}.png")
    plt.savefig(out_file, dpi=150)
    print(f"\n[Success] Plot saved to: {out_file}")
    
    local_file = f"summary_{hostname}.png"
    plt.savefig(local_file, dpi=150)
    print(f"[Success] Copy saved to current dir: {local_file}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("path", nargs="?", default=".", help="Path to CSV or dir")
    args = parser.parse_args()
    output_dir, csv_file = get_target_file(args.path)
    process_and_plot(csv_file, output_dir)