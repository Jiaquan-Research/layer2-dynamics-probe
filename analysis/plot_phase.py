import matplotlib.pyplot as plt
import argparse
import os
import sys

# 动态添加路径，确保能找到同目录下的 parse_log.py
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

# 尝试导入解析函数（兼容直接运行和模块运行）
try:
    from parse_log import parse_log_file
except ImportError:
    from analysis.parse_log import parse_log_file


def plot_phase_portrait(data, out_path):
    """
    Plot κ–H phase portrait from parsed data.
    """
    styles = {
        "H": dict(color="#2ca02c", label="Healthy Baseline", alpha=0.6),  # Green
        "L": dict(color="#1f77b4", label="Lock-in Attractor", alpha=0.8),  # Blue
        "D": dict(color="#ff7f0e", label="Decoupled Oscillation", alpha=0.8),  # Orange
        "G": dict(color="#9467bd", label="Geodesic Flow", alpha=0.9),  # Purple
    }

    plt.figure(figsize=(10, 8))
    # 尝试设置好看的样式，如果没有则回退
    try:
        plt.style.use('seaborn-v0_8-darkgrid')
    except:
        plt.style.use('ggplot')

    count = 0
    for tag, points in data.items():
        if not points:
            continue
        count += 1
        xs = [p["consistency"] for p in points]
        ys = [p["entropy"] for p in points]

        # 绘制连线
        plt.plot(xs, ys, linestyle='-', linewidth=1.5, **styles[tag])

        # 绘制数据点
        plt.scatter(xs, ys, s=60, edgecolors='white', linewidth=0.5,
                    color=styles[tag]['color'], zorder=5)

        # 标记起点(S)和终点(E)
        if xs:
            plt.text(xs[0], ys[0], 'S', fontsize=8, fontweight='bold', ha='right', color=styles[tag]['color'])
            plt.text(xs[-1], ys[-1], 'E', fontsize=8, fontweight='bold', ha='left', color=styles[tag]['color'])

    plt.xlabel("Structural Consistency $\kappa$ (Cosine Sim)", fontsize=12)
    plt.ylabel("Local Entropy $H$ (Uncertainty)", fontsize=12)
    plt.title("Layer-2 Phase Portrait: Semantic Dynamics", fontsize=14, pad=15)
    plt.legend(loc='upper right', frameon=True, shadow=True)
    plt.grid(True)
    plt.xlim(-0.1, 1.1)

    # 绘制软死锁区域 (Soft Lock-in Zone)
    plt.axvspan(0.85, 1.0, color='blue', alpha=0.05)
    plt.text(0.92, plt.ylim()[1] * 0.9, "Soft Lock-in\nZone", ha='center', fontsize=9, color='navy')

    plt.tight_layout()
    plt.savefig(out_path, dpi=300)
    plt.close()
    return count


def main():
    parser = argparse.ArgumentParser(description="Generate Layer-2 Phase Portrait")
    parser.add_argument("--log", type=str, required=True, help="Path to raw log file")
    parser.add_argument("--out", type=str, default="phase_portrait.png", help="Output image path")
    args = parser.parse_args()

    print(f"Parsing log file: {args.log}...")
    data = parse_log_file(args.log)

    if not any(data.values()):
        print("Error: No valid data parsed. Check log file format.")
        return

    print(f"Generating plot to: {args.out}...")
    os.makedirs(os.path.dirname(args.out) or '.', exist_ok=True)

    count = plot_phase_portrait(data, args.out)
    print(f"[OK] Plot generated successfully! ({count} trajectories plotted)")


if __name__ == "__main__":
    main()