"""
生成 RAG 检索指标图表（论文风格）
数据来源：experiments/rag/logs/eval_combined_20260403_124230.json

输出：
  docs/diagrams/rag_metrics_overall.png   —— ALL 数据集，4 方法 A/B 对比
  docs/diagrams/rag_metrics_by_type.png   —— GE vs O 分问题类型对比

运行：python docs/diagrams/gen_metrics.py
"""

import json
import numpy as np
import matplotlib
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from pathlib import Path

matplotlib.use("Agg")

# ─────────────────────────────────────────────────────────────────────────────
# 配置：论文风格 rcParams
# ─────────────────────────────────────────────────────────────────────────────

plt.rcParams.update({
    "font.family":        "serif",
    "font.serif":         ["Times New Roman", "DejaVu Serif", "Georgia"],
    "font.size":          11,
    "axes.labelsize":     12,
    "axes.titlesize":     13,
    "axes.titleweight":   "bold",
    "xtick.labelsize":    11,
    "ytick.labelsize":    10,
    "legend.fontsize":    10,
    "legend.framealpha":  0.92,
    "legend.edgecolor":   "#cccccc",
    "axes.spines.top":    False,
    "axes.spines.right":  False,
    "axes.grid":          True,
    "axes.grid.axis":     "y",
    "grid.alpha":         0.35,
    "grid.linestyle":     "--",
    "grid.color":         "#999999",
    "figure.dpi":         150,
    "savefig.dpi":        300,
    "savefig.bbox":       "tight",
    "savefig.facecolor":  "white",
    "svg.fonttype":       "none",   # use system fonts in SVG (smaller file)
})

# ─────────────────────────────────────────────────────────────────────────────
# 数据
# ─────────────────────────────────────────────────────────────────────────────

ROOT = Path(__file__).parents[2]
LOG  = ROOT / "experiments/rag/logs/eval_combined_20260403_124230.json"
OUT  = Path(__file__).parent

data = json.loads(LOG.read_text())["results"]

METHODS = ["bi", "graph", "rrf", "rrf_path_rr"]
LABELS  = ["Bi-Encoder", "Graph RAG", "RRF", "RRF + Rerank (ours)"]
METRICS = ["hit1", "hit3", "hit5", "mrr"]
M_LABELS = ["Hit@1", "Hit@3", "Hit@5", "MRR"]

# 每个方法的颜色（colorblind-friendly）
METHOD_COLORS = {
    "bi":          "#4878d0",   # blue
    "graph":       "#8c7be8",   # purple
    "rrf":         "#ee854a",   # orange
    "rrf_path_rr": "#d65f5f",   # red (our method)
}
# rrf_path_rr 加 hatch 以突出显示
HATCHES = {
    "bi": "", "graph": "", "rrf": "", "rrf_path_rr": "//"
}


# ─────────────────────────────────────────────────────────────────────────────
# 图 1：ALL 数据集，4 指标 grouped bar chart
# ─────────────────────────────────────────────────────────────────────────────

def plot_overall():
    """Two-panel figure: Hit@K (left) and MRR (right) kept separate."""
    fig, (ax_hit, ax_mrr) = plt.subplots(
        1, 2, figsize=(11, 4.5),
        gridspec_kw={"width_ratios": [3, 1.4], "wspace": 0.32},
    )

    # ── Left: Hit@1 / Hit@3 / Hit@5 ─────────────────────────────────────
    hit_metrics = ["hit1", "hit3", "hit5"]
    hit_labels  = ["Hit@1", "Hit@3", "Hit@5"]
    n_hit = len(hit_metrics)
    n_methods = len(METHODS)
    bar_w = 0.18
    group_w = n_methods * bar_w + 0.10

    xs = np.arange(n_hit) * group_w
    offsets = np.linspace(0, (n_methods - 1) * bar_w, n_methods)
    offsets -= offsets.mean()

    for i, (method, label) in enumerate(zip(METHODS, LABELS)):
        vals = [data["ALL"][method][m] for m in hit_metrics]
        bars = ax_hit.bar(
            xs + offsets[i], vals,
            width=bar_w,
            color=METHOD_COLORS[method],
            hatch=HATCHES[method],
            edgecolor="white", linewidth=0.6,
            label=label, zorder=3, alpha=0.92,
        )
        if method == "rrf_path_rr":
            for bar, val in zip(bars, vals):
                ax_hit.text(
                    bar.get_x() + bar.get_width() / 2,
                    bar.get_height() + 0.013,
                    f"{val:.2f}",
                    ha="center", va="bottom",
                    fontsize=8.5, fontweight="bold",
                    color=METHOD_COLORS[method],
                )

    ax_hit.set_xticks(xs)
    ax_hit.set_xticklabels(hit_labels)
    ax_hit.set_ylabel("Score")
    ax_hit.set_title("Hit@K  —  Combined Test Set (n=60)", pad=8)
    ax_hit.set_ylim(0, 1.12)
    ax_hit.yaxis.set_major_formatter(matplotlib.ticker.FormatStrFormatter("%.1f"))
    ax_hit.legend(loc="upper left", ncol=1, handlelength=1.3,
                  handletextpad=0.5, frameon=True, fontsize=9.5)

    # ── Right: MRR ───────────────────────────────────────────────────────
    mrr_vals = [data["ALL"][m]["mrr"] for m in METHODS]
    x_mrr = np.arange(n_methods)

    bars_mrr = ax_mrr.bar(
        x_mrr, mrr_vals,
        width=0.5,
        color=[METHOD_COLORS[m] for m in METHODS],
        hatch=[HATCHES[m] for m in METHODS],
        edgecolor="white", linewidth=0.6,
        zorder=3, alpha=0.92,
    )
    for bar, val, method in zip(bars_mrr, mrr_vals, METHODS):
        weight = "bold" if method == "rrf_path_rr" else "normal"
        ax_mrr.text(
            bar.get_x() + bar.get_width() / 2,
            bar.get_height() + 0.013,
            f"{val:.2f}",
            ha="center", va="bottom",
            fontsize=9, fontweight=weight,
            color=METHOD_COLORS[method],
        )

    ax_mrr.set_xticks(x_mrr)
    ax_mrr.set_xticklabels(
        ["Bi-Enc.", "Graph", "RRF", "RRF+RR"],
        fontsize=9.5,
    )
    ax_mrr.set_ylabel("MRR")
    ax_mrr.set_title("MRR", pad=8)
    ax_mrr.set_ylim(0, 1.12)
    ax_mrr.yaxis.set_major_formatter(matplotlib.ticker.FormatStrFormatter("%.1f"))

    fig.suptitle(
        "Retrieval Performance on Combined Test Set",
        fontsize=13, fontweight="bold", y=1.01,
    )
    fig.tight_layout()
    out = OUT / "rag_metrics_overall.svg"
    fig.savefig(out)
    print(f"✅ {out}")
    plt.close(fig)


# ─────────────────────────────────────────────────────────────────────────────
# 图 2：GE vs O 分问题类型对比（指标：Hit@1 + MRR，2 × subplot）
# ─────────────────────────────────────────────────────────────────────────────

def plot_by_type():
    fig, axes = plt.subplots(1, 2, figsize=(11, 4.5), sharey=False)

    SPLITS     = ["GE", "O"]
    SPLIT_LABELS = [
        "GE  (Graph-Enhanced)\ncross-concept reasoning questions",
        "O  (Original In-Distribution)",
    ]

    for ax, split, split_label in zip(axes, SPLITS, SPLIT_LABELS):
        for metric_idx, (metric, m_label) in enumerate([("hit1", "Hit@1"), ("mrr", "MRR")]):
            n_methods = len(METHODS)
            bar_w = 0.3
            group_gap = 0.15
            group_w = 2 * bar_w + group_gap  # 2 metrics per method group

            xs = np.arange(n_methods) * (group_w + 0.2)
            offset = (metric_idx - 0.5) * bar_w

            vals = [data[split][m][metric] for m in METHODS]
            alpha = 0.9 if metric == "hit1" else 0.6
            hatch = "" if metric == "hit1" else "xx"
            bars = ax.bar(
                xs + offset, vals,
                width=bar_w,
                color=[METHOD_COLORS[m] for m in METHODS],
                hatch=hatch,
                edgecolor="white",
                linewidth=0.6,
                alpha=alpha,
                zorder=3,
                label=m_label,
            )
            # value labels on top
            for bar, val in zip(bars, vals):
                ax.text(
                    bar.get_x() + bar.get_width() / 2,
                    bar.get_height() + 0.012,
                    f"{val:.2f}",
                    ha="center", va="bottom",
                    fontsize=8, color="#333333",
                )

        ax.set_xticks(xs)
        ax.set_xticklabels(LABELS, rotation=12, ha="right", fontsize=10)
        ax.set_ylabel("Score")
        ax.set_title(split_label, pad=8, fontsize=11)
        ax.set_ylim(0, 1.18)
        ax.yaxis.set_major_formatter(matplotlib.ticker.FormatStrFormatter("%.1f"))

        # 图例（仅第一个 subplot）
        if ax is axes[0]:
            solid = mpatches.Patch(color="#888888", alpha=0.9, label="Hit@1")
            hatch_p = mpatches.Patch(
                facecolor="#888888", hatch="xx", edgecolor="white",
                alpha=0.6, label="MRR"
            )
            ax.legend(handles=[solid, hatch_p], loc="upper right",
                      frameon=True, fontsize=10)

    # 方法颜色图例（底部）
    handles = [
        mpatches.Patch(color=METHOD_COLORS[m], label=lbl)
        for m, lbl in zip(METHODS, LABELS)
    ]
    fig.legend(handles=handles, loc="lower center", ncol=4,
               bbox_to_anchor=(0.5, -0.06), frameon=True,
               title="Retrieval Method", title_fontsize=10)

    fig.suptitle(
        "Per-Question-Type Retrieval Performance: GE vs. Original",
        fontsize=13, fontweight="bold", y=1.02
    )
    fig.tight_layout()
    out = OUT / "rag_metrics_by_type.svg"
    fig.savefig(out)
    print(f"✅ {out}")
    plt.close(fig)


# ─────────────────────────────────────────────────────────────────────────────
# 图 3：Summary heat-table（论文里常见的结果表格可视化）
# ─────────────────────────────────────────────────────────────────────────────

def plot_heatmap():
    # 行：method × split，列：metric
    row_labels = []
    matrix = []
    for split in ["GE", "O", "ALL"]:
        for method in METHODS:
            row_labels.append(f"{LABELS[METHODS.index(method)]}  [{split}]")
            matrix.append([data[split][method][m] for m in METRICS])

    mat = np.array(matrix)

    fig, ax = plt.subplots(figsize=(8, 6.5))
    im = ax.imshow(mat, aspect="auto", cmap="RdYlGn", vmin=0, vmax=1)

    ax.set_xticks(range(len(METRICS)))
    ax.set_xticklabels(M_LABELS, fontsize=11)
    ax.set_yticks(range(len(row_labels)))
    ax.set_yticklabels(row_labels, fontsize=9)
    ax.set_title("Retrieval Score Heatmap  (GE / O / ALL × Methods)",
                 pad=10, fontsize=12)

    # 数值标注
    for i in range(mat.shape[0]):
        for j in range(mat.shape[1]):
            val = mat[i, j]
            color = "white" if val > 0.7 or val < 0.2 else "black"
            ax.text(j, i, f"{val:.2f}", ha="center", va="center",
                    fontsize=9, color=color, fontweight="bold")

    # 分隔线（每4行一组，对应同一 split）
    for sep in [3.5, 7.5]:
        ax.axhline(sep, color="white", linewidth=2.5)

    # split 标注（左侧）
    for si, split in enumerate(["GE", "O", "ALL"]):
        ax.text(-0.7, si * 4 + 1.5, split,
                ha="right", va="center", fontsize=11,
                fontweight="bold", transform=ax.transData,
                color="#444444")

    plt.colorbar(im, ax=ax, shrink=0.7, label="Score")
    ax.spines[:].set_visible(False)
    fig.tight_layout()
    out = OUT / "rag_metrics_heatmap.svg"
    fig.savefig(out)
    print(f"✅ {out}")
    plt.close(fig)


if __name__ == "__main__":
    plot_overall()
    plot_by_type()
    plot_heatmap()
    print("\nAll charts saved to docs/diagrams/")
