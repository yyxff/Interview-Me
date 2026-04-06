"""
RAG Evaluation Charts — SVG Output
====================================
数据来源：experiments/rag/logs/eval_combined_20260403_124230.json
         experiments/rag/logs/03_graph_eval/graph_eval_20260401_130939.json

输出（docs/diagrams/）：
  fig1_mrr_comparison.svg  — 四方法 × O/GE/ALL 数据集 MRR@5 主对比
  fig2_hitk_curves.svg     — Hit@K 曲线，O 与 GE 并排
  fig3_graph_benchmark.svg — 图谱专项：GA/GB/GC 场景
  fig4_heatmap.svg         — 方法 × 数据集 MRR@5 热力图

运行：python docs/diagrams/plot_results.py
"""

import json
import numpy as np
import matplotlib
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import matplotlib.ticker
from pathlib import Path

matplotlib.use("Agg")

ROOT = Path(__file__).parents[2]
OUT  = Path(__file__).parent

# ─────────────────────────────────────────────────────────────────────────────
# 论文风格 rcParams（与 gen_metrics.py 保持一致）
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
    "svg.fonttype":       "none",
})

# ─────────────────────────────────────────────────────────────────────────────
# 配色（与 gen_metrics.py 完全一致）
# ─────────────────────────────────────────────────────────────────────────────

METHODS = ["bi", "graph", "rrf", "rrf_path_rr"]
LABELS  = ["Bi-Encoder", "Graph RAG", "RRF", "RRF + Rerank (ours)"]

METHOD_COLORS = {
    "bi":          "#4878d0",
    "graph":       "#8c7be8",
    "rrf":         "#ee854a",
    "rrf_path_rr": "#d65f5f",
}
HATCHES = {
    "bi": "", "graph": "", "rrf": "", "rrf_path_rr": "//"
}

# ─────────────────────────────────────────────────────────────────────────────
# 数据
# ─────────────────────────────────────────────────────────────────────────────

COMBINED = json.loads(
    (ROOT / "experiments/rag/logs/eval_combined_20260403_124230.json").read_text()
)["results"]

GRAPH_EVAL = json.loads(
    (ROOT / "experiments/rag/logs/03_graph_eval/graph_eval_20260401_130939.json").read_text()
)["results"]

GRAPH_EVAL_LABELS = {
    "GA": "Single-Edge\n(n=50)",
    "GB": "Two-Hop Path\n(n=44)",
    "GC": "Cross-Modal\n(n=31)",
}


# ══════════════════════════════════════════════════════════════════════════════
# Fig 1 — MRR@5 主对比（O / GE / ALL × 四方法）
# ══════════════════════════════════════════════════════════════════════════════

def fig1_mrr_comparison():
    datasets  = ["O", "GE", "ALL"]
    ds_labels = {
        "O":   "In-Distribution\n(O, n=30)",
        "GE":  "Cross-Concept\n(GE, n=30)",
        "ALL": "Combined\n(ALL, n=60)",
    }

    n_ds  = len(datasets)
    n_met = len(METHODS)
    bar_w = 0.18
    x     = np.arange(n_ds)
    offsets = np.linspace(0, (n_met - 1) * bar_w, n_met)
    offsets -= offsets.mean()

    fig, ax = plt.subplots(figsize=(9, 5))

    for i, (method, label) in enumerate(zip(METHODS, LABELS)):
        values = [COMBINED[ds][method]["mrr"] for ds in datasets]
        bars = ax.bar(
            x + offsets[i], values, bar_w,
            label=label,
            color=METHOD_COLORS[method],
            hatch=HATCHES[method],
            edgecolor="white",
            linewidth=0.6,
            alpha=0.92,
            zorder=3,
        )
        if method == "rrf_path_rr":
            for bar, val in zip(bars, values):
                ax.text(
                    bar.get_x() + bar.get_width() / 2,
                    bar.get_height() + 0.015,
                    f"{val:.2f}",
                    ha="center", va="bottom",
                    fontsize=8.5, fontweight="bold",
                    color=METHOD_COLORS[method],
                )

    ax.set_xticks(x)
    ax.set_xticklabels([ds_labels[d] for d in datasets])
    ax.set_ylabel("MRR@5")
    ax.set_ylim(0, 1.05)
    ax.set_title("Retrieval Performance: MRR@5 by Dataset and Method", pad=10)
    ax.legend(loc="upper left", frameon=True, ncol=2, handlelength=1.3,
              handletextpad=0.5, fontsize=9.5)
    ax.yaxis.set_major_formatter(matplotlib.ticker.FormatStrFormatter("%.1f"))

    # "Ours" 注释箭头
    best_x   = x[2] + offsets[3]
    best_val = COMBINED["ALL"]["rrf_path_rr"]["mrr"]
    ax.annotate(
        "* Ours",
        xy=(best_x, best_val + 0.02),
        xytext=(best_x + 0.38, best_val + 0.13),
        fontsize=9, color=METHOD_COLORS["rrf_path_rr"], fontweight="bold",
        arrowprops=dict(arrowstyle="->", color=METHOD_COLORS["rrf_path_rr"], lw=1.5),
    )

    fig.tight_layout()
    out = OUT / "fig1_mrr_comparison.svg"
    fig.savefig(out)
    plt.close(fig)
    print(f"✅ {out}")


# ══════════════════════════════════════════════════════════════════════════════
# Fig 2 — Hit@K 曲线（O 与 GE 并排）
# ══════════════════════════════════════════════════════════════════════════════

def fig2_hitk_curves():
    ks       = [1, 3, 5]
    hit_keys = ["hit1", "hit3", "hit5"]
    ds_titles = {
        "O":  "In-Distribution (O)",
        "GE": "Cross-Concept (GE)",
    }

    fig, axes = plt.subplots(1, 2, figsize=(11, 4.5), sharey=True)

    for ax, ds in zip(axes, ["O", "GE"]):
        for method, label in zip(METHODS, LABELS):
            vals     = [COMBINED[ds][method][hk] for hk in hit_keys]
            is_ours  = method == "rrf_path_rr"
            ax.plot(
                ks, vals,
                color=METHOD_COLORS[method],
                label=label,
                linewidth=2.2 if is_ours else 1.4,
                linestyle="-"  if is_ours else "--",
                marker="o"     if is_ours else "s",
                markersize=7   if is_ours else 5,
                alpha=1.0      if is_ours else 0.8,
                zorder=5       if is_ours else 3,
            )
            ax.annotate(
                f"{vals[-1]:.2f}",
                xy=(5, vals[-1]),
                xytext=(5.15, vals[-1]),
                fontsize=8, color=METHOD_COLORS[method], va="center",
            )

        ax.set_title(ds_titles[ds], pad=8)
        ax.set_xlabel("K")
        ax.set_xticks(ks)
        ax.set_xlim(0.7, 5.9)
        ax.set_ylim(0, 1.1)
        ax.yaxis.set_major_formatter(matplotlib.ticker.FormatStrFormatter("%.1f"))

    axes[0].set_ylabel("Hit@K")
    axes[1].legend(loc="lower right", frameon=True, fontsize=9.5, ncol=1)

    fig.suptitle("Hit@K Curves Across Retrieval Methods",
                 fontsize=13, fontweight="bold", y=1.01)
    fig.tight_layout()
    out = OUT / "fig2_hitk_curves.svg"
    fig.savefig(out)
    plt.close(fig)
    print(f"✅ {out}")


# ══════════════════════════════════════════════════════════════════════════════
# Fig 3 — 图谱专项 benchmark（GA / GB / GC）
# ══════════════════════════════════════════════════════════════════════════════

def fig3_graph_benchmark():
    methods3 = ["bi", "rrf", "graph"]
    labels3  = {m: LABELS[METHODS.index(m)] for m in methods3}
    scenarios = [r["label"] for r in GRAPH_EVAL]

    n_sc  = len(scenarios)
    n_met = len(methods3)
    bar_w = 0.22
    x     = np.arange(n_sc)
    offsets = np.linspace(0, (n_met - 1) * bar_w, n_met)
    offsets -= offsets.mean()

    fig, ax = plt.subplots(figsize=(8, 4.8))

    for i, method in enumerate(methods3):
        values = [r["metrics"][method] for r in GRAPH_EVAL]
        bars = ax.bar(
            x + offsets[i], values, bar_w,
            label=labels3[method],
            color=METHOD_COLORS[method],
            hatch=HATCHES[method],
            edgecolor="white",
            linewidth=0.6,
            alpha=0.92,
            zorder=3,
        )
        for bar, val in zip(bars, values):
            ax.text(
                bar.get_x() + bar.get_width() / 2,
                bar.get_height() + 0.012,
                f"{val:.2f}",
                ha="center", va="bottom",
                fontsize=9, color="#333333",
            )

    ax.set_xticks(x)
    ax.set_xticklabels([GRAPH_EVAL_LABELS[s] for s in scenarios])
    ax.set_ylabel("Score")
    ax.set_ylim(0, 0.85)
    ax.set_title("Graph RAG Benchmark: Multi-Hop & Cross-Modal Retrieval", pad=10)
    ax.legend(loc="upper right", frameon=True, handlelength=1.3,
              handletextpad=0.5, fontsize=9.5)
    ax.yaxis.set_major_formatter(matplotlib.ticker.FormatStrFormatter("%.2f"))

    fig.tight_layout()
    out = OUT / "fig3_graph_benchmark.svg"
    fig.savefig(out)
    plt.close(fig)
    print(f"✅ {out}")


# ══════════════════════════════════════════════════════════════════════════════
# Fig 4 — 热力图总览
# ══════════════════════════════════════════════════════════════════════════════

def fig4_heatmap():
    metrics  = ["hit1", "hit3", "hit5", "mrr"]
    m_labels = ["Hit@1", "Hit@3", "Hit@5", "MRR"]
    datasets = ["O", "GE", "ALL"]

    data = np.array([
        [COMBINED[ds][m]["mrr"] for ds in datasets]
        for m in METHODS
    ])

    fig, ax = plt.subplots(figsize=(5.5, 3.8))
    im = ax.imshow(data, cmap="RdYlGn", vmin=0, vmax=1, aspect="auto")

    ax.set_xticks(range(len(datasets)))
    ax.set_xticklabels(["In-Dist (O)", "Cross-Concept (GE)", "Combined"], fontsize=11)
    ax.set_yticks(range(len(METHODS)))
    ax.set_yticklabels(LABELS, fontsize=10)

    for i in range(len(METHODS)):
        for j in range(len(datasets)):
            val = data[i, j]
            color = "white" if val > 0.7 or val < 0.25 else "black"
            ax.text(j, i, f"{val:.2f}", ha="center", va="center",
                    fontsize=12, fontweight="bold", color=color)

    # 标注 "Ours" 行
    ax.add_patch(mpatches.FancyBboxPatch(
        (-0.5, 2.5), len(datasets), 1.0,
        boxstyle="round,pad=0.05",
        linewidth=2, edgecolor=METHOD_COLORS["rrf_path_rr"],
        facecolor="none", zorder=5,
    ))

    ax.spines[:].set_visible(False)
    plt.colorbar(im, ax=ax, label="MRR@5", fraction=0.03, pad=0.02, shrink=0.85)
    ax.set_title("MRR@5 Summary Heatmap", pad=10)

    fig.tight_layout()
    out = OUT / "fig4_heatmap.svg"
    fig.savefig(out)
    plt.close(fig)
    print(f"✅ {out}")


# ─────────────────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    print(f"输出目录: {OUT}\n")
    fig1_mrr_comparison()
    fig2_hitk_curves()
    fig3_graph_benchmark()
    fig4_heatmap()
    print("\n全部完成 ✅")
