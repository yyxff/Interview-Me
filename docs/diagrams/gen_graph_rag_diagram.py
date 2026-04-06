"""
gen_graph_rag_diagram.py
========================
生成知识图谱 Graph RAG 示意图（学术 poster 风格，白底）。

配色与字体和 gen_metrics.py 保持一致：
  白底 / Times New Roman serif / 无网格 / 论文风格配色

图示核心论点：
  Vector Search 仅召回语义最近邻（1 个节点）；
  Graph RAG 沿关系边逐跳扩展，召回更多上下文（+6 节点），路径可解释。

输出：docs/diagrams/fig5_graph_rag_diagram.svg
运行：python docs/diagrams/gen_graph_rag_diagram.py
"""

import math
from pathlib import Path

OUT = Path(__file__).parent / "fig5_graph_rag_diagram.svg"
W, H = 820, 540

# ── 配色（与 gen_metrics.py METHOD_COLORS 同色系）────────────────────────────
C_BG         = "#ffffff"
C_VECTOR     = "#4878d0"   # 直接命中（blue，同 bi 方法色）
C_HOP1       = "#8c7be8"   # 1-hop 召回（purple，同 graph 方法色）
C_HOP2       = "#b8b0f0"   # 2-hop 召回（lighter purple）
C_NONE_FILL  = "#f7f7f7"   # 未命中节点填充
C_NONE_STR   = "#cccccc"   # 未命中节点描边
C_NONE_TXT   = "#999999"   # 未命中节点文字
C_HIT_TXT    = "#ffffff"   # 命中节点文字
C_EDGE_ACT   = "#4878d0"   # 活跃遍历边
C_EDGE_HOP2  = "#8c7be8"   # hop2 边
C_EDGE_IDLE  = "#e0e0e0"   # 未遍历边
C_LBL_ACT    = "#555555"   # 活跃边标签
C_LBL_IDLE   = "#cccccc"   # 未遍历边标签
C_QUERY      = "#ee854a"   # Query 节点（orange，同 rrf 方法色）
C_QUERY_STR  = "#dd6b20"
C_TITLE      = "#1a1a2e"   # 标题颜色
FONT         = "Times New Roman, Georgia, serif"

# ── 节点定义 ──────────────────────────────────────────────────────────────────
# tag: "vector" | "hop1" | "hop2" | "none"
# 位置刻意不规律，边数更多，角度更自由
NODES = {
    "vm":    dict(label=["Trans-","former"],     x=390, y=248, r=34, tag="vector"),
    "pt":    dict(label=["Attention"],           x=238, y=152, r=28, tag="hop1"),
    "tlb":   dict(label=["Multi-","Head"],       x=112, y=88,  r=22, tag="hop2"),
    "heap":  dict(label=["FFN"],                x=248, y=308, r=24, tag="hop1"),
    "ks":    dict(label=["Encoder"],             x=518, y=328, r=28, tag="hop1"),
    "mmap":  dict(label=["Layer","Norm"],        x=155, y=390, r=22, tag="hop2"),
    "sc":    dict(label=["Pos.","Enc."],         x=635, y=415, r=22, tag="hop2"),
    "mi":    dict(label=["Decoder"],             x=535, y=158, r=27, tag="none"),
    "proc":  dict(label=["Cross","Attn"],        x=668, y=240, r=23, tag="none"),
    "fork":  dict(label=["Self","Attn"],         x=460, y=82,  r=20, tag="none"),
    "ctx":   dict(label=["Masking"],             x=680, y=145, r=23, tag="none"),
    "sched": dict(label=["Softmax"],             x=720, y=330, r=22, tag="none"),
    "seg":   dict(label=["Dropout"],             x=128, y=205, r=22, tag="none"),
    "stack": dict(label=["Residual"],            x=418, y=405, r=21, tag="none"),
    "usr":   dict(label=["Embed-","ding"],       x=318, y=412, r=23, tag="none"),
}

# (src, tgt, label, style)  style: "active"|"hop2"|"idle"
EDGES = [
    # ── 活跃遍历路径（Graph RAG 召回）─────────────────────
    ("vm",   "pt",    "core of",     "active"),
    ("vm",   "ks",    "built from",  "active"),
    ("vm",   "heap",  "contains",    "active"),
    ("pt",   "tlb",   "extended as", "hop2"),
    ("ks",   "sc",    "uses",        "hop2"),
    ("heap", "mmap",  "wrapped by",  "hop2"),
    # ── 未遍历边（体现图的丰富性）─────────────────────────
    ("vm",   "mi",    "paired with", "idle"),
    ("vm",   "seg",   "trained with","idle"),
    ("pt",   "seg",   "uses",        "idle"),
    ("heap", "stack", "followed by", "idle"),
    ("ks",   "usr",   "inputs",      "idle"),
    ("mi",   "proc",  "contains",    "idle"),
    ("mi",   "usr",   "uses",        "idle"),
    ("proc", "fork",  "type of",     "idle"),
    ("proc", "ctx",   "applies",     "idle"),
    ("proc", "sched", "normalised by","idle"),
    ("ctx",  "sched", "before",      "idle"),
    ("usr",  "stack", "added via",   "idle"),
    ("fork", "mi",    "within",      "idle"),
    ("sched","ks",    "inside",      "idle"),
]

# ── 几何工具 ──────────────────────────────────────────────────────────────────

def unit_vec(x1, y1, x2, y2):
    dx, dy = x2 - x1, y2 - y1
    L = math.hypot(dx, dy) or 1.0
    return dx / L, dy / L

def edge_pts(src, tgt, gap=11):
    x1, y1 = src["x"], src["y"]
    x2, y2 = tgt["x"], tgt["y"]
    ux, uy = unit_vec(x1, y1, x2, y2)
    r1 = src["r"] + 2
    r2 = tgt["r"] + gap
    return x1 + ux*r1, y1 + uy*r1, x2 - ux*r2, y2 - uy*r2

def label_offset(x1, y1, x2, y2, dist=10):
    """Edge label: midpoint + perpendicular offset."""
    mx, my = (x1+x2)/2, (y1+y2)/2
    ux, uy = unit_vec(x1, y1, x2, y2)
    return mx - uy*dist, my + ux*dist

# ── SVG 构建 ──────────────────────────────────────────────────────────────────

parts = []
e = parts.append

e(f'<svg xmlns="http://www.w3.org/2000/svg" '
  f'width="{W}" height="{H}" viewBox="0 0 {W} {H}">')

# ── defs ──────────────────────────────────────────────────────────────────────
e(f"""<defs>
  <marker id="arr-active" viewBox="0 0 10 8" refX="9" refY="4"
          markerWidth="7" markerHeight="7" orient="auto"
          markerUnits="userSpaceOnUse">
    <path d="M0,0 L10,4 L0,8 Z" fill="{C_EDGE_ACT}"/>
  </marker>
  <marker id="arr-hop2" viewBox="0 0 10 8" refX="9" refY="4"
          markerWidth="6" markerHeight="6" orient="auto"
          markerUnits="userSpaceOnUse">
    <path d="M0,0 L10,4 L0,8 Z" fill="{C_EDGE_HOP2}"/>
  </marker>
  <marker id="arr-query" viewBox="0 0 10 8" refX="9" refY="4"
          markerWidth="7" markerHeight="7" orient="auto"
          markerUnits="userSpaceOnUse">
    <path d="M0,0 L10,4 L0,8 Z" fill="{C_QUERY_STR}"/>
  </marker>
</defs>""")

# ── 背景 ──────────────────────────────────────────────────────────────────────
e(f'<rect width="{W}" height="{H}" fill="{C_BG}"/>')

# 细边框
e(f'<rect x="1" y="1" width="{W-2}" height="{H-2}" '
  f'fill="none" stroke="#e8e8e8" stroke-width="1"/>')

# ── 标题 ──────────────────────────────────────────────────────────────────────
e(f'<text x="{W//2}" y="34" text-anchor="middle" '
  f'font-family="{FONT}" font-size="17" font-weight="bold" fill="{C_TITLE}">'
  f'Graph RAG: Relation-Aware Retrieval Expansion</text>')

e(f'<text x="{W//2}" y="54" text-anchor="middle" '
  f'font-family="{FONT}" font-size="12" fill="#666666" font-style="italic">'
  f'Query: &#x201C;How does the Transformer model process sequences?&#x201D;</text>')

# ── Query 节点 → VM 箭头 ──────────────────────────────────────────────────────
qx, qy = 50, 248
vm = NODES["vm"]
ux, uy = unit_vec(qx, qy, vm["x"], vm["y"])
ax1, ay1 = qx + ux*38, qy + uy*38
ax2, ay2 = vm["x"] - ux*(vm["r"]+11), vm["y"] - uy*(vm["r"]+11)

# query 节点（圆角矩形）
e(f'<rect x="{qx-34}" y="{qy-16}" width="68" height="32" rx="6" '
  f'fill="{C_QUERY}" stroke="{C_QUERY_STR}" stroke-width="1.5"/>')
e(f'<text x="{qx}" y="{qy-4}" text-anchor="middle" '
  f'font-family="{FONT}" font-size="11" font-weight="bold" fill="white">Query</text>')
e(f'<text x="{qx}" y="{qy+9}" text-anchor="middle" '
  f'font-family="{FONT}" font-size="9" fill="white" font-style="italic" opacity="0.9">'
  f'semantic</text>')

# query → vm 箭头
e(f'<line x1="{ax1:.1f}" y1="{ay1:.1f}" x2="{ax2:.1f}" y2="{ay2:.1f}" '
  f'stroke="{C_QUERY_STR}" stroke-width="2.2" stroke-dasharray="5,3" '
  f'marker-end="url(#arr-query)"/>')

# ── 所有边（先画，节点在上层）────────────────────────────────────────────────
for src_id, tgt_id, label, style in EDGES:
    src, tgt = NODES[src_id], NODES[tgt_id]
    x1, y1, x2, y2 = edge_pts(src, tgt)

    if style == "active":
        color, width, marker, opacity = C_EDGE_ACT, 2.0, "url(#arr-active)", "1"
        lbl_color = C_LBL_ACT
    elif style == "hop2":
        color, width, marker, opacity = C_EDGE_HOP2, 1.6, "url(#arr-hop2)", "0.9"
        lbl_color = C_LBL_ACT
    else:
        color, width, marker, opacity = C_EDGE_IDLE, 1.0, "none", "0.8"
        lbl_color = C_LBL_IDLE

    e(f'<line x1="{x1:.1f}" y1="{y1:.1f}" x2="{x2:.1f}" y2="{y2:.1f}" '
      f'stroke="{color}" stroke-width="{width}" opacity="{opacity}" '
      f'marker-end="{marker}"/>')

    # 边标签（只为 active/hop2 加，idle 留着但颜色很淡）
    if label:
        lx, ly = label_offset(x1, y1, x2, y2, dist=9)
        font_sz = 9 if style in ("active", "hop2") else 8
        e(f'<text x="{lx:.1f}" y="{ly:.1f}" text-anchor="middle" '
          f'dominant-baseline="central" font-family="{FONT}" '
          f'font-size="{font_sz}" font-style="italic" fill="{lbl_color}" '
          f'opacity="{"0.95" if style != "idle" else "0.45"}">'
          f'{label}</text>')

# ── 节点 ──────────────────────────────────────────────────────────────────────
for nid, n in NODES.items():
    x, y, r, tag = n["x"], n["y"], n["r"], n["tag"]
    label = n["label"]

    if tag == "vector":
        fill, stroke, sw, txt = C_VECTOR, "#3a62b0", "2", C_HIT_TXT
    elif tag == "hop1":
        fill, stroke, sw, txt = C_HOP1, "#6a60d0", "1.5", C_HIT_TXT
    elif tag == "hop2":
        fill, stroke, sw, txt = C_HOP2, "#8c7be8", "1.5", "#3d3270"
    else:
        fill, stroke, sw, txt = C_NONE_FILL, C_NONE_STR, "1", C_NONE_TXT

    e(f'<circle cx="{x}" cy="{y}" r="{r}" '
      f'fill="{fill}" stroke="{stroke}" stroke-width="{sw}"/>')

    fs = 11 if r >= 30 else 10 if r >= 24 else 9
    if len(label) == 1:
        e(f'<text x="{x}" y="{y}" text-anchor="middle" dominant-baseline="central" '
          f'font-family="{FONT}" font-size="{fs}" font-weight="bold" fill="{txt}">'
          f'{label[0]}</text>')
    else:
        gap = fs - 1
        y0 = y - gap / 2
        for i, part in enumerate(label):
            e(f'<text x="{x}" y="{y0 + i*gap:.1f}" text-anchor="middle" '
              f'dominant-baseline="central" font-family="{FONT}" '
              f'font-size="{fs}" font-weight="bold" fill="{txt}">{part}</text>')

# ── 召回数量标注 ──────────────────────────────────────────────────────────────
# "Vector only: 1" 标注在 VM 右上
vx, vy, vr = vm["x"], vm["y"], vm["r"]
e(f'<rect x="{vx+vr-2}" y="{vy-vr-24}" width="84" height="20" rx="4" '
  f'fill="#eef2ff" stroke="#c7d2fe" stroke-width="1"/>')
e(f'<text x="{vx+vr+40}" y="{vy-vr-11}" text-anchor="middle" '
  f'font-family="{FONT}" font-size="10" fill="#4f46e5">'
  f'Vector: 1 result</text>')

# "Graph RAG: 7" 标注在 VM 右下
e(f'<rect x="{vx+vr-2}" y="{vy+vr+4}" width="96" height="20" rx="4" '
  f'fill="#f0fdf4" stroke="#86efac" stroke-width="1"/>')
e(f'<text x="{vx+vr+46}" y="{vy+vr+17}" text-anchor="middle" '
  f'font-family="{FONT}" font-size="10" fill="#16a34a">'
  f'Graph RAG: 7 results</text>')

# ── 图例 ──────────────────────────────────────────────────────────────────────
legend_items = [
    (C_QUERY,    C_QUERY_STR, "Query"),
    (C_VECTOR,   "#3a62b0",   "Vector Match"),
    (C_HOP1,     "#6a60d0",   "1-hop Retrieved"),
    (C_HOP2,     "#8c7be8",   "2-hop Retrieved"),
    (C_NONE_FILL,C_NONE_STR,  "Not Retrieved"),
]

leg_y  = H - 38
n_item = len(legend_items)
item_w = 148
leg_x0 = (W - n_item * item_w) / 2

# 图例背景
e(f'<rect x="{leg_x0 - 12:.0f}" y="{leg_y - 14}" '
  f'width="{n_item * item_w + 24:.0f}" height="28" rx="5" '
  f'fill="#fafafa" stroke="#e8e8e8" stroke-width="1"/>')

for i, (fill, stroke, lbl) in enumerate(legend_items):
    lx = leg_x0 + i * item_w + 8
    ly = leg_y
    e(f'<circle cx="{lx:.0f}" cy="{ly}" r="7" '
      f'fill="{fill}" stroke="{stroke}" stroke-width="1.2"/>')
    e(f'<text x="{lx+12:.0f}" y="{ly}" dominant-baseline="central" '
      f'font-family="{FONT}" font-size="11" fill="#333333">{lbl}</text>')

# ── 边线例（active / idle）────────────────────────────────────────────────────
edge_leg_y = H - 14
# active edge sample
e(f'<line x1="200" y1="{edge_leg_y}" x2="240" y2="{edge_leg_y}" '
  f'stroke="{C_EDGE_ACT}" stroke-width="2" marker-end="url(#arr-active)"/>')
e(f'<text x="244" y="{edge_leg_y}" dominant-baseline="central" '
  f'font-family="{FONT}" font-size="10" fill="#555555">Traversed</text>')

# idle edge sample
e(f'<line x1="330" y1="{edge_leg_y}" x2="370" y2="{edge_leg_y}" '
  f'stroke="{C_EDGE_IDLE}" stroke-width="1"/>')
e(f'<text x="374" y="{edge_leg_y}" dominant-baseline="central" '
  f'font-family="{FONT}" font-size="10" fill="#999999">Not Traversed</text>')

e('</svg>')

# ── 写文件 ────────────────────────────────────────────────────────────────────
OUT.write_text('\n'.join(parts), encoding="utf-8")
print(f"✅  {OUT}  ({W}×{H}px)")
