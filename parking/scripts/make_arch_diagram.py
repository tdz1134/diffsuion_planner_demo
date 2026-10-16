# -*- coding: utf-8 -*-
"""生成泊车扩散规划器结构框图 (PNG + PDF/SVG 矢量)。

遵循 CVPR/ICLR/NeurIPS 论文插图规范:
- 白底优先, 最多 3 个强调色(Okabe-Ito 色盲安全), 描边+浅填充而非彩底块, 灰度打印可辨;
- 主数据流实线, 共享/次要连接虚线; 左->右一致; 同级块严格对齐等距;
- 扩散模型惯例: (a) Training / (b) Inference 双面板, 共享条件编码器;
- 字号偏大、框内紧凑(高字号 / 小框), 便于单栏阅读。
标签用英文避免缺 CJK 字体的方框问题。
run: PYTHONPATH=. venv_py38/bin/python -m parking.scripts.make_arch_diagram
"""
import os
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import FancyBboxPatch
from matplotlib.lines import Line2D

plt.rcParams.update({"font.family": "DejaVu Sans", "font.size": 10})

# ---- Okabe-Ito (colorblind-safe) ------------------------------------------
BLUE   = "#0072B2"   # network / inference
ORANGE = "#D55E00"   # operation / output
GREEN  = "#009E73"   # input / data
GREY   = "#8c8c8c"
DGREY  = "#2b2b2b"

# font sizes (pt)
S_PANEL = 13.0
S_TITLE = 10.5
S_BODY  = 8.5
S_SMALL = 7.5
S_LEG   = 8.5

# ---- primitives ------------------------------------------------------------
def rbox(ax, x, y, w, h, ec=BLUE, fc="white", lw=1.3, ls="-", r=1.0, z=2):
    ax.add_patch(FancyBboxPatch((x, y), w, h,
                 boxstyle=f"round,pad=0,rounding_size={r}",
                 linewidth=lw, edgecolor=ec, facecolor=fc, linestyle=ls, zorder=z))

def text(ax, x, y, s, size=10, weight="normal", color=DGREY,
         ha="center", va="center", style="normal", z=3):
    ax.text(x, y, s, fontsize=size, fontweight=weight, color=color,
            ha=ha, va=va, style=style, zorder=z, linespacing=1.3)

def node(ax, x, y, w, h, title, body="", ec=BLUE, fc="white", ls="-",
         ts=S_TITLE, bs=S_BODY, tcolor=DGREY):
    rbox(ax, x, y, w, h, ec=ec, fc=fc, ls=ls)
    if body:
        text(ax, x + w / 2, y + h * 0.71, title, size=ts, weight="bold", color=tcolor)
        text(ax, x + w / 2, y + h * 0.32, body, size=bs, color=GREY)
    else:
        text(ax, x + w / 2, y + h / 2, title, size=ts, weight="bold", color=tcolor)

def arrow(ax, p0, p1, color=GREY, lw=1.3, ls="-", z=1):
    ax.annotate("", xy=p1, xytext=p0,
                arrowprops=dict(arrowstyle="-|>", color=color, lw=lw, ls=ls,
                                shrinkA=0, shrinkB=0, mutation_scale=12), zorder=z)

def elbow(ax, p0, p1, color=GREY, lw=1.3, ls="-", orient="hv", z=1):
    (x0, y0), (x1, y1) = p0, p1
    mid = (x1, y0) if orient == "hv" else (x0, y1)
    ax.add_line(Line2D([x0, mid[0]], [y0, mid[1]], color=color, lw=lw, ls=ls, zorder=z))
    arrow(ax, mid, p1, color=color, lw=lw, ls=ls, z=z)

# ---- canvas ----------------------------------------------------------------
fig, ax = plt.subplots(figsize=(12.6, 8.2))
ax.set_xlim(0, 100); ax.set_ylim(0, 100); ax.axis("off")

# global legend (top-right)
def swatch(ax, x, y, color, ls="-", label=""):
    rbox(ax, x, y, 2.8, 2.8, ec=color, fc="white", lw=1.8, ls=ls)
    text(ax, x + 3.6, y + 1.4, label, size=S_LEG, color=GREY, ha="left")
lg_y = 97.2
swatch(ax, 37, lg_y, GREEN,  "-",  "input / data")
swatch(ax, 54, lg_y, BLUE,   "-",  "network")
swatch(ax, 69, lg_y, ORANGE, "-",  "operation / output")
swatch(ax, 87.5, lg_y, GREY, "--", "loss / shared")

# divider between panels
ax.add_line(Line2D([1, 99], [52, 52], color="#d9d9d9", lw=1.0, ls=(0, (5, 5)), zorder=1))

# ===========================================================================
# (a) TRAINING
# ===========================================================================
text(ax, 1, 96.5, "(a)  Training", size=S_PANEL, weight="bold", color=DGREY, ha="left")

# condition encoders (top row, left -> right)
node(ax, 3, 86, 15, 8, "Map grid", "maps (B,2,H,W)", ec=GREEN)
node(ax, 20, 86, 16, 8, "Frozen Map VAE", "z (B,8,9,16)", ec=BLUE, ls="--")
node(ax, 39, 86, 17, 8, "MapConditioner", "map_emb (B,64)", ec=BLUE)
node(ax, 60, 86, 16, 8, "start / goal", "start,goal (B,4)", ec=GREEN)

# hero: denoiser (hand-drawn for a tight inset)
rbox(ax, 34, 64, 40, 14, ec=BLUE)
text(ax, 54, 75.2, "trans2 conditional denoiser  (\u03b5-pred, DiT-style)", size=11, weight="bold", color=DGREY)
text(ax, 54, 71.6, "x_t, t, map_emb, start, goal  \u2192  \u03b5\u0302 (B,80,dim)", size=S_BODY, color=GREY)
rbox(ax, 41, 65.4, 26, 4.8, ec=BLUE, fc="#f2f8fd", ls=":", r=0.8, z=3)
text(ax, 54, 68.8, "\u00d74  _CondBlock", size=S_BODY, weight="bold", color=BLUE, z=4)
text(ax, 54, 66.4, "self-attn \u00b7 cross-attn(cond) \u00b7 FFN \u00b7 AdaLN-Zero", size=S_SMALL, color=GREY, z=4)

# left inputs: noisy state + timestep
node(ax, 3, 70, 17, 8, "x_t  (noisy traj)", "(B,80,dim)  = q(x\u2080,\u03b5,t)", ec=BLUE)
node(ax, 3, 58, 17, 8, "timestep t", "\u2192 temb (B,64)", ec=BLUE)

# right outputs: prediction + loss
node(ax, 79, 70, 16, 8, "pred = \u03b5\u0302", "(B,80,dim)", ec=ORANGE)
node(ax, 79, 58, 16, 8, "L = MSE(\u03b5\u0302, \u03b5)", "\u03b5 ~ N(0,I)", ec=GREY, ls="--")

# arrows (a)
arrow(ax, (18, 90), (20, 90)); arrow(ax, (36, 90), (39, 90))          # encoder chain
arrow(ax, (47.5, 86), (47.5, 78.2))                                   # map_emb -> denoiser
arrow(ax, (68, 86), (68, 78.2))                                       # start/goal -> denoiser
arrow(ax, (20, 74), (34, 74))                                         # x_t -> denoiser
arrow(ax, (20, 62), (34, 62))                                         # temb -> denoiser
arrow(ax, (74, 74), (79, 74))                                         # denoiser -> pred
arrow(ax, (87, 70), (87, 66))                                         # pred -> loss

# ===========================================================================
# (b) INFERENCE
# ===========================================================================
text(ax, 1, 45.5, "(b)  Inference  (DDPM reverse sampling, T=200)",
     size=S_PANEL, weight="bold", color=DGREY, ha="left")

node(ax, 3, 27, 14, 9, "shared condition", "map_emb (B,64)\nstart,goal (B,4)", ec=GREEN, ls="--", bs=S_SMALL)
node(ax, 3, 11, 14, 9, "x_T ~ N(0,I)", "(B,80,dim)", ec=GREEN)
node(ax, 20, 25, 19, 14, "DDPM reverse step  \u00d7T", "trans2 denoiser (shared)\n+ posterior \u2192 x_{t\u22121}\npin endpoints (pose)", ec=BLUE, bs=S_SMALL)
node(ax, 42, 27, 14, 9, "gen", "(B,80,dim)", ec=BLUE)
node(ax, 59, 24, 22, 15, "post-process", "pose: denormalize \u2192 traj\nctrl: integrate_ctrl \u2192 traj\n(+ M18b goal-refine)", ec=ORANGE, bs=S_SMALL)
node(ax, 84, 27, 14, 9, "traj", "(B,80,4) world", ec=ORANGE)
node(ax, 84, 5, 14, 10, "repair + evaluate", "collision \u00b7 slip\ncurvature\nend-pos & ang err", ec=ORANGE, bs=S_SMALL)

# arrows (b)
arrow(ax, (17, 31.5), (20, 31.5))                                     # condition -> step
elbow(ax, (17, 15.5), (24, 25), orient="hv")                          # x_T -> step (up into bottom)
arrow(ax, (39, 31.5), (42, 31.5))                                     # step -> gen
arrow(ax, (56, 31.5), (59, 31.5))                                     # gen -> post
arrow(ax, (81, 31.5), (84, 31.5))                                     # post -> traj
arrow(ax, (91, 27), (91, 15))                                         # traj -> eval

fig.tight_layout()
base = os.path.abspath(os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
                                    "..", "figs", "parking"))
os.makedirs(base, exist_ok=True)
png = os.path.join(base, "architecture.png")
fig.savefig(png, dpi=200, bbox_inches="tight")
fig.savefig(os.path.join(base, "architecture.pdf"), bbox_inches="tight")
fig.savefig(os.path.join(base, "architecture.svg"), bbox_inches="tight")
plt.close(fig)
print("[saved]", png)
