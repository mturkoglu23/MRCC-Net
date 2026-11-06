"""Figure 1 (v3, English) - MRCC-Net overview.

Layout, left to right:
  (1) pre-processing: the input clip gives the full-frame view directly and the child-centric view
      through YOLO11x-pose tracking;
  (2) frozen feature extraction: each view feeds its four streams (VideoMAE-L at 4/8/16 fps,
      V-JEPA 2-L at 8 fps) through one bus; every stream row shows sampling -> frozen encoder ->
      pooled 1024-d descriptor;
  (3) the eight descriptors are concatenated (rows line up with their blocks) and passed to the
      only trained component, a standardised L2 logistic regression; clip probabilities are
      averaged per video.
All connectors are straight/orthogonal and share one style.
Output: results/figures/fig_architecture_v3.{png,pdf}
"""
import matplotlib.pyplot as plt
from matplotlib.patches import Arc, Circle, FancyArrowPatch, FancyBboxPatch, Polygon, Rectangle

from common import RESULTS

BLUE, BLUE_L, BLUE_D = "#2a78d6", "#dbe9fb", "#1b4f91"
VIO, VIO_L, VIO_D = "#6250d6", "#e6e2fa", "#3f3296"
AQUA = "#139e6d"
ORANGE, ORANGE_L = "#e8632b", "#fde3d8"
INK, MUTED, LINE, PANEL, DARK = "#1a1a1a", "#5a5955", "#4a4a48", "#f4f3ef", "#1c2b4a"
FIG_C, WALL, FLOOR = "#3b3b3b", "#f1f0ec", "#dedcd4"

plt.rcParams.update({"font.family": "Arial", "font.size": 6, "mathtext.fontset": "stixsans"})

fig, ax = plt.subplots(figsize=(7.2, 3.75))
ax.set_xlim(0, 144)
ax.set_ylim(-3.5, 71.5)
ax.set_aspect("equal")
ax.axis("off")


# ------------------------------------------------------------------ primitives
def rbox(x, y, w, h, fc="white", ec=LINE, lw=0.7, r=1.0, ls="-", z=2):
    ax.add_patch(FancyBboxPatch((x, y), w, h, boxstyle=f"round,pad=0,rounding_size={r}", fc=fc, ec=ec, lw=lw,
                                ls=ls, zorder=z))


def txt(x, y, s, fs=6.0, c=INK, ha="center", va="center", w="normal", st="normal", z=9):
    ax.text(x, y, s, fontsize=fs, color=c, ha=ha, va=va, weight=w, style=st, zorder=z, linespacing=1.15)


def arrow(pts, c=LINE, lw=0.75, ms=5.5, z=6):
    """orthogonal polyline; arrow head on the last segment"""
    for (x1, y1), (x2, y2) in zip(pts[:-2], pts[1:-1]):
        ax.plot([x1, x2], [y1, y2], color=c, lw=lw, zorder=z, solid_capstyle="butt")
    ax.add_patch(FancyArrowPatch(pts[-2], pts[-1], arrowstyle="-|>", mutation_scale=ms, color=c, lw=lw,
                                 shrinkA=0, shrinkB=0, zorder=z))


def lock(x, y, s=0.62, c=MUTED):
    ax.add_patch(Rectangle((x - 1.2 * s, y - 1.0 * s), 2.4 * s, 1.85 * s, fc=c, ec="none", zorder=10))
    ax.add_patch(Arc((x, y + 0.85 * s), 1.5 * s, 1.9 * s, theta1=0, theta2=180, color=c, lw=0.7, zorder=10))


def person(cx, by, h, c=FIG_C, z=7):
    """filled pictogram of a child with raised, flapping arms (feet at by, height h)"""
    lw = 0.26 * h            # stroke width in points scales with the figure
    hr = 0.10 * h
    ax.add_patch(Circle((cx, by + h - hr), hr, fc=c, ec="none", zorder=z))
    neck, hip, sh = by + h - 2.25 * hr, by + 0.44 * h, by + 0.70 * h
    kw = dict(color=c, lw=lw, solid_capstyle="round", solid_joinstyle="round", zorder=z)
    ax.plot([cx, cx], [neck, hip], **kw)
    for s in (-1, 1):
        ax.plot([cx, cx + s * 0.17 * h, cx + s * 0.26 * h], [sh, sh + 0.10 * h, sh + 0.25 * h], **kw)
        ax.plot([cx, cx + s * 0.13 * h], [hip, by + 0.03 * h], **kw)


def scene(x, y, w, h, child_h, child_x, z=3):
    """room thumbnail: wall, floor and a window"""
    ax.add_patch(Rectangle((x, y), w, h, fc=WALL, ec="none", zorder=z))
    ax.add_patch(Rectangle((x, y), w, 0.3 * h, fc=FLOOR, ec="none", zorder=z))
    ax.add_patch(Rectangle((x + 0.08 * w, y + 0.5 * h), 0.22 * w, 0.34 * h, fc="white", ec="#c9c7bf", lw=0.4,
                           zorder=z))
    person(child_x, y + 0.14 * h, child_h, z=z + 2)


def panel(x, y, w, h, n, title):
    rbox(x, y, w, h, fc=PANEL, ec="none", r=1.8, z=0)
    ax.add_patch(Circle((x + 3.2, y + h - 3.1), 2.0, fc=DARK, ec="none", zorder=3))
    txt(x + 3.2, y + h - 3.15, str(n), fs=6.2, c="white", w="bold")
    txt(x + 6.3, y + h - 3.1, title, fs=6.8, w="bold", ha="left")


# ------------------------------------------------------------------ geometry
FULL_ROWS, CHILD_ROWS = [54.0, 49.0, 44.0, 39.0], [29.5, 24.5, 19.5, 14.5]
STREAMS = [("4 fps", 4, "vm"), ("8 fps", 2, "vm"), ("16 fps", 1, "vm"), ("8 fps", 2, "vj")]
YC_FULL, YC_CHILD = sum(FULL_ROWS) / 4, sum(CHILD_ROWS) / 4          # 46.5, 22.0

panel(0.5, 4.5, 44.0, 66.5, 1, "Pre-processing")
panel(46.0, 4.5, 57.5, 66.5, 2, "Frozen multi-rate feature extraction")
panel(105.0, 4.5, 38.5, 66.5, 3, "Fusion and trainable head")

# ------------------------------------------------------------------ (1) pre-processing
# input clip: three stacked frames
for k, (dx, a) in enumerate([(2.4, 0.45), (1.2, 0.7), (0.0, 1.0)]):
    x0, y0 = 3.0 + dx, YC_FULL - 5.0 + dx
    rbox(x0, y0, 13.0, 9.6, fc="white", ec="#9b9a95", lw=0.5, r=0.4, z=3 + k)
    if k == 2:
        scene(x0 + 0.4, y0 + 0.4, 12.2, 8.8, 4.6, x0 + 8.0, z=6)
txt(9.5, YC_FULL - 7.8, "input clip (16 fps)", fs=5.6, c=MUTED)

# YOLO tracking box
rbox(2.2, YC_CHILD - 5.0, 16.6, 10.0, fc="white", ec=AQUA, lw=0.9, r=0.8, z=3)
txt(10.5, YC_CHILD + 2.3, "YOLO11x-pose", fs=5.6, w="bold")
txt(10.5, YC_CHILD - 0.6, "child detection", fs=5.0, c=MUTED)
txt(10.5, YC_CHILD - 2.9, "and tracking", fs=5.0, c=MUTED)
arrow([(9.5, YC_FULL - 9.6), (9.5, YC_CHILD + 5.2)])

# the two views
TX, TW, TH = 28.4, 14.4, 10.8
for yc, kind in ((YC_FULL, "full"), (YC_CHILD, "child")):
    x0, y0 = TX, yc - TH / 2
    if kind == "full":
        scene(x0, y0, TW, TH, 5.0, x0 + 9.4, z=3)
        ax.add_patch(Rectangle((x0 + 7.3, y0 + 1.0), 4.2, 6.2, fc="none", ec=AQUA, lw=0.7, ls=(0, (2, 1)),
                               zorder=8))
        rbox(x0, y0, TW, TH, fc="none", ec="#9b9a95", lw=0.6, r=0.3, z=9)
    else:
        scene(x0, y0, TW, TH, 8.6, x0 + TW / 2 + 1.0, z=3)
        rbox(x0, y0, TW, TH, fc="none", ec=AQUA, lw=1.0, r=0.3, z=9)
txt(TX + TW / 2, YC_FULL - TH / 2 - 2.4, "full frame", fs=6.0, w="bold")
txt(TX + TW / 2, YC_CHILD - TH / 2 - 2.4, "child-centric crop", fs=6.0, w="bold", c=AQUA)
arrow([(16.6, YC_FULL), (TX - 0.4, YC_FULL)])
arrow([(19.0, YC_CHILD), (TX - 0.4, YC_CHILD)])
txt(23.6, YC_CHILD + 1.8, "square crop", fs=4.3, c=MUTED)
txt(23.6, YC_CHILD - 1.9, "boxes ×1.25", fs=4.3, c=MUTED)
txt(23.4, YC_FULL + 1.9, "whole frame", fs=4.4, c=MUTED)
txt(22.5, 9.4, "each view is also horizontally flipped;\ndescriptors are averaged over both", fs=4.9, c=MUTED,
    st="italic")

# ------------------------------------------------------------------ (2) frozen streams
X_BUS, X_RATE, X_TICK, X_ENC, W_ENC, X_DESC = 48.4, 50.2, 57.4, 75.4, 16.2, 94.0
txt(61.4, 63.0, "temporal sampling", fs=5.8, w="bold")
txt(61.4, 60.4, "16-frame windows, 0.5 s hop", fs=4.8, c=MUTED)
txt(X_ENC + W_ENC / 2, 63.0, "frozen encoder", fs=5.8, w="bold")
txt(X_ENC + W_ENC / 2, 60.4, "no weight updates", fs=4.8, c=MUTED)
txt(98.0, 63.0, "descriptor", fs=5.8, w="bold")
txt(98.0, 60.4, "1024-d", fs=4.8, c=MUTED)
ax.plot([47.5, 102.8], [58.2, 58.2], color="#d3d2cc", lw=0.5, zorder=1)
ax.plot([47.5, 102.8], [34.25, 34.25], color="#d3d2cc", lw=0.5, ls=(0, (3, 2)), zorder=1)

for rows, yc in ((FULL_ROWS, YC_FULL), (CHILD_ROWS, YC_CHILD)):
    # bus from the view: one arrow in, one stub per stream
    ax.plot([TX + TW + 0.3, X_BUS], [yc, yc], color=LINE, lw=0.75, zorder=6)
    ax.plot([X_BUS, X_BUS], [rows[-1], rows[0]], color=LINE, lw=0.75, zorder=6)
    for (rate, sec, bb), y in zip(STREAMS, rows):
        c, cl, cd = (BLUE, BLUE_L, BLUE_D) if bb == "vm" else (VIO, VIO_L, VIO_D)
        arrow([(X_BUS, y), (X_RATE - 0.3, y)], ms=4.5)
        txt(X_RATE, y, rate, fs=5.6, w="bold", ha="left")
        span = 3.0 * sec
        for k in range(16):
            xi = X_TICK + k * span / 15
            ax.plot([xi, xi], [y - 1.15, y + 1.15], color=INK, lw=0.35, zorder=5)
        txt(X_TICK + 12.9, y, f"{sec} s", fs=5.0, c=MUTED, ha="left")
        arrow([(X_TICK + 15.5, y), (X_ENC - 0.2, y)], c=c, ms=4.5)
        rbox(X_ENC, y - 1.85, W_ENC, 3.7, fc=cl, ec=c, lw=0.8, r=0.6, z=4)
        txt(X_ENC + 6.9, y, "VideoMAE-L" if bb == "vm" else "V-JEPA 2-L", fs=5.4, w="bold", c=cd)
        lock(X_ENC + W_ENC - 1.9, y - 0.1)
        arrow([(X_ENC + W_ENC + 0.2, y), (X_DESC - 0.2, y)], c=c, ms=4.5)
        for k in range(6):
            ax.add_patch(Rectangle((X_DESC + k * 1.4, y - 1.25), 1.4, 2.5, fc=c, ec="white", lw=0.3,
                                   alpha=0.38 + 0.14 * (k % 3), zorder=5))
txt(75.0, 8.6, "8 streams, 2 frozen models; rows of the same colour share one set of weights:\n"
    "VideoMAE-L (ViT-L/16, Kinetics-400) · V-JEPA 2-L (ViT-L, SSv2)", fs=4.7, c=MUTED, st="italic")

# ------------------------------------------------------------------ (3) fusion + head
X_CAT, W_CAT = 107.4, 4.4
ALL = list(zip(FULL_ROWS, STREAMS)) + list(zip(CHILD_ROWS, STREAMS))
for y, (rate, sec, bb) in ALL:
    c = BLUE if bb == "vm" else VIO
    ax.plot([X_DESC + 8.6, X_CAT - 0.7], [y, y], color=c, lw=0.55, ls=(0, (1.2, 1.0)), zorder=4)
    ax.add_patch(Rectangle((X_CAT, y - 2.3), W_CAT, 4.6, fc=c, ec="white", lw=0.7, alpha=0.8, zorder=5))
rbox(X_CAT - 0.5, CHILD_ROWS[-1] - 2.8, W_CAT + 1.0, FULL_ROWS[0] - CHILD_ROWS[-1] + 5.6, fc="none", ec=LINE,
     lw=0.6, r=0.6, z=6)
txt(X_CAT + W_CAT / 2, 63.0, "concat", fs=5.8, w="bold")
txt(X_CAT + W_CAT / 2, 60.4, "8 × 1024", fs=4.9, c=MUTED)
txt(X_CAT + W_CAT / 2, 8.4, r"$\mathbf{z}\in\mathbb{R}^{8192}$", fs=6.0)

YM = 45.0                                                       # head box centre
HX, HW = 116.4, 25.4
rbox(HX, YM - 6.6, HW, 14.6, fc=ORANGE_L, ec=ORANGE, lw=1.1, r=0.9, z=4)
txt(HX + HW / 2, YM + 5.3, "standardisation", fs=5.3, c=MUTED)
txt(HX + HW / 2, YM + 3.2, "↓", fs=5.0, c=MUTED)
txt(HX + HW / 2, YM + 0.9, "L2 logistic regression", fs=5.9, w="bold")
txt(HX + HW / 2, YM - 2.0, "≈33k trainable parameters", fs=5.1, c=ORANGE, w="bold")
txt(HX + HW / 2, YM - 4.6, "C = 0.05, class-balanced", fs=4.8, c=MUTED)
arrow([(X_CAT + W_CAT + 0.6, YM), (HX - 0.2, YM)], lw=0.9, ms=6)

txt(HX + HW / 2, 63.0, "trained classifier", fs=5.8, w="bold")
txt(HX + HW / 2, 60.4, "fit on training folds only", fs=4.8, c=MUTED)
arrow([(HX + HW / 2, YM - 6.8), (HX + HW / 2, 35.4)], lw=0.75)
txt(HX + HW / 2, 33.6, "clip-level probabilities", fs=5.4, w="bold")
for i, (lab, p) in enumerate(zip(["Arm flapping", "Hand action*", "Head banging", "Spinning"],
                                 [0.84, 0.08, 0.05, 0.03])):
    y = 30.4 - i * 2.8
    txt(HX + 11.4, y, lab, fs=5.0, ha="right", w="bold" if i == 0 else "normal")
    ax.add_patch(Rectangle((HX + 12.2, y - 1.0), 12.6, 2.0, fc="#e4e3de", ec="none", zorder=4))
    ax.add_patch(Rectangle((HX + 12.2, y - 1.0), 12.6 * p, 2.0, fc=ORANGE if i == 0 else "#bdbbb4", ec="none",
                           zorder=5))
arrow([(HX + HW / 2, 20.0), (HX + HW / 2, 17.6)], lw=0.75)
rbox(HX, 8.6, HW, 8.8, fc="white", ec=LINE, lw=0.7, r=0.8, z=4)
txt(HX + HW / 2, 14.9, "video-level decision", fs=5.6, w="bold")
txt(HX + HW / 2, 12.3, "mean of the clip probabilities", fs=4.7, c=MUTED)
txt(HX + HW / 2, 10.4, "of each source video", fs=4.7, c=MUTED)
txt(HX + HW - 0.2, 6.3, "* ESBD only", fs=4.4, c=MUTED, ha="right")

# ------------------------------------------------------------------ legend
lx, ly = 2.0, -1.4
for kind, lab in [("lock", "frozen (not updated)"), ("train", "trained"), ("vm", "VideoMAE-L stream"),
                  ("vj", "V-JEPA 2-L stream"), ("box", "child box / crop")]:
    if kind == "lock":
        lock(lx + 1.2, ly - 0.05, 0.7)
    elif kind == "box":
        ax.add_patch(Rectangle((lx, ly - 1.1), 2.6, 2.2, fc="none", ec=AQUA, lw=0.8, ls=(0, (2, 1)), zorder=6))
    else:
        fc, ec = {"train": (ORANGE_L, ORANGE), "vm": (BLUE_L, BLUE), "vj": (VIO_L, VIO)}[kind]
        rbox(lx, ly - 1.1, 2.6, 2.2, fc=fc, ec=ec, lw=0.8, r=0.3, z=6)
    txt(lx + 3.5, ly, lab, fs=5.2, c=MUTED, ha="left")
    lx += 7.0 + len(lab) * 0.95

out = RESULTS / "figures"
fig.savefig(out / "fig_architecture_v3.png", dpi=600, bbox_inches="tight", pad_inches=0.03)
fig.savefig(out / "fig_architecture_v3.pdf", bbox_inches="tight", pad_inches=0.03)
print("ok")
