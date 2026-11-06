"""Figure 1 (v2, English) - MRCC-Net overview as one left-to-right pipeline.

The eight streams are drawn as explicit rows (view x rate x backbone) whose descriptors line up
with their blocks in the concatenated vector, so the "8 streams -> 8192-d" step is visible.
Output: results/figures/fig_architecture_v2.{png,pdf}
"""
import matplotlib.pyplot as plt
from matplotlib.patches import Arc, Circle, FancyArrowPatch, FancyBboxPatch, Rectangle

from common import RESULTS

BLUE, BLUE_L = "#2a78d6", "#dbe9fb"
VIO, VIO_L = "#6250d6", "#e6e2fa"
AQUA = "#139e6d"
ORANGE, ORANGE_L = "#eb6834", "#fde3d8"
INK, MUTED, LINE, PANEL, DARK, GRID = "#0b0b0b", "#52514e", "#8a8984", "#f5f4f0", "#1c2b4a", "#d6d5cf"

FS = 6.4
plt.rcParams.update({"font.family": "DejaVu Sans", "font.size": FS, "mathtext.fontset": "dejavusans"})

fig, ax = plt.subplots(figsize=(7.2, 4.75))
ax.set_xlim(0, 144)
ax.set_ylim(-3, 92)
ax.set_aspect("equal")
ax.axis("off")


def rbox(x, y, w, h, fc="white", ec=LINE, lw=0.8, r=1.4, ls="-", z=1):
    ax.add_patch(FancyBboxPatch((x, y), w, h, boxstyle=f"round,pad=0,rounding_size={r}", fc=fc, ec=ec, lw=lw,
                                ls=ls, zorder=z))


def txt(x, y, s, fs=FS, c=INK, ha="center", va="center", w="normal", st="normal", z=7, rot=0):
    ax.text(x, y, s, fontsize=fs, color=c, ha=ha, va=va, weight=w, style=st, zorder=z, rotation=rot)


def arrow(p, q, c=LINE, lw=0.9, cs="arc3", ms=6, z=4):
    ax.add_patch(FancyArrowPatch(p, q, arrowstyle="-|>", mutation_scale=ms, color=c, lw=lw, connectionstyle=cs,
                                 shrinkA=0, shrinkB=0, zorder=z))


def panel(x, y, w, h, n, title, sub=None):
    rbox(x, y, w, h, fc=PANEL, ec="none", r=2.0, z=0)
    ax.add_patch(Circle((x + 3.6, y + h - 3.6), 2.3, fc=DARK, ec="none", zorder=3))
    txt(x + 3.6, y + h - 3.65, str(n), fs=6.6, c="white", w="bold")
    txt(x + 7.2, y + h - 3.6, title, fs=7.0, w="bold", ha="left")
    if sub:
        txt(x + 7.2, y + h - 7.4, sub, fs=5.6, c=MUTED, ha="left", st="italic")


def lock(x, y, s=1.0, c=MUTED):
    ax.add_patch(Rectangle((x - 1.2 * s, y - 1.0 * s), 2.4 * s, 1.9 * s, fc=c, ec="none", zorder=8))
    ax.add_patch(Arc((x, y + 0.85 * s), 1.6 * s, 2.0 * s, theta1=0, theta2=180, color=c, lw=0.9, zorder=8))


def child(cx, by, h, phase=1.0, c=INK, lw=0.7, z=6):
    hr = 0.11 * h
    ax.add_patch(Circle((cx, by + h - hr), hr, fc="white", ec=c, lw=lw, zorder=z))
    sh, hip = by + 0.68 * h, by + 0.38 * h
    ax.plot([cx, cx], [by + h - 2 * hr, hip], color=c, lw=lw, zorder=z)
    a = 0.28 * h
    for sgn in (-1, 1):
        ax.plot([cx, cx + sgn * 0.6 * a, cx + sgn * a], [sh, sh + phase * a * 0.5, sh + phase * a * 1.1],
                color=c, lw=lw, zorder=z, solid_capstyle="round")
        ax.plot([cx, cx + sgn * 0.22 * h], [hip, by], color=c, lw=lw, zorder=z)


def view_chip(x, y, view, w=6.4, h=4.4):
    """tiny thumbnail: full frame (small child in a room) or child crop (large child)"""
    ec = LINE if view == "full" else AQUA
    rbox(x, y - h / 2, w, h, fc="white", ec=ec, lw=0.8, r=0.5, z=5)
    if view == "full":
        ax.add_patch(Rectangle((x + 0.3, y - h / 2 + 0.3), w - 0.6, 1.3, fc="#efeee9", ec="none", zorder=5))
        child(x + 4.2, y - h / 2 + 0.6, 2.6, lw=0.45)
    else:
        child(x + w / 2, y - h / 2 + 0.4, 3.6, lw=0.55)


# ============================================================ (1) preprocessing
panel(1, 7, 37.5, 84, 1, "Pre-processing", "per clip, before any model")
# film strip
fx, fy = 4.5, 66.5
ax.add_patch(Rectangle((fx, fy), 30, 12.4, fc="#2e2e2e", ec="none", zorder=1))
for k in range(10):
    for yy in (fy + 0.7, fy + 10.7):
        ax.add_patch(Rectangle((fx + 0.9 + k * 3.0, yy), 1.4, 1.0, fc="white", ec="none", zorder=2))
for k, ph in enumerate([-0.6, 0.4, 1.1, 0.2]):
    x = fx + 0.8 + k * 7.3
    ax.add_patch(Rectangle((x, fy + 2.2), 6.6, 8.0, fc="white", ec="none", zorder=2))
    child(x + 3.3, fy + 3.1, 6.0, ph, lw=0.6)
    ax.add_patch(Rectangle((x + 1.0, fy + 2.6), 4.6, 7.2, fc="none", ec=AQUA, lw=0.8, ls=(0, (2, 1)), zorder=6))
txt(19.5, 63.6, "trimmed behaviour clip (16 fps)", fs=5.6, c=MUTED)
arrow((19.5, 61.6), (19.5, 58.4))
rbox(3.8, 49.5, 31.4, 8.6, fc="white", ec=AQUA, lw=0.9)
txt(19.5, 55.6, "YOLO11x-pose", fs=6.2, w="bold")
txt(19.5, 52.0, r"child box $\mathbf{b}_t$ tracked per frame", fs=5.2, c=MUTED)
arrow((19.5, 49.2), (19.5, 46.0))
txt(19.5, 43.6, "two views of every window", fs=6.0, w="bold")
# full frame + child crop
rbox(4.5, 23.5, 13.5, 16.5, fc="white", ec=LINE, r=0.6, z=2)
ax.add_patch(Rectangle((5.0, 24.0), 12.5, 5.0, fc="#efeee9", ec="none", zorder=2))
child(12.5, 26.0, 7.0)
ax.add_patch(Rectangle((9.6, 25.4), 5.8, 8.4, fc="none", ec=AQUA, lw=0.8, ls=(0, (2, 1)), zorder=6))
rbox(21.0, 23.5, 13.5, 16.5, fc="white", ec=AQUA, lw=1.1, r=0.6, z=2)
child(27.75, 24.8, 13.6, lw=0.9)
for (x1, y1), (x2, y2) in (((15.4, 33.8), (21.0, 40.0)), ((15.4, 25.4), (21.0, 23.5))):
    ax.plot([x1, x2], [y1, y2], color=AQUA, lw=0.5, ls=(0, (2, 1.5)), zorder=4)
txt(11.25, 20.6, "full frame", fs=5.6, w="bold")
txt(27.75, 20.6, "child crop", fs=5.6, w="bold", c=AQUA)
txt(19.5, 15.8, r"crop: square around $\cup\,\mathbf{b}_t$ × 1.25", fs=5.4, c=MUTED)
txt(19.5, 12.4, "each view also horizontally flipped", fs=5.4, c=MUTED, st="italic")

# ============================================================ (2) frozen feature extraction
panel(40, 7, 62.5, 84, 2, "Frozen feature extraction: 8 streams", "no weights are updated in this stage")
# temporal sampling legend
txt(42.5, 80.0, "each stream: 16-frame windows, 0.5 s hop", fs=5.6, c=MUTED, ha="left")
for i, (lab, span, sec, c) in enumerate([("4 fps", 22.0, "4 s", "(slow, e.g. spinning)"),
                                         ("8 fps", 11.0, "2 s", ""), ("16 fps", 5.5, "1 s", "(fast, e.g. flapping)")]):
    y = 76.4 - i * 3.5
    txt(42.5, y, lab, fs=5.6, w="bold", ha="left")
    x0 = 52.0
    for k in range(16):
        xi = x0 + k * span / 15
        ax.plot([xi, xi], [y - 1.1, y + 1.1], color=INK, lw=0.5, zorder=3)
    txt(x0 + span + 1.2, y, f"= {sec}" + (f"  {c}" if c else ""), fs=5.0, c=MUTED, ha="left")

# column headers
HY = 64.4
txt(46.4, HY, "view", fs=5.4, c=MUTED)
txt(55.2, HY, "rate", fs=5.4, c=MUTED)
txt(72.0, HY, "frozen backbone", fs=5.4, c=MUTED)
txt(92.8, HY, "descriptor", fs=5.4, c=MUTED)
ax.plot([42, 102], [HY - 1.8, HY - 1.8], color=GRID, lw=0.5, zorder=2)

ROWS = [("full", "4 fps", "vm"), ("full", "8 fps", "vm"), ("full", "16 fps", "vm"),
        ("child", "4 fps", "vm"), ("child", "8 fps", "vm"), ("child", "16 fps", "vm"),
        ("full", "8 fps", "vj"), ("child", "8 fps", "vj")]
YS = [58.6, 53.6, 48.6, 43.6, 38.6, 33.6, 23.8, 18.8]
for (view, rate, bb), y in zip(ROWS, YS):
    c = BLUE if bb == "vm" else VIO
    view_chip(43.2, y, view)
    txt(55.2, y, rate, fs=5.6, w="bold")
    arrow((59.4, y), (62.4, y), c=c, lw=0.8, ms=5)
    arrow((81.6, y), (85.4, y), c=c, lw=0.8, ms=5)
    for k in range(6):
        ax.add_patch(Rectangle((86.0 + k * 1.75, y - 1.3), 1.75, 2.6, fc=c, ec="white", lw=0.35,
                               alpha=0.35 + 0.12 * (k % 3), zorder=5))
# backbone boxes (one shared model per family)
rbox(62.6, 30.6, 19.0, 31.0, fc=BLUE_L, ec=BLUE, lw=1.2, r=1.2, z=3)
lock(79.2, 59.0)
txt(72.1, 51.0, "VideoMAE-L", fs=7.0, w="bold")
txt(72.1, 47.4, "ViT-L/16", fs=5.4, c=MUTED)
txt(72.1, 44.8, "Kinetics-400", fs=5.4, c=MUTED)
txt(72.1, 40.8, "appearance\n+ action", fs=5.4, c=BLUE)
txt(72.1, 34.4, "6 streams,\nshared weights", fs=5.0, c=MUTED, st="italic")
rbox(62.6, 15.4, 19.0, 12.0, fc=VIO_L, ec=VIO, lw=1.2, r=1.2, z=3)
lock(79.4, 25.2)
txt(70.8, 23.6, "V-JEPA 2-L", fs=6.6, w="bold")
txt(72.1, 20.0, "ViT-L · SSv2", fs=5.4, c=MUTED)
txt(72.1, 17.2, "motion dynamics", fs=5.0, c=VIO)
txt(93.0, 12.6, "1024-d per stream:\nmean over all windows\nand both flips", fs=5.0, c=MUTED)

# ============================================================ (3) fusion + classification
panel(104, 7, 39.5, 84, 3, "Fusion & classification", "the only trained part")
# concatenated vector: one block per stream, aligned with the rows
for (view, rate, bb), y in zip(ROWS, YS):
    c = BLUE if bb == "vm" else VIO
    ax.add_patch(Rectangle((106.6, y - 2.2), 5.2, 4.4, fc=c, ec="white", lw=0.6,
                           alpha=0.85 if view == "full" else 0.55, zorder=5))
    ax.plot([97.2, 106.4], [y, y], color=c, lw=0.6, ls=(0, (1.5, 1.2)), zorder=3)
txt(109.2, HY, "concat", fs=5.4, c=MUTED)
txt(109.2, 13.4, r"$\mathbf{z}\in\mathbb{R}^{8192}$", fs=6.0)
txt(109.2, 10.2, "(8 × 1024)", fs=5.0, c=MUTED)
# classifier
rbox(114.2, 45.4, 28.6, 15.6, fc=ORANGE_L, ec=ORANGE, lw=1.2, r=1.2, z=3)
txt(128.4, 58.0, "standardisation", fs=5.3, c=MUTED)
txt(128.4, 54.6, "L2 logistic regression", fs=5.2, w="bold")
txt(128.4, 51.2, "≈33k trainable parameters", fs=4.7, c=ORANGE, w="bold")
txt(128.4, 48.0, "C = 0.05, class-balanced", fs=4.6, c=MUTED)
arrow((112.2, 40.0), (114.4, 49.0), c=LINE, lw=0.9, cs="arc3,rad=-0.25")
arrow((128.4, 45.0), (128.4, 42.0), c=LINE, lw=0.8, ms=5)
txt(128.4, 39.6, r"clip probabilities $p(y\,|\,\mathbf{z})$", fs=5.1)
for i, (lab, p) in enumerate(zip(["Arm flapping", "Hand action*", "Head banging", "Spinning"],
                                 [0.84, 0.08, 0.05, 0.03])):
    y = 35.4 - i * 3.8
    txt(129.0, y, lab, fs=5.0, ha="right", w="bold" if i == 0 else "normal")
    ax.add_patch(Rectangle((129.8, y - 1.15), 11.4, 2.3, fc="#ebeae5", ec="none", zorder=3))
    ax.add_patch(Rectangle((129.8, y - 1.15), 11.4 * p, 2.3, fc=ORANGE if i == 0 else "#c9c8c2", ec="none", zorder=4))
txt(128.4, 18.4, "video decision:\nmean of its clip\nprobabilities", fs=5.0, c=MUTED, st="italic")
txt(141.6, 9.8, "* ESBD only", fs=4.8, c=MUTED, ha="right")

# stage arrows between panels
arrow((38.4, 31.8), (42.4, 31.8), c=LINE, lw=1.3, ms=8)
txt(40.4, 29.4, "8 inputs", fs=4.6, c=MUTED)

# ============================================================ legend
lx, ly = 3, 1.8
for kind, lab in [("lock", "frozen (not updated)"), ("train", "trained"), ("vm", "VideoMAE-L stream"),
                  ("vj", "V-JEPA 2-L stream"), ("box", "child box / crop"),
                  ("dark", "darker = full frame, lighter = child crop")]:
    if kind == "lock":
        lock(lx + 1.4, ly - 0.1, 0.75)
    elif kind == "box":
        ax.add_patch(Rectangle((lx, ly - 1.2), 2.8, 2.4, fc="none", ec=AQUA, lw=0.8, ls=(0, (2, 1)), zorder=6))
    elif kind == "dark":
        ax.add_patch(Rectangle((lx, ly - 1.2), 1.4, 2.4, fc=BLUE, alpha=0.85, ec="none", zorder=6))
        ax.add_patch(Rectangle((lx + 1.4, ly - 1.2), 1.4, 2.4, fc=BLUE, alpha=0.55, ec="none", zorder=6))
    else:
        fc, ec = {"train": (ORANGE_L, ORANGE), "vm": (BLUE_L, BLUE), "vj": (VIO_L, VIO)}[kind]
        ax.add_patch(Rectangle((lx, ly - 1.2), 2.8, 2.4, fc=fc, ec=ec, lw=0.8, zorder=6))
    txt(lx + 3.8, ly, lab, fs=5.2, c=MUTED, ha="left")
    lx += 6.0 + len(lab) * 0.98

out = RESULTS / "figures"
fig.savefig(out / "fig_architecture_v2.png", dpi=400, bbox_inches="tight", pad_inches=0.02)
fig.savefig(out / "fig_architecture_v2.pdf", bbox_inches="tight", pad_inches=0.02)
print("ok")
