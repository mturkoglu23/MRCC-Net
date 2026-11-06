"""Figure 1 - MRCC-Net overview, drawn at print size (7.2 in wide) in Turkish and English.

results/figures/fig_architecture_{tr,en}.{png,pdf}; the Turkish one is also saved as
fig_architecture.png for the manuscript.
"""
import shutil

import matplotlib.pyplot as plt
import numpy as np
from matplotlib.patches import Arc, Circle, FancyArrowPatch, FancyBboxPatch, Polygon, Rectangle

from common import RESULTS

BLUE, BLUE_L = "#2a78d6", "#dbe9fb"
VIO, VIO_L = "#6250d6", "#e6e2fa"
AQUA = "#139e6d"
ORANGE, ORANGE_L = "#eb6834", "#fde3d8"
INK, MUTED, LINE, PANEL, DARK = "#0b0b0b", "#52514e", "#8a8984", "#f5f4f0", "#1c2b4a"

T = {
    "tr": dict(
        s1="Girdi ve çocuk takibi", s2="İki görünüm", s3="Çok hızlı pencereleme",
        s4="Dondurulmuş temel modeller (2 omurga, 8 akış)", s5="Birleştirme",
        s6="Sınıflandırma",
        video=r"girdi videosu  $V=\{I_t\}$, 16 fps", yolo="YOLO11x-pose", yolo2=r"çocuk kutusu $\mathbf{b}_t$",
        track="en büyük → en yakın kutu ile izleme",
        full="tüm kare", child="çocuk merkezli", crop1="pencere başına sabit kare:",
        crop2=r"$\cup\,\mathbf{b}_t$ × 1,25", flip="+ yatay ayna  $f\in\{0,1\}$",
        win="16 kare / pencere · 0,5 s kaydırma", slow="yavaş döngü\n(dönme, sallanma)",
        fast="hızlı döngü\n(el çırpma)",
        rows=["4 fps", "8 fps", "16 fps"], cols=["tüm", "çocuk"],
        vm2="ViT-L/16 · Kinetics-400", vm3="görünüm + eylem",
        vj2="ViT-L · SSv2", vj3="hareket dinamiği",
        shared="aynı model, 6 girdi (paylaşılan ağırlık)", shared2="aynı model, 2 girdi",
        emb=r"$\mathbf{e}_{s,f,w}\in\mathbb{R}^{1024}$",
        pool="pencere + ayna\nortalaması", concat=r"$\mathbf{z}\in\mathbb{R}^{8192}$",
        clf1="standartlaştırma", clf2="L2 lojistik regresyon", clf3="eğitilen tek bileşen",
        clf4="(≈33 bin parametre)",
        classes=["El çırpma", "El hareketi*", "Kafa vurma", "Dönme"], prob=r"$p(y\,|\,\mathbf{z})$",
        video_dec="video kararı: klip\nolasılıklarının ortalaması", esbd="* yalnız ESBD",
        lg_frozen="dondurulmuş (güncellenmez)", lg_train="eğitilen", lg_vm="VideoMAE-L akışı",
        lg_vj="V-JEPA 2-L akışı", lg_box="çocuk kutusu / kırpma"),
    "en": dict(
        s1="Input and child tracking", s2="Two views", s3="Multi-rate windowing",
        s4="Frozen backbones (2 models, 8 streams)", s5="Fusion",
        s6="Classification",
        video=r"input video  $V=\{I_t\}$, 16 fps", yolo="YOLO11x-pose", yolo2=r"child box $\mathbf{b}_t$",
        track="largest → nearest-box tracking",
        full="full frame", child="child-centric", crop1="fixed square per window:",
        crop2=r"$\cup\,\mathbf{b}_t$ × 1.25", flip="+ horizontal flip  $f\in\{0,1\}$",
        win="16 frames / window · 0.5 s hop", slow="slow cycles\n(spinning)",
        fast="fast cycles\n(flapping)",
        rows=["4 fps", "8 fps", "16 fps"], cols=["full", "child"],
        vm2="ViT-L/16 · Kinetics-400", vm3="appearance + action",
        vj2="ViT-L · SSv2", vj3="motion dynamics",
        shared="one model, 6 inputs (shared weights)", shared2="one model, 2 inputs",
        emb=r"$\mathbf{e}_{s,f,w}\in\mathbb{R}^{1024}$",
        pool="window + flip\naveraging", concat=r"$\mathbf{z}\in\mathbb{R}^{8192}$",
        clf1="standardisation", clf2="L2 logistic regression", clf3="only trained part",
        clf4="(≈33k parameters)",
        classes=["Arm flapping", "Hand action*", "Head banging", "Spinning"], prob=r"$p(y\,|\,\mathbf{z})$",
        video_dec="video decision: mean of\nclip probabilities", esbd="* ESBD only",
        lg_frozen="frozen (not updated)", lg_train="trained", lg_vm="VideoMAE-L stream",
        lg_vj="V-JEPA 2-L stream", lg_box="child box / crop"),
}

FS = 6.6
plt.rcParams.update({"font.family": "DejaVu Sans", "font.size": FS, "mathtext.fontset": "dejavusans"})


def draw(lang):
    S = T[lang]
    fig, ax = plt.subplots(figsize=(7.2, 6.6))
    ax.set_xlim(0, 144)
    ax.set_ylim(-9, 126)
    ax.set_aspect("equal")
    ax.axis("off")

    def rbox(x, y, w, h, fc="white", ec=LINE, lw=0.8, r=1.6, ls="-", z=1):
        ax.add_patch(FancyBboxPatch((x, y), w, h, boxstyle=f"round,pad=0,rounding_size={r}", fc=fc, ec=ec,
                                    lw=lw, ls=ls, zorder=z))

    def txt(x, y, s, fs=FS, c=INK, ha="center", va="center", w="normal", st="normal", z=5):
        ax.text(x, y, s, fontsize=fs, color=c, ha=ha, va=va, weight=w, style=st, zorder=z)

    def arrow(p, q, c=LINE, lw=0.9, cs="arc3", ms=7, z=4):
        ax.add_patch(FancyArrowPatch(p, q, arrowstyle="-|>", mutation_scale=ms, color=c, lw=lw,
                                     connectionstyle=cs, shrinkA=0, shrinkB=0, zorder=z))

    def panel(x, y, w, h, n, title):
        rbox(x, y, w, h, fc=PANEL, ec="none", r=2.2, z=0)
        ax.add_patch(Circle((x + 4, y + h - 4), 2.6, fc=DARK, ec="none", zorder=3))
        txt(x + 4, y + h - 4.05, str(n), fs=7, c="white", w="bold")
        txt(x + 8, y + h - 4, title, fs=6.9, w="bold", ha="left")

    def lock(x, y, s=1.0, c=MUTED):
        ax.add_patch(Rectangle((x - 1.3 * s, y - 1.1 * s), 2.6 * s, 2.0 * s, fc=c, ec="none", zorder=6))
        ax.add_patch(Arc((x, y + 0.9 * s), 1.8 * s, 2.2 * s, theta1=0, theta2=180, color=c, lw=1.0, zorder=6))

    def child(cx, by, h, phase=0.0, c=INK, lw=0.8):
        """stick figure, feet at by, height h; phase moves the arms"""
        hr = 0.11 * h
        ax.add_patch(Circle((cx, by + h - hr), hr, fc="white", ec=c, lw=lw, zorder=5))
        sh, hip = by + 0.68 * h, by + 0.38 * h
        ax.plot([cx, cx], [by + h - 2 * hr, hip], color=c, lw=lw, zorder=5)
        a = 0.28 * h
        for sgn in (-1, 1):
            ax.plot([cx, cx + sgn * 0.6 * a, cx + sgn * a], [sh, sh + phase * a * 0.5, sh + phase * a * 1.1],
                    color=c, lw=lw, zorder=5, solid_capstyle="round")
            ax.plot([cx, cx + sgn * 0.22 * h], [hip, by], color=c, lw=lw, zorder=5)

    def vec(x, y, c, n=5, w=1.6, h=1.4):
        for i in range(n):
            ax.add_patch(Rectangle((x, y + i * h), w, h, fc=c, ec="white", lw=0.4, alpha=0.35 + 0.13 * (i % 3),
                                   zorder=5))

    # ======================================================== row 1
    Y0, H = 66, 58
    # (1) input + tracking ------------------------------------------------
    panel(2, Y0, 46, H, 1, S["s1"])
    ax.add_patch(Rectangle((4.5, 92), 41, 17, fc="#2e2e2e", ec="none", zorder=1))
    for k in range(12):
        for yy in (93.0, 106.6):
            ax.add_patch(Rectangle((5.6 + k * 3.4, yy), 1.6, 1.2, fc="white", ec="none", zorder=2))
    phases = [-0.6, 0.4, 1.1, 0.2]
    for k, ph in enumerate(phases):
        x = 5.4 + k * 10.0
        ax.add_patch(Rectangle((x, 95.2), 9.2, 10.6, fc="white", ec="none", zorder=2))
        child(x + 4.6, 96.6, 7.6, ph)
        ax.add_patch(Rectangle((x + 1.6, 96.0), 6.0, 9.2, fc="none", ec=AQUA, lw=0.9, ls=(0, (2, 1)), zorder=6))
    txt(25, 88.6, S["video"], fs=6.4, c=MUTED)
    arrow((25, 86.2), (25, 82.4))
    rbox(9, 72, 32, 10, fc="white", ec=AQUA, lw=0.9)
    txt(25, 78.6, S["yolo"], w="bold")
    txt(25, 74.8, S["yolo2"], fs=6.3, c=MUTED)
    txt(25, 69.0, S["track"], fs=6.0, c=MUTED, st="italic")

    # (2) two views -------------------------------------------------------
    panel(50, Y0, 46, H, 2, S["s2"])
    rbox(53, 86, 18, 20, fc="white", ec=LINE, r=0.6)                   # full frame
    ax.add_patch(Rectangle((53.6, 86.6), 16.8, 7.0, fc="#efeee9", ec="none", zorder=1))   # floor
    child(64.5, 89.0, 8.4, 1.0)
    ax.add_patch(Rectangle((61.0, 88.3), 7.0, 10.2, fc="none", ec=AQUA, lw=0.9, ls=(0, (2, 1)), zorder=6))
    rbox(76, 86, 18, 20, fc="white", ec=AQUA, lw=1.2, r=0.6)           # child crop
    child(85, 88.0, 16.4, 1.0, lw=1.1)
    for (x1, y1), (x2, y2) in (((68.0, 98.5), (76, 106)), ((68.0, 88.3), (76, 86))):
        ax.plot([x1, x2], [y1, y2], color=AQUA, lw=0.6, ls=(0, (2, 1.5)), zorder=4)
    txt(62, 83.0, S["full"], w="bold")
    txt(62, 79.6, r"$V^{\mathrm{full}}$", fs=6.4, c=MUTED)
    txt(85, 83.0, S["child"], w="bold", c=AQUA)
    txt(85, 79.6, r"$V^{\mathrm{child}}$", fs=6.4, c=MUTED)
    txt(73, 74.6, S["crop1"] + " " + S["crop2"], fs=6.0, c=MUTED)
    txt(73, 70.2, S["flip"], fs=6.0, c=MUTED, st="italic")
    arrow((46.6, 100), (52.4, 100), lw=1.1)

    # (3) multi-rate windows ----------------------------------------------
    panel(98, Y0, 44, H, 3, S["s3"])
    txt(120, 113.2, S["win"], fs=6.0, c=MUTED)
    spans = [(4, "4 s", 104.5, 28.0), (8, "2 s", 94.0, 14.0), (16, "1 s", 83.5, 7.0)]
    for (r, dur, y, wd), lab in zip(spans, S["rows"]):
        x0 = 111.0
        ax.plot([x0 - 0.5, 140.5], [y, y], color="#c9c8c2", lw=0.6, zorder=2)
        for i in range(16):
            xi = x0 + i * wd / 15
            ax.plot([xi, xi], [y - 1.4, y + 1.4], color=BLUE, lw=0.8, zorder=3)
        ax.annotate("", xy=(x0, y - 3.0), xytext=(x0 + wd, y - 3.0),
                    arrowprops=dict(arrowstyle="<->", lw=0.6, color=MUTED, shrinkA=0, shrinkB=0))
        txt(x0 + wd / 2, y - 5.0, dur, fs=5.8, c=MUTED)
        txt(100.5, y, lab, fs=6.2, w="bold", ha="left")
    ax.annotate("", xy=(104, 70.5), xytext=(139, 70.5),
                arrowprops=dict(arrowstyle="<->", lw=0.7, color=LINE, shrinkA=0, shrinkB=0))
    txt(105, 74.0, S["slow"], fs=5.6, c=MUTED, ha="left")
    txt(139, 74.0, S["fast"], fs=5.6, c=MUTED, ha="right")
    arrow((94.6, 96), (99.6, 96), lw=1.1)

    # row connector: (3) -> (4)
    ax.add_patch(FancyArrowPatch((120, 66.2), (34, 62.4), arrowstyle="-|>", mutation_scale=8, color=LINE, lw=1.1,
                                 connectionstyle="angle,angleA=-90,angleB=180,rad=3", shrinkA=0, shrinkB=0,
                                 zorder=4))

    # ======================================================== row 2
    Y1, H1 = 2, 60
    # (4) frozen encoders ---------------------------------------------------
    panel(2, Y1, 78, H1, 4, S["s4"])
    def grid(y_rows, labels, fc, ec_full, y_top):
        """input chips (rate x view) grouped in one tray, single arrow out"""
        y_lo, y_hi = min(y_rows) - 3.2, max(y_rows) + 3.2
        rbox(6.2, y_lo, 16.0, y_hi - y_lo, fc="white", ec="#d6d5cf", lw=0.6, r=1.0, z=2)
        for lab, y in zip(labels, y_rows):
            txt(4.6, y, lab, fs=5.8, c=MUTED, ha="right")
            rbox(7.6, y - 2.0, 6.0, 4.0, fc=fc, ec=ec_full, lw=0.8, r=0.6, z=3)
            rbox(14.8, y - 2.0, 6.0, 4.0, fc=fc, ec=AQUA, lw=0.8, r=0.6, z=3)
        return (y_lo + y_hi) / 2

    txt(10.6, 54.4, S["cols"][0], fs=5.6, c=MUTED)
    txt(17.8, 54.4, S["cols"][1], fs=5.6, c=AQUA)
    ym = grid([48.0, 42.0, 36.0], ["4", "8", "16"], BLUE_L, BLUE, 0)
    arrow((22.4, ym), (25.6, ym), c=BLUE, lw=1.1)
    rbox(26, 31, 28, 22, fc=BLUE_L, ec=BLUE, lw=1.3, r=1.4, z=3)
    lock(51.0, 49.6)
    txt(40, 46.6, "VideoMAE-L", fs=7.4, w="bold")
    txt(40, 42.2, S["vm2"], fs=5.5, c=MUTED)
    txt(40, 37.4, S["vm3"], fs=5.5, c=BLUE)
    txt(40, 27.8, S["shared"], fs=5.5, c=MUTED, st="italic")
    for i in range(6):
        vec(57.6 + i * 3.4, 35.0, BLUE)
    txt(66.6, 31.0, S["emb"], fs=5.9, c=MUTED)
    arrow((54.2, 42), (57.2, 42), c=BLUE, lw=0.9)
    # V-JEPA
    ym = grid([14.5], ["8"], VIO_L, VIO, 0)
    arrow((22.4, ym), (25.6, ym), c=VIO, lw=1.1)
    rbox(26, 6, 28, 17, fc=VIO_L, ec=VIO, lw=1.3, r=1.4, z=3)
    lock(51.0, 19.8)
    txt(40, 18.4, "V-JEPA 2-L", fs=7.4, w="bold")
    txt(40, 14.0, S["vj2"], fs=5.5, c=MUTED)
    txt(40, 9.6, S["vj3"], fs=5.5, c=VIO)
    txt(40, 3.4, S["shared2"], fs=5.5, c=MUTED, st="italic")
    for i in range(2):
        vec(57.6 + i * 3.4, 10.0, VIO)
    arrow((54.2, 14.5), (57.2, 14.5), c=VIO, lw=0.9)

    # (5) pooling + concatenation ------------------------------------------
    panel(82, Y1, 26, H1, 5, S["s5"])
    txt(95, 50.0, S["pool"], fs=6.0, c=MUTED)
    txt(95, 43.6, r"$\mathbf{z}_s=\frac{1}{2W_s}\sum_{f}\sum_{w}\mathbf{e}_{s,f,w}$", fs=6.8)
    cols8 = [VIO] * 2 + [BLUE] * 6
    for i, c in enumerate(cols8):
        ax.add_patch(Rectangle((91.0, 9.0 + i * 3.6), 8.0, 3.3, fc=c, ec="white", lw=0.5,
                               alpha=0.9 if i % 2 == 0 else 0.6, zorder=4))
    txt(95, 6.0, S["concat"], fs=6.6)
    arrow((78.6, 40.0), (90.6, 30.0), c=BLUE, lw=0.9, cs="arc3,rad=-0.1")
    arrow((64.6, 14.5), (90.6, 12.5), c=VIO, lw=0.9, cs="arc3,rad=0")

    # (6) classifier ---------------------------------------------------------
    panel(110, Y1, 32, H1, 6, S["s6"])
    rbox(113, 37, 26, 15, fc=ORANGE_L, ec=ORANGE, lw=1.3, r=1.2, z=3)
    txt(126, 48.8, S["clf1"], fs=6.0, c=MUTED)
    txt(126, 45.2, S["clf2"], fs=5.6, w="bold")
    txt(126, 41.6, S["clf3"], fs=5.8, c=ORANGE, w="bold")
    txt(126, 38.9, S["clf4"], fs=5.4, c=MUTED)
    arrow((99.6, 26), (112.6, 42), c=LINE, lw=1.0, cs="arc3,rad=-0.15")
    txt(126, 33.0, S["prob"], fs=6.4)
    probs = [0.84, 0.08, 0.05, 0.03]
    for i, (lab, p) in enumerate(zip(S["classes"], probs)):
        y = 27.6 - i * 4.4
        txt(124.8, y, lab, fs=5.8, ha="right", w="bold" if i == 0 else "normal")
        ax.add_patch(Rectangle((125.6, y - 1.3), 14.0, 2.6, fc="#ebeae5", ec="none", zorder=3))
        ax.add_patch(Rectangle((125.6, y - 1.3), 14.0 * p, 2.6, fc=ORANGE if i == 0 else "#c9c8c2", ec="none",
                               zorder=4))
    arrow((126, 36.6), (126, 34.8), c=LINE, lw=0.8, ms=5)
    txt(126, 8.4, S["video_dec"], fs=5.6, c=MUTED, st="italic")
    txt(140.5, 3.6, S["esbd"], fs=5.2, c=MUTED, ha="right")

    # ======================================================== legend
    lx, ly = 4, -5.2
    items = [("lock", S["lg_frozen"]), ("train", S["lg_train"]), ("vm", S["lg_vm"]), ("vj", S["lg_vj"]),
             ("box", S["lg_box"])]
    for kind, lab in items:
        if kind == "lock":
            lock(lx + 1.6, ly - 0.2, 0.8)
        elif kind == "box":
            ax.add_patch(Rectangle((lx, ly - 1.4), 3.2, 2.8, fc="none", ec=AQUA, lw=0.9, ls=(0, (2, 1)), zorder=6))
        else:
            fc, ec = {"train": (ORANGE_L, ORANGE), "vm": (BLUE_L, BLUE), "vj": (VIO_L, VIO)}[kind]
            ax.add_patch(Rectangle((lx, ly - 1.4), 3.2, 2.8, fc=fc, ec=ec, lw=0.9, zorder=6))
        txt(lx + 4.4, ly, lab, fs=5.6, c=MUTED, ha="left")
        lx += 7.0 + len(lab) * 1.2

    out = RESULTS / "figures"
    fig.savefig(out / f"fig_architecture_{lang}.png", dpi=400, bbox_inches="tight", pad_inches=0.02)
    fig.savefig(out / f"fig_architecture_{lang}.pdf", bbox_inches="tight", pad_inches=0.02)
    plt.close(fig)


for lang in T:
    draw(lang)
shutil.copy(RESULTS / "figures" / "fig_architecture_tr.png", RESULTS / "figures" / "fig_architecture.png")
print("ok")
