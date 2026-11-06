"""Paper figures from results/final (300 dpi PNG + PDF)."""
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from common import RESULTS

F = RESULTS / "final"
OUT = RESULTS / "figures"
BLUE, ORANGE, INK, MUTED = "#2a78d6", "#eb6834", "#0b0b0b", "#52514e"
plt.rcParams.update({"font.family": "DejaVu Sans", "font.size": 9, "axes.edgecolor": MUTED,
                     "axes.labelcolor": INK, "xtick.color": MUTED, "ytick.color": MUTED,
                     "axes.spines.top": False, "axes.spines.right": False})
SHORT = {"VideoMAE-L (K400)": "VideoMAE-L", "VideoMAE-H (K400)": "VideoMAE-H", "V-JEPA 2-L (SSv2)": "V-JEPA 2-L",
         "V-JEPA 2-g (SSL)": "V-JEPA 2-g", "MRCC-Net (proposed)": "MRCC-Net"}


def save(fig, name):
    fig.savefig(OUT / f"{name}.png", dpi=300, bbox_inches="tight")
    fig.savefig(OUT / f"{name}.pdf", bbox_inches="tight")
    plt.close(fig)


# Fig. 1 - leakage: grouped vs random accuracy per model
m = pd.read_csv(F / "main_results.csv")
fig, axes = plt.subplots(1, 2, figsize=(7.2, 2.8), sharey=True)
for ax, ds in zip(axes, ["esbd", "ssbd"]):
    d = m[m.dataset == ds]
    models = list(SHORT)
    x = np.arange(len(models))
    for off, proto, col, lab in [(-0.19, "grouped", BLUE, "Source-grouped 5-fold (P1)"),
                                 (0.19, "random", ORANGE, "Random clip 5-fold")]:
        v = [100 * d[(d.model == k) & (d.protocol == proto)].acc_mean.item() for k in models]
        e = [100 * d[(d.model == k) & (d.protocol == proto)].acc_std.item() for k in models]
        ax.bar(x + off, v, 0.36, yerr=e, color=col, label=lab, error_kw=dict(lw=0.8, ecolor=MUTED, capsize=2))
    for i, k in enumerate(models):
        g = 100 * d[(d.model == k) & (d.protocol == "grouped")].acc_mean.item()
        r = 100 * d[(d.model == k) & (d.protocol == "random")].acc_mean.item()
        ax.text(i, r + 1.2, f"+{r - g:.1f}", ha="center", fontsize=7, color=MUTED)
    ax.set_xticks(x, [SHORT[k] for k in models], rotation=25, ha="right")
    ax.set_ylim(75, 103)
    ax.set_title(ds.upper(), color=INK, fontsize=10)
    ax.yaxis.grid(True, color="#e6e5e0", lw=0.6)
    ax.set_axisbelow(True)
axes[0].set_ylabel("Accuracy (%)")
fig.legend(*axes[0].get_legend_handles_labels(), frameon=False, fontsize=8, loc="upper center",
           ncol=2, bbox_to_anchor=(0.5, 1.08))
save(fig, "fig_leakage")

# Fig. 2 - ablation (grouped protocol)
a = pd.read_csv(F / "ablation.csv")
a = a[a.protocol == "grouped"]
order = list(dict.fromkeys(a.model))
fig, axes = plt.subplots(1, 2, figsize=(7.2, 3.0), sharey=True)
for ax, ds in zip(axes, ["esbd", "ssbd"]):
    d = a[a.dataset == ds].set_index("model").loc[order]
    y = np.arange(len(order))[::-1]
    cols = [BLUE if k.startswith("MRCC") else "#86b6ef" for k in order]
    ax.barh(y, 100 * d.acc_mean, xerr=100 * d.acc_std, color=cols, height=0.62,
            error_kw=dict(lw=0.8, ecolor=MUTED, capsize=2))
    for yi, v, e in zip(y, 100 * d.acc_mean, 100 * d.acc_std):
        ax.text(v + e + 0.4, yi, f"{v:.1f}", va="center", fontsize=7, color=INK)
    ax.set_xlim(80, 97.5)
    ax.set_title(f"{ds.upper()} (source-grouped)", color=INK, fontsize=10)
    ax.xaxis.grid(True, color="#e6e5e0", lw=0.6)
    ax.set_axisbelow(True)
    ax.set_xlabel("Accuracy (%)")
axes[0].set_yticks(np.arange(len(order))[::-1], order, fontsize=7.5)
save(fig, "fig_ablation")

# Fig. 3 - confusion matrices of the proposed model (grouped) + cross-dataset
mats = [("confusion_esbd_grouped.csv", "ESBD, source-grouped"), ("confusion_ssbd_grouped.csv", "SSBD, source-grouped"),
        ("confusion_cross_ssbd_to_esbd.csv", "SSBD → ESBD"), ("confusion_cross_esbd_to_ssbd.csv", "ESBD → SSBD")]
fig, axes = plt.subplots(1, 4, figsize=(11, 2.9), gridspec_kw=dict(wspace=0.55))
for ax, (f, t) in zip(axes, mats):
    c = pd.read_csv(F / f, index_col=0)
    r = c.values / c.values.sum(1, keepdims=True)
    ax.imshow(r, cmap="Blues", vmin=0, vmax=1)
    for i in range(r.shape[0]):
        for j in range(r.shape[1]):
            ax.text(j, i, f"{100 * r[i, j]:.0f}", ha="center", va="center", fontsize=7.5,
                    color="white" if r[i, j] > 0.6 else INK)
    lab = [s.replace("Flapping", "Flap.").replace("Banging", "Bang.").replace("Action", "Act.") for s in c.columns]
    ax.set_xticks(range(len(lab)), lab, rotation=40, ha="right", fontsize=7)
    ax.set_yticks(range(len(lab)), lab, fontsize=7)
    ax.set_title(t, fontsize=9, color=INK)
    for s in ax.spines.values():
        s.set_visible(False)
axes[0].set_ylabel("True")
fig.supxlabel("Predicted (row-normalised %)", fontsize=8, color=MUTED, y=-0.08)
save(fig, "fig_confusion")
print("ok")

# Fig. 4 - sensitivity to the regularisation strength C (source-grouped protocol)
s = pd.read_csv(F / "sensitivity_C.csv")
s = s[s.protocol == "grouped"]
fig, axes = plt.subplots(1, 2, figsize=(7.2, 2.6), sharey=False)
for ax, ds in zip(axes, ["esbd", "ssbd"]):
    for model, col, lab in [("MRCC-Net (proposed)", BLUE, "MRCC-Net"), ("A1 VideoMAE-L baseline", ORANGE, "VideoMAE-L")]:
        d = s[(s.dataset == ds) & (s.model == model)].sort_values("C")
        mu, sd = 100 * d.acc_mean.values, 100 * d.acc_std.values
        ax.plot(d.C, mu, color=col, lw=2, marker="o", ms=4, label=lab)
        ax.fill_between(d.C, mu - sd, mu + sd, color=col, alpha=0.15, lw=0)
        ax.text(d.C.values[-1] * 1.25, mu[-1], lab, color=col, fontsize=7.5, va="center")
    ax.axvline(0.05, color=MUTED, lw=0.8, ls="--")
    ax.text(0.05, ax.get_ylim()[1], " C = 0.05 (fixed)", fontsize=6.8, color=MUTED, va="top")
    ax.set_xscale("log")
    ax.set_xlim(0.0007, 4)
    ax.set_title(f"{ds.upper()} (source-grouped)", color=INK, fontsize=10)
    ax.set_xlabel("C (inverse L2 regularisation)")
    ax.yaxis.grid(True, color="#e6e5e0", lw=0.6)
    ax.set_axisbelow(True)
axes[0].set_ylabel("Accuracy (%)")
fig.tight_layout()
save(fig, "fig_sensitivity")
print("ok2")
