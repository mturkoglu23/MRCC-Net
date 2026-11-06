"""Cluster-level paired inference for MRCC-Net vs. comparators under P1.

The unit of resampling is the source group (all its clips and all three CV repeats stay
together), so neither the clips of one video nor the three repeats of one clip are
treated as independent observations.
  effect  : Delta = mean over repeats of [acc(MRCC-Net) - acc(other)]
            = sum_g D_g / N,  D_g = sum_{i in g} mean_r (c^A_ri - c^B_ri)
  CI      : percentile cluster bootstrap over groups (B = 10000)
  p-value : paired cluster sign-flip permutation test on D_g (B = 20000), two-sided
  family  : the 8 pre-specified primary comparisons (MRCC-Net vs. the 4 single-backbone
            probes on both datasets) are Holm-adjusted; ablations and fine-tuned models
            are reported as exploratory (unadjusted).
Output: results/final/significance_cluster.csv
"""
import re

import numpy as np
import pandas as pd

from common import RESULTS
from final_experiments import ABLATION, BASELINES

OUT = RESULTS / "final"
safe = lambda s: re.sub(r"[^A-Za-z0-9]+", "_", s).strip("_")


def load(ds, name):
    z = np.load(OUT / f"pred_{ds}_grouped_{safe(name)}.npz")
    return z["p"], z["y"], z["groups"]


def cluster_test(pa, pb, y, g, nb=10000, nperm=20000, seed=0):
    ca = (pa.argmax(-1) == y).mean(0)           # per-clip correctness averaged over repeats
    cb = (pb.argmax(-1) == y).mean(0)
    ug, inv = np.unique(g, return_inverse=True)
    D = np.bincount(inv, ca - cb)                # per-group summed difference
    n = np.bincount(inv)
    delta = D.sum() / n.sum()
    rng = np.random.default_rng(seed)
    s = rng.integers(0, len(ug), (nb, len(ug)))
    boot = D[s].sum(1) / n[s].sum(1)
    flips = rng.choice([-1.0, 1.0], (nperm, len(ug)))
    tperm = np.abs(flips @ D)
    p = (1 + (tperm >= abs(D.sum()) - 1e-12).sum()) / (1 + nperm)
    return delta, *np.percentile(boot, [2.5, 97.5]), p, len(ug)


def holm(p):
    p = np.asarray(p)
    o = np.argsort(p)
    adj = np.maximum.accumulate((len(p) - np.arange(len(p))) * p[o]).clip(max=1)
    out = np.empty_like(adj)
    out[o] = adj
    return out


def main():
    rows = []
    for ds in ["esbd", "ssbd"]:
        pa, y, g = load(ds, "MRCC-Net (proposed)")
        comps = [(n, "primary", load(ds, n)[0]) for n in BASELINES]
        comps += [(n, "ablation", load(ds, n)[0]) for n in ABLATION if not n.startswith("MRCC")]
        for m in ["videomae-B", "videomae-L"]:
            f = OUT / f"pred_ft_{m}_{ds}_grouped.npz"
            if f.exists():
                z = np.load(f)
                assert (z["y"] == y).all()
                comps.append((f"{m} fine-tuned", "fine-tuned", z["p"]))
        for name, fam, pb in comps:
            d, lo, hi, p, k = cluster_test(pa, pb, y, g)
            rows.append(dict(dataset=ds, vs=name, family=fam, delta=d, ci_lo=lo, ci_hi=hi, p_perm=p, n_groups=k))
    df = pd.DataFrame(rows)
    prim = df.family == "primary"
    df["p_holm"] = np.nan
    df.loc[prim, "p_holm"] = holm(df.loc[prim, "p_perm"])
    df.to_csv(OUT / "significance_cluster.csv", index=False)
    print(df.to_string(float_format=lambda v: f"{v:.4f}"))


if __name__ == "__main__":
    main()
