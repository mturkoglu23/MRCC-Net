"""Video-cluster bootstrap CIs for the cross-dataset and external tests of MRCC-Net,
plus the per-source composition of the external set (results/final/external_ci.csv)."""
import numpy as np
import pandas as pd

from common import RESULTS

OUT = RESULTS / "final"


def boot(c, g, nb=10000, seed=0):
    ug, inv = np.unique(g, return_inverse=True)
    hit, n = np.bincount(inv, c), np.bincount(inv)
    s = np.random.default_rng(seed).integers(0, len(ug), (nb, len(ug)))
    return np.percentile(hit[s].sum(1) / n[s].sum(1), [2.5, 97.5])


def main():
    rows = []
    for tag, sc in [("ssbd_to_esbd", "SSBD->ESBD"), ("esbd_to_ssbd", "ESBD->SSBD"),
                    ("ssbd_esbd_to_external", "external")]:
        z = np.load(OUT / f"pred_cross_{tag}.npz", allow_pickle=True)
        c = (z["p"].argmax(1) == z["y"]).astype(float)
        g = z["groups"]
        lo, hi = boot(c, g)
        rows.append(dict(scenario=sc, subset="all", clips=len(c), videos=len(set(g)), acc=c.mean(), ci_lo=lo, ci_hi=hi))
        if sc == "external":
            src = np.array([s.split("@")[1] for s in g])
            for s in np.unique(src):
                k = src == s
                rows.append(dict(scenario=sc, subset=s, clips=int(k.sum()), videos=len(set(g[k])), acc=c[k].mean(),
                                 ci_lo=np.nan, ci_hi=np.nan))
    df = pd.DataFrame(rows)
    df.to_csv(OUT / "external_ci.csv", index=False)
    print(df.to_string())


if __name__ == "__main__":
    main()
