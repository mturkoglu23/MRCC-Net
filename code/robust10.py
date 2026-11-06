"""Robustness check: P1 with 10 split repeats (seeds 0-9; seeds 0-2 identical to the main
protocol) for all frozen-feature models. Outputs results/final/robust10.csv and
robust10_significance.csv (cluster sign-flip test + Holm on the primary family)."""
import numpy as np
import pandas as pd
from sklearn.model_selection import StratifiedGroupKFold

from common import CLASSES, RESULTS, index_dataset
from eval_v2 import load_desc, run
from final_experiments import ABLATION, BASELINES
from stats_cluster import cluster_test, holm
from train_eval import metrics

OUT = RESULTS / "final"
SEEDS = range(10)


def main():
    rows, sig = [], []
    for ds in ["esbd", "ssbd"]:
        recs, F = load_desc(ds)
        y = np.array([CLASSES[ds].index(r["label"]) for r in recs])
        g = np.array([r["group"] for r in recs])
        folds = [[te for _, te in StratifiedGroupKFold(5, shuffle=True, random_state=s).split(y, y, g)] for s in SEEDS]
        specs = {**{k: ("concat", v) for k, v in BASELINES.items()},
                 **{k: (v if isinstance(v, tuple) else ("concat", v)) for k, v in ABLATION.items()}}
        P = {}
        for name, (method, streams) in specs.items():
            P[name] = np.stack([run(F, y, g, folds[s], method, streams)[0] for s in SEEDS])
            acc = [(P[name][s].argmax(1) == y).mean() for s in SEEDS]
            vid = [metrics(y, P[name][s], g)["video_acc"] for s in SEEDS]
            rows.append(dict(dataset=ds, model=name, acc_mean=np.mean(acc), acc_std=np.std(acc),
                             acc_min=np.min(acc), acc_max=np.max(acc), video_acc_mean=np.mean(vid)))
            print(ds, f"{name:45s} {np.mean(acc):.4f}±{np.std(acc):.4f}", flush=True)
        pa = P["MRCC-Net (proposed)"]
        for name in specs:
            if name.startswith("MRCC"):
                continue
            d, lo, hi, p, k = cluster_test(pa, P[name], y, g)
            sig.append(dict(dataset=ds, vs=name, family="primary" if name in BASELINES else "ablation",
                            delta=d, ci_lo=lo, ci_hi=hi, p_perm=p, n_groups=k))
    S = pd.DataFrame(sig)
    S["p_holm"] = np.nan
    prim = S.family == "primary"
    S.loc[prim, "p_holm"] = holm(S.loc[prim, "p_perm"])
    pd.DataFrame(rows).to_csv(OUT / "robust10.csv", index=False)
    S.to_csv(OUT / "robust10_significance.csv", index=False)
    print(S.to_string(float_format=lambda v: f"{v:.4f}"))


if __name__ == "__main__":
    main()
