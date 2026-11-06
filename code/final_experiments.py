"""Final, pre-registered experiment set for the paper (protocol in PROTOCOL.md).

Proposed model: MRCC-Net (Multi-Rate Child-Centric two-backbone fusion)
  streams = VideoMAE-L (K400) at 4/8/16 fps  x {full, child-crop}
          + V-JEPA 2-L (SSv2) at 8 fps        x {full, child-crop}
  clip descriptor per stream = mean over windows and over original/flipped video
  classifier = StandardScaler + L2 logistic regression (C=0.05, class-balanced) on
               the concatenation of the 8 stream descriptors.

Outputs (results/final/):
  main_results.csv        baselines + proposed, both protocols, both datasets
  ablation.csv            component ablation (grouped + random)
  per_class.csv           per-class P/R/F1 of the proposed model
  confusion_<ds>_<proto>.csv
  significance.csv        proposed vs. baselines (McNemar on pooled clips, video-cluster bootstrap CI)
  leakage.csv             random - grouped gap per model
"""
import json
import re

import numpy as np
import pandas as pd
from scipy.stats import binomtest
from sklearn.metrics import confusion_matrix, precision_recall_fscore_support

from common import CLASSES, RESULTS
from eval_v2 import load_desc, run
from train_eval import metrics

OUT = RESULTS / "final"
OUT.mkdir(parents=True, exist_ok=True)
L_MR = [f"vmaeL_r{r}_{v}" for r in (4, 8, 16) for v in ("full", "person")]
J = ["vjL_ssv2_r8_full", "vjL_ssv2_r8_person"]
PROPOSED = L_MR + J

BASELINES = {  # single backbone, whole frame, 8 fps (or native) - standard linear probe
    "VideoMAE-L (K400)": ["vmaeL_r8_full"],
    "VideoMAE-H (K400)": ["vmaeH_r8_full"],
    "V-JEPA 2-L (SSv2)": ["vjL_ssv2_r8_full"],
    "V-JEPA 2-g (SSL)": ["vjG_r16_full"],
}
ABLATION = {
    "A1 VideoMAE-L, 8 fps, full frame": ["vmaeL_r8_full"],
    "A2 + multi-rate (4/8/16 fps)": [s for s in L_MR if s.endswith("full")],
    "A3 + child-centric crop (1-backbone)": L_MR,
    "A4 V-JEPA 2-L only (full+crop)": J,
    "A5 proposed w/o child crop": [s for s in PROPOSED if s.endswith("full")],
    "A6 proposed w/o multi-rate": ["vmaeL_r8_full", "vmaeL_r8_person"] + J,
    "A7 proposed, late fusion (mean prob.)": ("mean", PROPOSED),
    "A8 proposed, learned stacking": ("mrcc", PROPOSED),
    "MRCC-Net (proposed)": PROPOSED,
}
SEEDS = 3


def cv(F, y, g, folds, spec):
    method, streams = spec if isinstance(spec, tuple) else ("concat", spec)
    return run(F, y, g, folds, method, streams)[0]


def summarise(res):
    return {f"{k}_{s}": f([m[k] for m in res]) for k in res[0] for s, f in (("mean", np.mean), ("std", np.std))}


def mcnemar(a, b):
    """exact McNemar on paired correctness vectors"""
    n01, n10 = int(((~a) & b).sum()), int((a & ~b).sum())
    return n01, n10, (binomtest(n01, n01 + n10, 0.5).pvalue if n01 + n10 else 1.0)


def cluster_boot(ca, cb, g, n=5000, seed=0):
    """95% CI of acc(b) - acc(a), resampling source videos with replacement"""
    rng = np.random.default_rng(seed)
    ug = np.unique(g)
    idx = {u: np.where(g == u)[0] for u in ug}
    d = []
    for _ in range(n):
        s = np.concatenate([idx[u] for u in rng.choice(ug, len(ug))])
        d.append(cb[s].mean() - ca[s].mean())
    return np.percentile(d, [2.5, 97.5])


def main():
    main_rows, abl_rows, pc_rows, sig_rows = [], [], [], []
    for ds in ["esbd", "ssbd"]:
        rows, F = load_desc(ds)
        split = json.loads((RESULTS / "splits" / f"{ds}.json").read_text())
        assert split["clips"] == [r["clip"] for r in rows]
        cls = CLASSES[ds]
        y = np.array([cls.index(r["label"]) for r in rows])
        g = np.array([r["group"] for r in rows])
        for proto in ["grouped", "random"]:
            preds = {}
            for name, spec in {**BASELINES, **ABLATION}.items():
                if name in preds:
                    continue
                Ps = [cv(F, y, g, split[proto][r], spec) for r in range(SEEDS)]
                preds[name] = Ps
                np.savez(OUT / f"pred_{ds}_{proto}_{re.sub(r'[^A-Za-z0-9]+', '_', name).strip('_')}.npz", p=np.stack(Ps), y=y, groups=g)
                res = [metrics(y, P, g) for P in Ps]
                row = dict(dataset=ds, protocol=proto, model=name, **summarise(res))
                (main_rows if name in BASELINES or name.startswith("MRCC") else []).append(row)
                if name in ABLATION:
                    abl_rows.append(row)
                print(f"{ds} {proto:8s} {name:40s} acc={row['acc_mean']:.4f}±{row['acc_std']:.4f} "
                      f"f1m={row['f1_macro_mean']:.4f} vid={row['video_acc_mean']:.4f}", flush=True)

            # per-class + confusion of the proposed model (summed over seeds)
            Pp = preds["MRCC-Net (proposed)"]
            yh = np.concatenate([P.argmax(1) for P in Pp])
            yy = np.tile(y, SEEDS)
            pr, rc, f1, sup = precision_recall_fscore_support(yy, yh, labels=range(len(cls)))
            for i, c in enumerate(cls):
                pc_rows.append(dict(dataset=ds, protocol=proto, cls=c, precision=pr[i], recall=rc[i], f1=f1[i],
                                    support=sup[i] // SEEDS))
            pd.DataFrame(confusion_matrix(yy, yh) / SEEDS, index=cls, columns=cls).to_csv(
                OUT / f"confusion_{ds}_{proto}.csv", float_format="%.2f")

            # significance vs. every baseline and every ablation (pooled over seeds)
            cp = np.concatenate([P.argmax(1) == y for P in Pp])
            gg = np.tile(g, SEEDS)
            for name in list(BASELINES) + [a for a in ABLATION if not a.startswith("MRCC")]:
                cb = np.concatenate([P.argmax(1) == y for P in preds[name]])
                n01, n10, p = mcnemar(cb, cp)
                lo, hi = cluster_boot(cb.astype(float), cp.astype(float), gg)
                sig_rows.append(dict(dataset=ds, protocol=proto, vs=name, acc_other=cb.mean(), acc_proposed=cp.mean(),
                                     only_proposed_correct=n01, only_other_correct=n10, mcnemar_p=p,
                                     diff_ci_lo=lo, diff_ci_hi=hi))

    M = pd.DataFrame(main_rows)
    M.to_csv(OUT / "main_results.csv", index=False)
    pd.DataFrame(abl_rows).to_csv(OUT / "ablation.csv", index=False)
    pd.DataFrame(pc_rows).to_csv(OUT / "per_class.csv", index=False)
    pd.DataFrame(sig_rows).to_csv(OUT / "significance.csv", index=False)
    lk = M.pivot_table(index=["dataset", "model"], columns="protocol", values="acc_mean").reset_index()
    lk["inflation_pp"] = 100 * (lk["random"] - lk["grouped"])
    lk.to_csv(OUT / "leakage.csv", index=False)
    print(lk.to_string())


if __name__ == "__main__":
    main()
