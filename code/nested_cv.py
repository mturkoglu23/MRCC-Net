"""Nested, source-grouped cross-validation of the whole model-selection procedure.

Outer loop: the fixed P1 splits (5 folds x 3 repeats). Inner loop: StratifiedGroupKFold(4)
on the outer training part only. The candidate space is the 10 configurations compared
while designing MRCC-Net: 4 single-backbone probes + 3 multi-stream VideoMAE-L variants + 3 two-backbone fusions
(CONFIGS below; late fusion and learned stacking are not included) x 5 regularisation strengths. The inner winner is refit on the
outer training part and scored once on the outer test fold, so the outer estimate is
not selected on the test folds. It is not free of all design bias, because the candidate set itself
was defined on the same data. Outputs results/final/nested_cv.csv and nested_cv_selection.csv.
"""
import json
from collections import Counter

import numpy as np
import pandas as pd
from joblib import Parallel, delayed
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import StratifiedGroupKFold
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

from common import CLASSES, RESULTS, index_dataset
from eval_v2 import load_desc
from final_experiments import BASELINES, PROPOSED
from train_eval import metrics

L = [f"vmaeL_r{r}_{v}" for r in (4, 8, 16) for v in ("full", "person")]
CONFIGS = {**{f"1B {k}": v for k, v in BASELINES.items()},
           "1M VideoMAE-L multi-rate full": [s for s in L if s.endswith("full")],
           "1M VideoMAE-L r8 full+person": ["vmaeL_r8_full", "vmaeL_r8_person"],
           "1M VideoMAE-L multi-rate full+person": L,
           "2M VMAE-L MR f+p + VJEPA2-L f+p (MRCC-Net)": PROPOSED,
           "2M VMAE-L MR f+p + VMAE-H f+p": L + ["vmaeH_r8_full", "vmaeH_r8_person"],
           "2M VMAE-L MR f+p + VJEPA2-g f+p": L + ["vjG_r16_full", "vjG_r16_person"]}
CS = [0.005, 0.01, 0.05, 0.1, 0.5]
OUT = RESULTS / "final"


def clf(C):
    return make_pipeline(StandardScaler(), LogisticRegression(C=C, max_iter=5000, class_weight="balanced"))


def inner_score(X, y, g, tr, C, seed):
    sgk = StratifiedGroupKFold(n_splits=4, shuffle=True, random_state=seed)
    hit = 0
    for a, b in sgk.split(tr, y[tr], g[tr]):
        m = clf(C).fit(X[tr[a]], y[tr[a]])
        hit += (m.predict(X[tr[b]]) == y[tr[b]]).sum()
    return hit / len(tr)


def main():
    res, sel = [], []
    for ds in ["esbd", "ssbd"]:
        rows, F = load_desc(ds, keys=sorted({s for v in CONFIGS.values() for s in v}))
        split = json.loads((RESULTS / "splits" / f"{ds}.json").read_text())
        y = np.array([CLASSES[ds].index(r["label"]) for r in rows])
        g = np.array([r["group"] for r in rows])
        X = {name: np.concatenate([F[s] for s in streams], 1) for name, streams in CONFIGS.items()}
        cands = [(n, C) for n in CONFIGS for C in CS]
        reps = []
        for rep in range(3):
            P = np.zeros((len(y), len(CLASSES[ds])))
            for k, te in enumerate(split["grouped"][rep]):
                te = np.array(te)
                tr = np.setdiff1d(np.arange(len(y)), te)
                sc = Parallel(n_jobs=-1)(delayed(inner_score)(X[n], y, g, tr, C, 100 * rep + k) for n, C in cands)
                n, C = cands[int(np.argmax(sc))]
                P[te] = clf(C).fit(X[n][tr], y[tr]).predict_proba(X[n][te])
                sel.append(dict(dataset=ds, repeat=rep, fold=k, config=n, C=C, inner_acc=max(sc)))
                print(ds, rep, k, n, C, round(max(sc), 4), flush=True)
            reps.append(metrics(y, P, g))
        res.append(dict(dataset=ds, **{f"{m}_{s}": f([r[m] for r in reps]) for m in reps[0]
                                       for s, f in (("mean", np.mean), ("std", np.std))}))
        print(ds, "nested acc", res[-1]["acc_mean"], res[-1]["acc_std"],
              Counter(d["config"] for d in sel if d["dataset"] == ds), flush=True)
    pd.DataFrame(res).to_csv(OUT / "nested_cv.csv", index=False)
    pd.DataFrame(sel).to_csv(OUT / "nested_cv_selection.csv", index=False)


if __name__ == "__main__":
    main()
