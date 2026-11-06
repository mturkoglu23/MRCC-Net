"""Nested CV for each *fixed* configuration: only C is chosen in the inner loop.

Same outer P1 splits and inner StratifiedGroupKFold(4) as nested_cv.py, but the model
(stream set) is held fixed, so the outer estimate belongs to one learning procedure,
e.g. the fixed eight-stream MRCC-Net. Out-of-fold probabilities are saved so the
primary comparisons can be repeated on outer (selection-free) predictions.
Outputs results/final/nested_fixed.csv, nested_fixed_C.csv, pred_nested_{ds}_{name}.npz.
"""
import json
import re

import numpy as np
import pandas as pd
from joblib import Parallel, delayed

from common import CLASSES, RESULTS
from eval_v2 import load_desc
from nested_cv import CONFIGS, CS, OUT, clf, inner_score
from train_eval import metrics


def tag(n):
    return re.sub(r"[^A-Za-z0-9]+", "_", n).strip("_")


def main():
    res, sel = [], []
    for ds in ["esbd", "ssbd"]:
        rows, F = load_desc(ds, keys=sorted({s for v in CONFIGS.values() for s in v}))
        split = json.loads((RESULTS / "splits" / f"{ds}.json").read_text())
        y = np.array([CLASSES[ds].index(r["label"]) for r in rows])
        g = np.array([r["group"] for r in rows])
        for name, streams in CONFIGS.items():
            X = np.concatenate([F[s] for s in streams], 1)
            reps, probs = [], []
            for rep in range(3):
                P = np.zeros((len(y), len(CLASSES[ds])))
                for k, te in enumerate(split["grouped"][rep]):
                    te = np.array(te)
                    tr = np.setdiff1d(np.arange(len(y)), te)
                    sc = Parallel(n_jobs=-1)(delayed(inner_score)(X, y, g, tr, C, 100 * rep + k) for C in CS)
                    C = CS[int(np.argmax(sc))]
                    P[te] = clf(C).fit(X[tr], y[tr]).predict_proba(X[te])
                    sel.append(dict(dataset=ds, config=name, repeat=rep, fold=k, C=C))
                reps.append(metrics(y, P, g))
                probs.append(P)
            np.savez(OUT / f"pred_nested_{ds}_{tag(name)}.npz", P=np.stack(probs), y=y, g=g)
            res.append(dict(dataset=ds, config=name, **{f"{m}_{s}": f([r[m] for r in reps]) for m in reps[0]
                                                         for s, f in (("mean", np.mean), ("std", np.std))}))
            print(ds, name, round(res[-1]["acc_mean"], 4), round(res[-1]["acc_std"], 4), flush=True)
    pd.DataFrame(res).to_csv(OUT / "nested_fixed.csv", index=False)
    pd.DataFrame(sel).to_csv(OUT / "nested_fixed_C.csv", index=False)


if __name__ == "__main__":
    main()
