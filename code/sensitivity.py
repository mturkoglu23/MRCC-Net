"""Hyper-parameter sensitivity of MRCC-Net (results/final/sensitivity_*.csv).

S1  regularisation strength C of the logistic regression, for the proposed model
    and for the single-backbone baseline (A1), so that the comparison holds at
    every C and not only at the pre-fixed C = 0.05.
S2  temporal pooling of window embeddings: mean (default), max, mean+std.
Same splits, 3 repeats, both protocols.
"""
import json

import numpy as np
import pandas as pd

import eval_v2
from common import CLASSES, RESULTS, index_dataset
from final_experiments import ABLATION, PROPOSED
from train_eval import metrics

OUT = RESULTS / "final"
CS = [0.001, 0.005, 0.01, 0.05, 0.1, 0.5, 1.0]
POOLS = {"mean": lambda a: a.mean(0), "max": lambda a: a.max(0),
         "mean+std": lambda a: np.concatenate([a.mean(0), a.std(0)])}
A1 = ABLATION["A1 VideoMAE-L, 8 fps, full frame"]


def load_pooled(name, pool):
    """window embeddings [2,W,D] -> flip-mean -> pooled over windows"""
    F = {k: [] for k in PROPOSED}
    for r in index_dataset(name):
        z = np.load(eval_v2.V2 / name / (r["clip"][:-4] + ".npz"))
        for k in PROPOSED:
            F[k].append(POOLS[pool](z[k].mean(0)))
    return {k: np.stack(v) for k, v in F.items()}


def evaluate(F, y, g, split, proto, streams):
    res = [metrics(y, eval_v2.run(F, y, g, split[proto][r], "concat", streams)[0], g) for r in range(3)]
    return {f"{k}_{s}": f([m[k] for m in res]) for k in ("acc", "f1_macro") for s, f in
            (("mean", np.mean), ("std", np.std))}


def main():
    rows_c, rows_p = [], []
    for ds in ["esbd", "ssbd"]:
        rows = index_dataset(ds)
        split = json.loads((RESULTS / "splits" / f"{ds}.json").read_text())
        y = np.array([CLASSES[ds].index(r["label"]) for r in rows])
        g = np.array([r["group"] for r in rows])
        Fm = load_pooled(ds, "mean")
        for proto in ["grouped", "random"]:
            for c in CS:
                eval_v2.C = c
                for model, streams in [("MRCC-Net (proposed)", PROPOSED), ("A1 VideoMAE-L baseline", A1)]:
                    m = evaluate(Fm, y, g, split, proto, streams)
                    rows_c.append(dict(dataset=ds, protocol=proto, C=c, model=model, **m))
                    print(f"{ds} {proto:8s} C={c:<6} {model:24s} acc={m['acc_mean']:.4f}±{m['acc_std']:.4f} "
                          f"f1m={m['f1_macro_mean']:.4f}", flush=True)
        eval_v2.C = 0.05
        for pool in POOLS:
            F = Fm if pool == "mean" else load_pooled(ds, pool)
            for proto in ["grouped", "random"]:
                m = evaluate(F, y, g, split, proto, PROPOSED)
                rows_p.append(dict(dataset=ds, protocol=proto, pooling=pool, **m))
                print(f"{ds} {proto:8s} pool={pool:9s} acc={m['acc_mean']:.4f}±{m['acc_std']:.4f} "
                      f"f1m={m['f1_macro_mean']:.4f}", flush=True)
    pd.DataFrame(rows_c).to_csv(OUT / "sensitivity_C.csv", index=False)
    pd.DataFrame(rows_p).to_csv(OUT / "sensitivity_pooling.csv", index=False)


if __name__ == "__main__":
    main()
