"""Evaluate v2 descriptors and the proposed MRCC-Net model (earlier working name: MRCC-Fusion).

MRCC-Net (Multi-Rate Child-Centric Network):
  1. For every (backbone, temporal rate, view) stream, a clip descriptor is the
     mean of window embeddings over time and over original/flipped video.
  2. One L2-regularised logistic model per stream ("experts").
  3. A stacking combiner learns how much to trust each expert from
     out-of-fold expert probabilities computed with an INNER source-video-grouped
     CV on the training fold only (no test information is used).
     Combiner = non-negative weights over experts (softmax-parametrised),
     fitted by minimising log-loss of the weighted mean probability.

All numbers are pooled out-of-fold predictions under the outer protocol
(grouped = source-grouped P1, random = literature-comparable P2), 3 repeats.
"""
import argparse
import json

import numpy as np
from scipy.optimize import minimize
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import StratifiedGroupKFold
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

from common import CLASSES, FEAT, RESULTS, index_dataset
from train_eval import metrics

V2 = FEAT.parent / "features_v2"
STREAMS = [f"{b}_{v}" for b in ["vmaeL_r4", "vmaeL_r8", "vmaeL_r16", "vmaeH_r8", "vjL_ssv2_r8", "vjG_r16"]
           for v in ["full", "person"]]
C = 0.05


def load_desc(name, keys=STREAMS):
    rows = index_dataset(name)
    D = {k: [] for k in keys}
    for r in rows:
        z = np.load(V2 / name / (r["clip"][:-4] + ".npz"))
        for k in keys:
            D[k].append(z[k].mean((0, 1)))
    return rows, {k: np.stack(v) for k, v in D.items()}


def lr():
    return make_pipeline(StandardScaler(), LogisticRegression(C=C, max_iter=5000, class_weight="balanced"))


def fit_weights(probs, y):
    """probs [S,N,K] expert OOF probabilities -> simplex weights minimising NLL."""
    S = probs.shape[0]

    def nll(a):
        w = np.exp(a - a.max())
        w /= w.sum()
        p = np.tensordot(w, probs, 1)
        return -np.log(p[np.arange(len(y)), y] + 1e-9).mean() + 1e-3 * (a ** 2).sum()
    a = minimize(nll, np.zeros(S), method="L-BFGS-B").x
    w = np.exp(a - a.max())
    return w / w.sum()


def mrcc_fit_predict(F, y, g, tr, te, streams, seed=0):
    n_cls = y.max() + 1
    inner = StratifiedGroupKFold(n_splits=4, shuffle=True, random_state=seed)
    oof = np.zeros((len(streams), len(tr), n_cls))
    for itr, ite in inner.split(tr, y[tr], g[tr]):
        for s, k in enumerate(streams):
            m = lr().fit(F[k][tr][itr], y[tr][itr])
            oof[s, ite] = m.predict_proba(F[k][tr][ite])
    w = fit_weights(oof, y[tr])
    Pte = np.stack([lr().fit(F[k][tr], y[tr]).predict_proba(F[k][te]) for k in streams])
    return np.tensordot(w, Pte, 1), w


def run(F, y, g, folds, method, streams):
    P = np.zeros((len(y), y.max() + 1))
    W = []
    for te in folds:
        te = np.array(te)
        tr = np.setdiff1d(np.arange(len(y)), te)
        if method == "single":
            P[te] = lr().fit(F[streams[0]][tr], y[tr]).predict_proba(F[streams[0]][te])
        elif method == "mean":
            P[te] = np.mean([lr().fit(F[k][tr], y[tr]).predict_proba(F[k][te]) for k in streams], 0)
        elif method == "concat":
            X = np.concatenate([F[k] for k in streams], 1)
            P[te] = lr().fit(X[tr], y[tr]).predict_proba(X[te])
        elif method == "mrcc":
            P[te], w = mrcc_fit_predict(F, y, g, tr, te, streams)
            W.append(w)
    return P, (np.mean(W, 0) if W else None)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--dataset", default="esbd")
    ap.add_argument("--protocols", nargs="+", default=["grouped", "random"])
    ap.add_argument("--save", action="store_true")
    args = ap.parse_args()
    rows, F = load_desc(args.dataset)
    split = json.loads((RESULTS / "splits" / f"{args.dataset}.json").read_text())
    assert split["clips"] == [r["clip"] for r in rows]
    y = np.array([CLASSES[args.dataset].index(r["label"]) for r in rows])
    g = np.array([r["group"] for r in rows])

    full = [s for s in STREAMS if s.endswith("_full")]
    configs = [(f"single:{s}", "single", [s]) for s in STREAMS]
    configs += [("mean:all", "mean", STREAMS), ("concat:all", "concat", STREAMS),
                ("mrcc:full_only", "mrcc", full),
                ("mrcc:vmaeL_r8_both", "mrcc", ["vmaeL_r8_full", "vmaeL_r8_person"]),
                ("mrcc:vmaeL_multirate_full", "mrcc", ["vmaeL_r4_full", "vmaeL_r8_full", "vmaeL_r16_full"]),
                ("mrcc:all (proposed)", "mrcc", STREAMS)]
    out = []
    for proto in args.protocols:
        for name, method, streams in configs:
            res, ws = [], []
            for r in range(3):
                P, w = run(F, y, g, split[proto][r], method, streams)
                res.append(metrics(y, P, g))
                if w is not None:
                    ws.append(w)
                if args.save:
                    np.savez(RESULTS / "preds" / f"v2_{args.dataset}_{proto}_{name.split()[0].replace(':', '-')}_r{r}.npz",
                             p=P, y=y, groups=g)
            agg = {k: (np.mean([m[k] for m in res]), np.std([m[k] for m in res])) for k in res[0]}
            out.append(dict(dataset=args.dataset, protocol=proto, config=name,
                            **{f"{k}_mean": v[0] for k, v in agg.items()}, **{f"{k}_std": v[1] for k, v in agg.items()}))
            line = (f"{args.dataset} {proto:8s} {name:32s} acc={agg['acc'][0]:.4f}±{agg['acc'][1]:.4f} "
                    f"f1m={agg['f1_macro'][0]:.4f} bacc={agg['bacc'][0]:.4f} vid={agg['video_acc'][0]:.4f}")
            if ws:
                wm = np.mean(ws, 0)
                top = np.argsort(-wm)[:3]
                line += "  w:" + ",".join(f"{streams[i]}={wm[i]:.2f}" for i in top)
            print(line, flush=True)
    if args.save:
        import pandas as pd
        path = RESULTS / "v2_results.csv"
        pd.DataFrame(out).to_csv(path, mode="a", header=not path.exists(), index=False)


if __name__ == "__main__":
    main()
