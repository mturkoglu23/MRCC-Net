"""Quick exploration of clip-level descriptors and their fusion (linear models).

Descriptors (all averaged over original + flipped views and over 4 s crops):
  vmae, xclip, dino           pooled backbone embeddings
  pose_spec                   joint-group velocity spectra
  dino_ac / pose_ac           temporal autocorrelation (lags 1..32) of DINO /
                              pose trajectories -> appearance-free periodicity
"""
import argparse
import json

import numpy as np
import torch
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

from common import CLASSES, RESULTS
from data import load_dataset
from models import pose_spectrum
from train_eval import metrics


def autocorr(x, max_lag=32):
    """x [T,C] -> mean normalised autocorrelation over channels for lags 1..max_lag."""
    x = x - x.mean(0, keepdims=True)
    x = x / (np.linalg.norm(x, axis=0, keepdims=True) + 1e-6)
    T = len(x)
    out = [((x[:-k] * x[k:]).sum(0)).mean() if k < T else 0.0 for k in range(1, max_lag + 1)]
    return np.array(out, np.float32)


def descriptors(c):
    d = {k: getattr(c, k).mean((0, 1)) for k in ["vmae", "xclip", "dino"]}
    views = [c.crop(s, 0)["pose"] for s in c.test_starts()]
    d["pose_spec"] = pose_spectrum(torch.from_numpy(np.stack(views))).flatten(1).mean(0).numpy()
    pcs = c.dino[0] - c.dino[0].mean(0)
    u, s, vt = np.linalg.svd(pcs, full_matrices=False)
    d["dino_ac"] = autocorr(pcs @ vt[:16].T)
    d["pose_ac"] = np.concatenate([autocorr(c.pose[0][:, idx].reshape(c.T, -1))
                                   for idx in ([0, 1, 2, 3, 4], [5, 6], [7, 8], [9, 10], [11, 12])])
    return d


def cv(X, y, groups, folds, C):
    P = np.zeros((len(y), y.max() + 1))
    for te in folds:
        te = np.array(te)
        tr = np.setdiff1d(np.arange(len(y)), te)
        clf = make_pipeline(StandardScaler(), LogisticRegression(C=C, max_iter=5000, class_weight="balanced"))
        clf.fit(X[tr], y[tr])
        P[te] = clf.predict_proba(X[te])
    return P


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--dataset", default="esbd")
    ap.add_argument("--protocol", default="grouped")
    args = ap.parse_args()
    clips = load_dataset(args.dataset)
    split = json.loads((RESULTS / "splits" / f"{args.dataset}.json").read_text())
    y = np.array([CLASSES[args.dataset].index(c.label) for c in clips])
    g = np.array([c.group for c in clips])
    D = [descriptors(c) for c in clips]
    feats = {k: np.stack([d[k] for d in D]) for k in D[0]}
    combos = [["vmae"], ["xclip"], ["pose_spec"], ["dino_ac"], ["pose_ac"],
              ["pose_spec", "pose_ac", "dino_ac"], ["vmae", "xclip"],
              ["vmae", "pose_spec"], ["vmae", "pose_ac", "dino_ac"],
              ["vmae", "pose_spec", "pose_ac", "dino_ac"], ["vmae", "xclip", "pose_spec", "pose_ac", "dino_ac"]]
    for combo in combos:
        for C in [0.01, 0.1]:
            X = np.concatenate([feats[k] for k in combo], 1)
            accs = [metrics(y, cv(X, y, g, split[args.protocol][r], C), g) for r in range(3)]
            print(f"{'+'.join(combo):45s} C={C:<5} acc={np.mean([a['acc'] for a in accs]):.4f} "
                  f"f1m={np.mean([a['f1_macro'] for a in accs]):.4f}", flush=True)
    # late fusion: average probabilities of per-descriptor models
    for combo in [["vmae", "xclip"], ["vmae", "pose_spec"], ["vmae", "xclip", "pose_spec"]]:
        accs = []
        for r in range(3):
            P = np.mean([cv(feats[k], y, g, split[args.protocol][r], 0.1) for k in combo], 0)
            accs.append(metrics(y, P, g))
        print(f"LATE {'+'.join(combo):40s} acc={np.mean([a['acc'] for a in accs]):.4f} "
              f"f1m={np.mean([a['f1_macro'] for a in accs]):.4f}", flush=True)


if __name__ == "__main__":
    main()
