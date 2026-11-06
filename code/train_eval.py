"""Cross-validated training / evaluation of baselines and RePAF-Net.

Hyper-parameters are fixed a priori (no tuning on test folds). Metrics are
computed on pooled out-of-fold predictions per repeat; mean +- std over the 3
repeats is reported. Video-level metrics average clip probabilities per source
video.

usage: python train_eval.py --dataset esbd --protocol grouped --methods repaf lr_vmae ...
"""
import argparse
import json
import time

import numpy as np
import pandas as pd
import torch
import torch.nn.functional as F
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, balanced_accuracy_score, f1_score
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

from common import CLASSES, RESULTS
from data import CROP, collate, load_dataset
from models import RePAFNet, pose_spectrum

DEV = "cuda"
HP = dict(epochs=40, views_per_epoch=4, bs=32, lr=3e-4, wd=0.05, smooth=0.1, n_init=3)

NET_VARIANTS = {
    "repaf":          dict(branches=("app", "rep", "spec"), gate=True),
    "abl_no_gate":    dict(branches=("app", "rep", "spec"), gate=False),
    "abl_app":        dict(branches=("app",), gate=False),
    "abl_rep":        dict(branches=("rep",), gate=False),
    "abl_spec":       dict(branches=("spec",), gate=False),
    "abl_app_rep":    dict(branches=("app", "rep"), gate=True),
    "abl_app_spec":   dict(branches=("app", "spec"), gate=True),
    "abl_rep_spec":   dict(branches=("rep", "spec"), gate=True),
}


# ---------------------------------------------------------------- baselines
def pooled(clip, key):
    if key == "pose_spec":
        views = [clip.crop(s, 0)["pose"] for s in clip.test_starts()]
        x = torch.from_numpy(np.stack(views))
        return pose_spectrum(x).flatten(1).mean(0).numpy()
    return getattr(clip, key).mean((0, 1))


def run_lr(clips, tr, te, key, n_cls, y):
    X = np.stack([pooled(c, key) for c in clips])
    clf = make_pipeline(StandardScaler(), LogisticRegression(C=0.1, max_iter=5000, class_weight="balanced"))
    clf.fit(X[tr], y[tr])
    return clf.predict_proba(X[te])


# ---------------------------------------------------------------- network
def sample_batch(clips, idx, rng):
    views = []
    for i in idx:
        c = clips[i]
        start = int(rng.integers(0, max(c.T - CROP, 0) + 1)) if c.T > CROP else int(rng.integers(0, c.T))
        views.append(c.crop(start, int(rng.integers(0, 2))))
    return {k: v.to(DEV) for k, v in collate(views).items()}


def train_net(clips, tr, y, n_cls, variant, seed):
    torch.manual_seed(seed)
    rng = np.random.default_rng(seed)
    net = RePAFNet(n_cls, **NET_VARIANTS[variant]).to(DEV)
    opt = torch.optim.AdamW(net.parameters(), lr=HP["lr"], weight_decay=HP["wd"])
    steps_per_epoch = int(np.ceil(len(tr) * HP["views_per_epoch"] / HP["bs"]))
    sched = torch.optim.lr_scheduler.OneCycleLR(opt, HP["lr"], total_steps=HP["epochs"] * steps_per_epoch, pct_start=0.1)
    counts = np.bincount(y[tr], minlength=n_cls)
    w = torch.tensor(len(tr) / (n_cls * np.maximum(counts, 1)), dtype=torch.float32, device=DEV)
    pool = np.repeat(tr, HP["views_per_epoch"])
    net.train()
    for _ in range(HP["epochs"]):
        rng.shuffle(pool)
        for s in range(0, len(pool), HP["bs"]):
            idx = pool[s:s + HP["bs"]]
            if len(idx) < 2:
                continue
            b = sample_batch(clips, idx, rng)
            loss = F.cross_entropy(net(b), torch.as_tensor(y[idx], device=DEV), weight=w,
                                   label_smoothing=HP["smooth"])
            opt.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(net.parameters(), 1.0)
            opt.step()
            sched.step()
    return net.eval()


@torch.no_grad()
def predict_net(nets, clips, te):
    out = []
    for i in te:
        c = clips[i]
        views = [c.crop(s, f) for s in c.test_starts() for f in (0, 1)]
        b = {k: v.to(DEV) for k, v in collate(views).items()}
        p = torch.stack([n(b).softmax(-1) for n in nets]).mean((0, 1))
        out.append(p.cpu().numpy())
    return np.stack(out)


# ---------------------------------------------------------------- metrics
def metrics(y, p, groups):
    yh = p.argmax(1)
    m = {"acc": accuracy_score(y, yh), "bacc": balanced_accuracy_score(y, yh),
         "f1_macro": f1_score(y, yh, average="macro"), "f1_weighted": f1_score(y, yh, average="weighted")}
    # evaluation unit = (source group, class): a linked recording may contain several behaviours
    df = pd.DataFrame({"g": [f"{a}|{b}" for a, b in zip(groups, y)], "y": y})
    P = pd.DataFrame(p).groupby(df["g"]).mean()
    Y = df.groupby("g")["y"].first().loc[P.index]
    m["video_acc"] = accuracy_score(Y, P.values.argmax(1))
    m["video_f1_macro"] = f1_score(Y, P.values.argmax(1), average="macro")
    return m


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--dataset", required=True)
    ap.add_argument("--protocol", default="grouped", choices=["grouped", "random"])
    ap.add_argument("--methods", nargs="+", default=["lr_vmae", "lr_xclip", "lr_dino", "lr_pose_spec", "repaf"])
    ap.add_argument("--repeats", type=int, default=3)
    args = ap.parse_args()

    classes = CLASSES[args.dataset]
    clips = load_dataset(args.dataset)
    split = json.loads((RESULTS / "splits" / f"{args.dataset}.json").read_text())
    assert split["clips"] == [c.clip for c in clips]
    y = np.array([classes.index(c.label) for c in clips])
    groups = np.array([c.group for c in clips])
    n_cls = len(classes)
    (RESULTS / "preds").mkdir(parents=True, exist_ok=True)
    rows = []
    for method in args.methods:
        for rep in range(args.repeats):
            t0 = time.time()
            P = np.zeros((len(clips), n_cls))
            for k, te in enumerate(split[args.protocol][rep]):
                te = np.array(te)
                tr = np.setdiff1d(np.arange(len(clips)), te)
                if method.startswith("lr_"):
                    P[te] = run_lr(clips, tr, te, method[3:], n_cls, y)
                else:
                    nets = [train_net(clips, tr, y, n_cls, method, seed=1000 * rep + 10 * k + s)
                            for s in range(HP["n_init"])]
                    P[te] = predict_net(nets, clips, te)
            m = metrics(y, P, groups)
            m.update(dataset=args.dataset, protocol=args.protocol, method=method, repeat=rep,
                     secs=round(time.time() - t0, 1))
            rows.append(m)
            np.savez(RESULTS / "preds" / f"{args.dataset}_{args.protocol}_{method}_r{rep}.npz",
                     p=P, y=y, groups=groups, clips=np.array([c.clip for c in clips]))
            print(f"{args.dataset:5s} {args.protocol:8s} {method:14s} r{rep} "
                  f"acc={m['acc']:.4f} f1m={m['f1_macro']:.4f} bacc={m['bacc']:.4f} "
                  f"vid_acc={m['video_acc']:.4f} ({m['secs']}s)", flush=True)
    df = pd.DataFrame(rows)
    path = RESULTS / "cv_results.csv"
    df.to_csv(path, mode="a", header=not path.exists(), index=False)


if __name__ == "__main__":
    main()
