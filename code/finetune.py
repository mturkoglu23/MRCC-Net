"""Fine-tuned (end-to-end) VideoMAE baseline under the same splits as MRCC-Net.

Represents the literature's approach (e.g. Deng et al. 2023 fine-tune a video
transformer) inside our protocol. Recipe fixed in advance, no tuning on test folds:
  input   16 frames at 8 fps (2 s), random-resized crop (scale 0.5-1) to 224 + h-flip
  optim   AdamW, lr 5e-5 backbone / 1e-3 head, wd 0.05, cosine, 1 warm-up epoch,
          12 epochs x 3 random windows per clip, batch 8, bf16, label smoothing 0.1,
          class-balanced cross-entropy
  test    all 2 s windows (1 s hop) x {original, flipped}, mean softmax per clip
Outputs results/final/finetune.csv and pred_ft_<model>_<ds>_<proto>.npz
"""
import argparse
import json
import math
import time
from pathlib import Path

import cv2
import numpy as np
import pandas as pd
import torch
import torch.nn.functional as F

from common import CLASSES, RESULTS, WORK, index_dataset
from extract_features import IMNET_MEAN, IMNET_STD, load_clip
from train_eval import metrics

DEV = "cuda"
CACHE = WORK / "ft_cache"
OUT = RESULTS / "final"
MODELS = {"videomae-B": "MCG-NJU/videomae-base-finetuned-kinetics",
          "videomae-L": "MCG-NJU/videomae-large-finetuned-kinetics"}
HP = dict(epochs=12, views=3, bs=8, lr=5e-5, lr_head=1e-3, wd=0.05, smooth=0.1, n=16, side=256, crop=224)


def cache(ds):
    """frames at 8 fps, short side 256, centre square 256 -> uint8 [T,256,256,3]"""
    d = CACHE / ds
    d.mkdir(parents=True, exist_ok=True)
    for r in index_dataset(ds):
        dst = d / (r["clip"][:-4] + ".npy")
        if dst.exists():
            continue
        fr, _ = load_clip(r["path"], short=HP["side"])
        fr = fr[::2]
        out = []
        for f in fr:
            h, w = f.shape[:2]
            s = HP["side"] / min(h, w)
            f = cv2.resize(f, (max(HP["side"], round(w * s)), max(HP["side"], round(h * s))), interpolation=cv2.INTER_AREA)
            h, w = f.shape[:2]
            t, l = (h - HP["side"]) // 2, (w - HP["side"]) // 2
            out.append(f[t:t + HP["side"], l:l + HP["side"]])
        np.save(dst, np.stack(out))


def norm(x):
    """uint8 [B,T,H,W,3] cuda -> float [B,T,3,H,W] normalised"""
    x = x.permute(0, 1, 4, 2, 3).float() / 255
    return (x - IMNET_MEAN.view(1, 1, 3, 1, 1)) / IMNET_STD.view(1, 1, 3, 1, 1)


def train_batch(clips, rng):
    n, c = HP["n"], HP["crop"]
    out = []
    for v in clips:
        T = len(v)
        s = rng.integers(0, max(T - n, 0) + 1)
        x = torch.from_numpy(v[np.arange(s, s + n) % T]).to(DEV)  # [n,256,256,3]
        sc = rng.uniform(0.5, 1.0)
        side = int(HP["side"] * math.sqrt(sc))
        t, l = rng.integers(0, HP["side"] - side + 1, 2)
        x = x[:, t:t + side, l:l + side].permute(0, 3, 1, 2).float()
        x = F.interpolate(x, size=(c, c), mode="bilinear", antialias=True, align_corners=False)
        if rng.random() < 0.5:
            x = x.flip(3)
        out.append(x.permute(0, 2, 3, 1))
    return norm(torch.stack(out))


@torch.no_grad()
def predict(model, v):
    n, c, T = HP["n"], HP["crop"], len(v)
    starts = list(range(0, max(T - n, 0) + 1, 8)) or [0]
    x = torch.from_numpy(np.stack([v[np.arange(s, s + n) % T] for s in starts])).to(DEV)
    B = x.shape[0]
    x = F.interpolate(x.flatten(0, 1).permute(0, 3, 1, 2).float(), size=(c, c), mode="bilinear",
                      antialias=True, align_corners=False).permute(0, 2, 3, 1).view(B, n, c, c, 3)
    x = norm(x)
    p = []
    for xx in (x, x.flip(4)):
        for i in range(0, B, 16):
            with torch.autocast("cuda", dtype=torch.bfloat16):
                p.append(model(pixel_values=xx[i:i + 16]).logits.float().softmax(-1))
    return torch.cat(p).mean(0).cpu().numpy()


def fit_predict(name, vids, y, tr, te, n_cls, seed):
    from transformers import VideoMAEForVideoClassification
    torch.manual_seed(seed)
    rng = np.random.default_rng(seed)
    model = VideoMAEForVideoClassification.from_pretrained(MODELS[name], num_labels=n_cls,
                                                           ignore_mismatched_sizes=True).to(DEV)
    if name.endswith("L"):
        model.gradient_checkpointing_enable()
    head = [p for n, p in model.named_parameters() if n.startswith("classifier") or "fc_norm" in n]
    body = [p for n, p in model.named_parameters() if not (n.startswith("classifier") or "fc_norm" in n)]
    opt = torch.optim.AdamW([{"params": body, "lr": HP["lr"]}, {"params": head, "lr": HP["lr_head"]}],
                            weight_decay=HP["wd"])
    order = np.repeat(tr, HP["views"])
    steps_ep = math.ceil(len(order) / HP["bs"])
    total, warm = HP["epochs"] * steps_ep, steps_ep
    sched = torch.optim.lr_scheduler.LambdaLR(
        opt, lambda s: (s + 1) / warm if s < warm else 0.5 * (1 + math.cos(math.pi * (s - warm) / (total - warm))))
    w = torch.tensor(len(tr) / (n_cls * np.bincount(y[tr], minlength=n_cls).clip(1)), dtype=torch.float32, device=DEV)
    yt = torch.from_numpy(y).to(DEV)
    model.train()
    for ep in range(HP["epochs"]):
        rng.shuffle(order)
        for i in range(0, len(order), HP["bs"]):
            b = order[i:i + HP["bs"]]
            x = train_batch([vids[j] for j in b], rng)
            with torch.autocast("cuda", dtype=torch.bfloat16):
                logits = model(pixel_values=x).logits.float()
            loss = F.cross_entropy(logits, yt[b], weight=w, label_smoothing=HP["smooth"])
            opt.zero_grad(set_to_none=True)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            opt.step()
            sched.step()
    model.eval()
    P = np.stack([predict(model, vids[j]) for j in te])
    del model, opt
    torch.cuda.empty_cache()
    return P


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", default="videomae-B", choices=list(MODELS))
    ap.add_argument("--datasets", nargs="+", default=["esbd", "ssbd"])
    ap.add_argument("--protocols", nargs="+", default=["grouped", "random"])
    ap.add_argument("--repeats", type=int, default=3)
    ap.add_argument("--folds", type=int, default=5, help="debug: limit folds")
    args = ap.parse_args()
    for ds in args.datasets:
        cache(ds)
        rows = index_dataset(ds)
        split = json.loads((RESULTS / "splits" / f"{ds}.json").read_text())
        assert split["clips"] == [r["clip"] for r in rows]
        y = np.array([CLASSES[ds].index(r["label"]) for r in rows])
        g = np.array([r["group"] for r in rows])
        vids = [np.load(CACHE / ds / (r["clip"][:-4] + ".npy")) for r in rows]
        for proto in args.protocols:
            res, Ps = [], []
            for rep in range(args.repeats):
                P = np.zeros((len(y), len(CLASSES[ds])))
                t0 = time.time()
                for k, te in enumerate(split[proto][rep][:args.folds]):
                    te = np.array(te)
                    tr = np.setdiff1d(np.arange(len(y)), te)
                    P[te] = fit_predict(args.model, vids, y, tr, te, len(CLASSES[ds]), seed=1000 * rep + k)
                    print(f"  {ds} {proto} r{rep} fold{k} done ({time.time() - t0:.0f}s)", flush=True)
                if args.folds < 5:
                    return
                m = metrics(y, P, g)
                res.append(m)
                Ps.append(P)
                print(f"{args.model} {ds} {proto:8s} r{rep} acc={m['acc']:.4f} f1m={m['f1_macro']:.4f} "
                      f"vid={m['video_acc']:.4f} ({time.time() - t0:.0f}s)", flush=True)
            np.savez(OUT / f"pred_ft_{args.model}_{ds}_{proto}.npz", p=np.stack(Ps), y=y, groups=g)
            row = dict(dataset=ds, protocol=proto, model=f"{args.model} fine-tuned",
                       **{f"{k}_{s}": f([m[k] for m in res]) for k in res[0]
                          for s, f in (("mean", np.mean), ("std", np.std))})
            path = OUT / "finetune.csv"
            pd.DataFrame([row]).to_csv(path, mode="a", header=not path.exists(), index=False)


if __name__ == "__main__":
    main()
