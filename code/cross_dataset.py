"""Cross-dataset and external-test evaluation on the 3 shared classes.

Scenarios
  ssbd -> esbd        train on all SSBD, test on ESBD (ArmFlapping/HeadBanging/Spinning)
  esbd -> ssbd        train on ESBD (3 shared classes), test on all SSBD
  both -> external    train on SSBD + ESBD (3 classes), test on clips downloaded on
                      2026-11-01 from SSBD (missing ones), SSBD+ and Wei et al. 2023,
                      excluding clips that share a source video with local ESBD.
Each network result is a 5-init ensemble; 3 independent repeats -> mean +- std.
"""
import numpy as np
import pandas as pd

from common import RESULTS
from data import load_dataset
from train_eval import metrics, predict_net, run_lr, train_net

SHARED = ["ArmFlapping", "HeadBanging", "Spinning"]
METHODS = ["lr_vmae", "lr_xclip", "lr_dino", "repaf"]


def subset(clips):
    return [c for c in clips if c.label in SHARED]


def main():
    esbd, ssbd = subset(load_dataset("esbd")), load_dataset("ssbd")
    ext = sum((load_dataset(n) for n in ["ext_ssbd", "ext_ssbdplus", "ext_wei2023"]), [])
    scenarios = {"ssbd->esbd": (ssbd, esbd), "esbd->ssbd": (esbd, ssbd),
                 "ssbd+esbd->external": (ssbd + esbd, ext)}
    rows = []
    for name, (train, test) in scenarios.items():
        clips = train + test
        y = np.array([SHARED.index(c.label) for c in clips])
        groups = np.array([c.group for c in clips])
        tr, te = np.arange(len(train)), np.arange(len(train), len(clips))
        for method in METHODS:
            for rep in range(3 if method == "repaf" else 1):
                if method.startswith("lr_"):
                    P = run_lr(clips, tr, te, method[3:], 3, y)
                else:
                    nets = [train_net(clips, tr, y, 3, method, seed=7000 + 100 * rep + s) for s in range(5)]
                    P = predict_net(nets, clips, te)
                m = metrics(y[te], P, groups[te])
                m.update(scenario=name, method=method, repeat=rep, n_train=len(tr), n_test=len(te))
                rows.append(m)
                np.savez(RESULTS / "preds" / f"cross_{name.replace('->', '_to_')}_{method}_r{rep}.npz",
                         p=P, y=y[te], groups=groups[te], clips=np.array([c.clip for c in test]))
                print(f"{name:22s} {method:10s} r{rep} acc={m['acc']:.4f} f1m={m['f1_macro']:.4f} "
                      f"bacc={m['bacc']:.4f} vid_acc={m['video_acc']:.4f}", flush=True)
    pd.DataFrame(rows).to_csv(RESULTS / "cross_dataset_results.csv", index=False)


if __name__ == "__main__":
    main()
