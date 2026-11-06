"""Cross-dataset / external evaluation on the 3 shared classes
(ArmFlapping, HeadBanging, Spinning), using the final model set.

Scenarios
  ssbd->esbd           train on all SSBD, test on ESBD (3 shared classes)
  esbd->ssbd           train on ESBD (3 shared classes), test on all SSBD
  ssbd+esbd->external  train on both, test on clips downloaded 2026-11-01 (missing SSBD
                       videos, SSBD+, Wei et al. 2023) that share no source video with
                       the local data (checked by perceptual hash + embedding NN check)
No hyper-parameter is tuned on the test side; learned-stacking weights come from
inner grouped CV on the training data only.
"""
import numpy as np
import pandas as pd
from sklearn.metrics import confusion_matrix

from common import CROSS_DUPLICATES, RESULTS
from eval_v2 import STREAMS, load_desc, lr, mrcc_fit_predict
from final_experiments import ABLATION, BASELINES, PROPOSED
from train_eval import metrics

OUT = RESULTS / "final"
SHARED = ["ArmFlapping", "HeadBanging", "Spinning"]
CONFIGS = {**BASELINES, "A3 + child-centric crop (1-backbone)": ABLATION["A3 + child-centric crop (1-backbone)"],
           "A5 proposed w/o child crop": ABLATION["A5 proposed w/o child crop"],
           "A7 proposed, late fusion (mean prob.)": ("mean", PROPOSED),
           "A8 proposed, learned stacking": ("mrcc", PROPOSED),
           "MRCC-Net (proposed)": PROPOSED}


def load(name):
    rows, F = load_desc(name)
    keep = np.array([r["label"] in SHARED for r in rows])
    rows = [dict(r, source=name) for r, k in zip(rows, keep) if k]
    return rows, {k: v[keep] for k, v in F.items()}


def drop_dups(part, ds):
    """remove videos that also exist in the other local dataset (test side only)"""
    keep = np.array([r["group"] not in CROSS_DUPLICATES[ds] for r in part[0]])
    return [r for r, k in zip(part[0], keep) if k], {k: v[keep] for k, v in part[1].items()}


def cat(*parts):
    rows = sum((p[0] for p in parts), [])
    return rows, {k: np.concatenate([p[1][k] for p in parts]) for k in STREAMS}


def predict(F, y, g, tr, te, spec):
    method, streams = spec if isinstance(spec, tuple) else ("concat", spec)
    if method == "concat":
        X = np.concatenate([F[k] for k in streams], 1)
        return lr().fit(X[tr], y[tr]).predict_proba(X[te])
    if method == "mean":
        return np.mean([lr().fit(F[k][tr], y[tr]).predict_proba(F[k][te]) for k in streams], 0)
    return mrcc_fit_predict(F, y, g, tr, te, streams)[0]


def unit(x):
    return x / np.linalg.norm(x, axis=1, keepdims=True)


def nn_check(train, test, key="vmaeL_r8_full"):
    """cosine similarity of every test clip to its nearest training clip (near-duplicate check)"""
    s = unit(test[1][key]) @ unit(train[1][key]).T
    j = s.argmax(1)
    return pd.DataFrame({"test_clip": [r["clip"] for r in test[0]], "nearest_train": [train[0][i]["clip"] for i in j],
                         "cos": s.max(1)}).sort_values("cos", ascending=False)


def main():
    esbd, ssbd = load("esbd"), load("ssbd")
    ext = cat(load("ext_ssbd"), load("ext_ssbdplus"), load("ext_wei2023"))
    both = cat(ssbd, esbd)
    nn = nn_check(both, ext)
    nn.to_csv(OUT / "external_nn_check.csv", index=False)
    # reference: nearest neighbour from a DIFFERENT source video inside the local data
    X = unit(both[1]["vmaeL_r8_full"])
    S = X @ X.T
    g = np.array([r["group"] + r["source"] for r in both[0]])
    S[g[:, None] == g[None, :]] = -1
    print(f"external->local NN cosine: max={nn.cos.max():.3f} median={nn.cos.median():.3f} | "
          f"local cross-video NN cosine: max={S.max(1).max():.3f} median={np.median(S.max(1)):.3f}", flush=True)

    scenarios = {"ssbd->esbd": (ssbd, drop_dups(esbd, "esbd")), "esbd->ssbd": (esbd, drop_dups(ssbd, "ssbd")), "ssbd+esbd->external": (both, ext)}
    out = []
    for sc, (train, test) in scenarios.items():
        rows, F = cat(train, test)
        y = np.array([SHARED.index(r["label"]) for r in rows])
        g = np.array([r["group"] + "@" + r["source"] for r in rows])
        src = np.array([r["source"] for r in rows])
        tr, te = np.arange(len(train[0])), np.arange(len(train[0]), len(rows))
        tag = sc.replace("->", "_to_").replace("+", "_")
        for name, spec in CONFIGS.items():
            P = predict(F, y, g, tr, te, spec)
            m = metrics(y[te], P, g[te])
            m.update(scenario=sc, model=name, n_train=len(tr), n_test=len(te), n_test_videos=len(set(g[te])))
            if sc.endswith("external"):
                for s in ["ext_ssbd", "ext_ssbdplus", "ext_wei2023"]:
                    k = src[te] == s
                    m[f"acc_{s}"] = (P[k].argmax(1) == y[te][k]).mean()
            out.append(m)
            if name.startswith("MRCC"):
                np.savez(OUT / f"pred_cross_{tag}.npz", p=P, y=y[te], groups=g[te],
                         clips=np.array([r["clip"] for r in test[0]]))
                pd.DataFrame(confusion_matrix(y[te], P.argmax(1), labels=range(3)), index=SHARED,
                             columns=SHARED).to_csv(OUT / f"confusion_cross_{tag}.csv")
            print(f"{sc:22s} {name:40s} acc={m['acc']:.4f} f1m={m['f1_macro']:.4f} "
                  f"bacc={m['bacc']:.4f} vid={m['video_acc']:.4f}", flush=True)
    pd.DataFrame(out).to_csv(OUT / "cross_dataset.csv", index=False)


if __name__ == "__main__":
    main()
