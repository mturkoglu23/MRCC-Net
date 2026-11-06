"""Linked-recording screen: rank cross-group source pairs by feature similarity for visual review.

For every pair of source groups the highest clip-to-clip cosine similarity is taken
(mean over three mean-centred, L2-normalised frozen descriptors: VideoMAE-L 8 fps full
frame and child crop, V-JEPA 2-L child crop). The top 36 (ESBD) and 30 (SSBD) pairs are
rendered as frame strips (20 % and 80 % of each clip) and inspected by eye for the same
child / room / session. Confirmed links go to common.GROUP_ALIAS.
Outputs results/figures/source_audit/{ds}_NN.jpg and pairs.json.
"""
import json

import cv2
import numpy as np

import eval_v2
from common import RESULTS

OUT = RESULTS / "figures" / "source_audit"
KEYS = ["vmaeL_r8_full", "vmaeL_r8_person", "vjL_ssv2_r8_person"]
TOP = {"esbd": 36, "ssbd": 30}


def frames(path):
    cap = cv2.VideoCapture(path)
    n = int(cap.get(7))
    fs = []
    for q in (0.2, 0.8):
        cap.set(1, int(q * n))
        ok, f = cap.read()
        fs.append(cv2.resize(f, (int(200 * f.shape[1] / f.shape[0]), 200)))
    return np.concatenate(fs, 1)


def main():
    OUT.mkdir(parents=True, exist_ok=True)
    allpairs = {}
    for ds, N in TOP.items():
        rows, F = eval_v2.load_desc(ds, keys=KEYS)
        g = np.array([r["group"] for r in rows])
        S = []
        for k in F:
            X = F[k] - F[k].mean(0)
            X /= np.linalg.norm(X, axis=1, keepdims=True)
            S.append(X @ X.T)
        S = np.mean(S, 0)
        G = sorted(set(g))
        idx = {x: np.where(g == x)[0] for x in G}
        pairs = []
        for i, a in enumerate(G):
            for b in G[i + 1:]:
                s = S[np.ix_(idx[a], idx[b])]
                ii, jj = np.unravel_index(s.argmax(), s.shape)
                pairs.append((float(s.max()), a, b, int(idx[a][ii]), int(idx[b][jj])))
        pairs = sorted(pairs, reverse=True)[:N]
        allpairs[ds] = pairs
        strips = []
        for r, (s, a, b, i, j) in enumerate(pairs):
            st = np.concatenate([frames(rows[i]["path"]), np.full((200, 12, 3), 255, np.uint8),
                                 frames(rows[j]["path"])], 1)
            st = cv2.resize(st, (1100, int(st.shape[0] * 1100 / st.shape[1])))
            cv2.putText(st, f"{r} {a} | {b}  {s:.2f}", (5, 24), cv2.FONT_HERSHEY_SIMPLEX, 0.75, (0, 255, 255), 2)
            strips.append(st)
        for k in range(0, len(strips), 6):
            cv2.imwrite(str(OUT / f"{ds}_{k // 6:02d}.jpg"), np.concatenate(strips[k:k + 6], 0),
                        [cv2.IMWRITE_JPEG_QUALITY, 80])
    json.dump(allpairs, open(OUT / "pairs.json", "w"), indent=0)


if __name__ == "__main__":
    main()
