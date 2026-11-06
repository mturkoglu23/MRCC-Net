"""Create evaluation splits.

Protocol A (primary, leakage-free): StratifiedGroupKFold, groups = source video,
5 folds x 3 repeats.
Protocol B (literature-comparable, leaky): StratifiedKFold over clips,
5 folds x 3 repeats. Clips of one source video can fall on both sides.
"""
import json
from collections import Counter

from sklearn.model_selection import StratifiedGroupKFold, StratifiedKFold

from common import RESULTS, index_dataset

SEEDS = [0, 1, 2]


def main():
    out = RESULTS / "splits"
    out.mkdir(parents=True, exist_ok=True)
    for name in ["esbd", "ssbd"]:
        rows = index_dataset(name)
        y = [r["label"] for r in rows]
        g = [r["group"] for r in rows]
        splits = {"clips": [r["clip"] for r in rows], "labels": y, "groups": g,
                  "grouped": [], "random": []}
        for seed in SEEDS:
            sgk = StratifiedGroupKFold(n_splits=5, shuffle=True, random_state=seed)
            splits["grouped"].append([te.tolist() for _, te in sgk.split(y, y, g)])
            skf = StratifiedKFold(n_splits=5, shuffle=True, random_state=seed)
            splits["random"].append([te.tolist() for _, te in skf.split(y, y)])
        # sanity: no group crosses folds in the grouped protocol
        for rep in splits["grouped"]:
            seen = {}
            for k, te in enumerate(rep):
                for i in te:
                    assert seen.setdefault(g[i], k) == k
        (out / f"{name}.json").write_text(json.dumps(splits))
        print(name, len(rows), "clips", len(set(g)), "groups", Counter(y))
        for k, te in enumerate(splits["grouped"][0]):
            print("  fold", k, Counter(y[i] for i in te))


if __name__ == "__main__":
    main()
