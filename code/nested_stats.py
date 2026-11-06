"""Primary comparisons repeated on outer (nested, C chosen in the inner loop) predictions.

Same cluster sign-flip permutation test and Holm family as stats_cluster.py, but both
models' out-of-fold probabilities come from nested_fixed.py, so neither side benefits
from choosing C on the test folds. Output: results/final/significance_nested.csv
"""
import numpy as np
import pandas as pd

from nested_cv import CONFIGS, OUT
from nested_fixed import tag
from stats_cluster import cluster_test, holm

MRCC = "2M VMAE-L MR f+p + VJEPA2-L f+p (MRCC-Net)"


def load(ds, name):
    z = np.load(OUT / f"pred_nested_{ds}_{tag(name)}.npz", allow_pickle=True)
    return z["P"], z["y"], z["g"]


def main():
    rows = []
    for ds in ["esbd", "ssbd"]:
        pa, y, g = load(ds, MRCC)
        for name in [n for n in CONFIGS if n.startswith("1B ")]:
            pb, _, _ = load(ds, name)
            d, lo, hi, p, ng = cluster_test(pa, pb, y, g)
            rows.append(dict(dataset=ds, vs=name[3:], acc_mrcc=(pa.argmax(-1) == y).mean(),
                             acc_other=(pb.argmax(-1) == y).mean(), delta=d, ci_lo=lo, ci_hi=hi, p_perm=p, n_groups=ng))
    df = pd.DataFrame(rows)
    df["p_holm"] = holm(df.p_perm.values)
    df.to_csv(OUT / "significance_nested.csv", index=False)
    print(df.round(4).to_string())


if __name__ == "__main__":
    main()
