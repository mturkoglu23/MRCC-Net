"""Compact (1-2 backbone) variants of the multi-rate child-centric fusion."""
import json
import sys

import numpy as np

from common import CLASSES, RESULTS, index_dataset
from eval_v2 import load_desc, run
from train_eval import metrics

L = [f"vmaeL_r{r}_{v}" for r in (4, 8, 16) for v in ("full", "person")]
CONFIGS = {
    "1M  VideoMAE-L r8 full (baseline)": ["vmaeL_r8_full"],
    "1M  VideoMAE-L multi-rate full": [s for s in L if s.endswith("full")],
    "1M  VideoMAE-L r8 full+person": ["vmaeL_r8_full", "vmaeL_r8_person"],
    "1M  VideoMAE-L multi-rate full+person": L,
    "2M  VMAE-L MR f+p + VJEPA2-L ssv2 f+p": L + ["vjL_ssv2_r8_full", "vjL_ssv2_r8_person"],
    "2M  VMAE-L MR f+p + VMAE-H f+p": L + ["vmaeH_r8_full", "vmaeH_r8_person"],
    "2M  VMAE-L MR f+p + VJEPA2-g f+p": L + ["vjG_r16_full", "vjG_r16_person"],
}

out = []
for ds in ["esbd", "ssbd"]:
    rows, F = load_desc(ds)
    split = json.loads((RESULTS / "splits" / f"{ds}.json").read_text())
    y = np.array([CLASSES[ds].index(r["label"]) for r in rows])
    g = np.array([r["group"] for r in rows])
    for proto in ["grouped", "random"]:
        for name, streams in CONFIGS.items():
            res = [metrics(y, run(F, y, g, split[proto][r], "concat", streams)[0], g) for r in range(3)]
            a = {k: (np.mean([m[k] for m in res]), np.std([m[k] for m in res])) for k in res[0]}
            out.append(dict(dataset=ds, protocol=proto, config=name, **{f"{k}_mean": v[0] for k, v in a.items()},
                            **{f"{k}_std": v[1] for k, v in a.items()}))
            print(f"{ds} {proto:8s} {name:40s} acc={a['acc'][0]:.4f}±{a['acc'][1]:.4f} "
                  f"f1m={a['f1_macro'][0]:.4f} wf1={a['f1_weighted'][0]:.4f} vid={a['video_acc'][0]:.4f}", flush=True)
import pandas as pd
pd.DataFrame(out).to_csv(RESULTS / "compact_results.csv", index=False)
