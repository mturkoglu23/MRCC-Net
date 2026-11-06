"""Collect every number used in the manuscript tables into paper/paper_data.json.

Re-run after any experiment changes; build_paper.js reads this file.
"""
import json

import numpy as np
import pandas as pd

from common import PROJECT, RESULTS

F = RESULTS / "final"
OUT = PROJECT / "paper"
OUT.mkdir(exist_ok=True)


def pct(m, s=None, d=2):
    if m is None or (isinstance(m, float) and np.isnan(m)):
        return "–"
    t = f"{100 * m:.{d}f}".replace(".", ",")
    return t if s is None else f"{t} ± {100 * s:.{d}f}".replace(".", ",")


def v1(ds, proto, method):
    """mean / std of v1 baselines from results/cv_results.csv (last 3 rows per setting)"""
    d = pd.read_csv(RESULTS / "cv_results.csv")
    d = d[(d.dataset == ds) & (d.protocol == proto) & (d.method == method)].tail(3)
    if len(d) < 3:
        return None
    return {k: (d[k].mean(), d[k].std(ddof=0)) for k in ["acc", "bacc", "f1_macro", "f1_weighted", "video_acc"]
            if k in d}


def main():
    data = {}
    m = pd.read_csv(F / "main_results.csv")
    ft = pd.read_csv(F / "finetune.csv") if (F / "finetune.csv").exists() else pd.DataFrame()
    ft = ft.drop_duplicates(["dataset", "protocol", "model"], keep="last") if len(ft) else ft
    order = [("v1", "lr_dino", "DINOv2-B (kare tabanlı)"), ("v1", "lr_pose_spec", "Poz hız spektrumu"),
             ("v1", "lr_xclip", "X-CLIP-B/16 (K600)"), ("v1", "repaf", "RePAF-Net (uçtan uca)"),
             ("v2", "V-JEPA 2-g (SSL)", "V-JEPA 2-g (dondurulmuş)"),
             ("v2", "V-JEPA 2-L (SSv2)", "V-JEPA 2-L (dondurulmuş)"),
             ("v2", "VideoMAE-H (K400)", "VideoMAE-H (dondurulmuş)"),
             ("v2", "VideoMAE-L (K400)", "VideoMAE-L (dondurulmuş)"),
             ("ft", "videomae-B fine-tuned", "VideoMAE-B (ince ayarlı)"),
             ("ft", "videomae-L fine-tuned", "VideoMAE-L (ince ayarlı)"),
             ("v2", "MRCC-Net (proposed)", "MRCC-Net (önerilen)")]
    for ds in ["esbd", "ssbd"]:
        rows = []
        for src, key, label in order:
            cell = {}
            for proto in ["grouped", "random"]:
                if src == "v1":
                    r = v1(ds, proto, key)
                    cell[proto] = None if r is None else {k: v for k, v in r.items()}
                else:
                    t = m if src == "v2" else ft
                    d = t[(t.dataset == ds) & (t.protocol == proto) & (t.model == key)] if len(t) else t
                    cell[proto] = None if not len(d) else {
                        k: (d[f"{k}_mean"].item(), d[f"{k}_std"].item())
                        for k in ["acc", "bacc", "f1_macro", "f1_weighted", "video_acc"]}
            g, r = cell["grouped"], cell["random"]
            rows.append([label,
                         pct(*g["acc"]) if g else "–", pct(g["bacc"][0]) if g else "–",
                         pct(g["f1_macro"][0]) if g else "–",
                         pct(g["f1_weighted"][0]) if g and "f1_weighted" in g else "–",
                         pct(g["video_acc"][0]) if g else "–",
                         pct(*r["acc"]) if r else "–", pct(r["f1_macro"][0]) if r else "–"])
        data[f"main_{ds}"] = rows

    a = pd.read_csv(F / "ablation.csv")
    s = pd.read_csv(F / "significance.csv")
    abl = []
    for name in dict.fromkeys(a.model):
        row = [name]
        for ds in ["esbd", "ssbd"]:
            for proto in ["grouped", "random"]:
                d = a[(a.dataset == ds) & (a.protocol == proto) & (a.model == name)]
                row.append(pct(d.acc_mean.item(), d.acc_std.item()))
            p = s[(s.dataset == ds) & (s.protocol == "grouped") & (s.vs == name)]
            row.append("–" if not len(p) else (f"{p.mcnemar_p.item():.3f}".replace(".", ",")
                                                if p.mcnemar_p.item() >= 0.001 else "<0,001"))
        abl.append(row)
    data["ablation"] = abl

    sig = []
    for ds in ["esbd", "ssbd"]:
        for vs in ["VideoMAE-L (K400)", "VideoMAE-H (K400)", "V-JEPA 2-L (SSv2)", "V-JEPA 2-g (SSL)"]:
            p = s[(s.dataset == ds) & (s.protocol == "grouped") & (s.vs == vs)].iloc[0]
            sig.append([ds.upper(), vs, f"+{100 * (p.acc_proposed - p.acc_other):.2f}".replace(".", ","),
                        f"[{100 * p.diff_ci_lo:+.1f}; {100 * p.diff_ci_hi:+.1f}]".replace(".", ","),
                        f"{p.only_proposed_correct}/{p.only_other_correct}",
                        (f"{p.mcnemar_p:.4f}".replace(".", ",") if p.mcnemar_p >= 1e-4 else "<0,0001")])
    data["significance"] = sig

    lk = pd.read_csv(F / "leakage.csv")
    data["leakage"] = [[r.model, pct(lk[(lk.dataset == "esbd") & (lk.model == r.model)].grouped.item()),
                        pct(lk[(lk.dataset == "esbd") & (lk.model == r.model)].random.item()),
                        f"+{lk[(lk.dataset == 'esbd') & (lk.model == r.model)].inflation_pp.item():.1f}".replace(".", ","),
                        pct(lk[(lk.dataset == "ssbd") & (lk.model == r.model)].grouped.item()),
                        pct(lk[(lk.dataset == "ssbd") & (lk.model == r.model)].random.item()),
                        f"+{lk[(lk.dataset == 'ssbd') & (lk.model == r.model)].inflation_pp.item():.1f}".replace(".", ",")]
                       for r in lk[lk.dataset == "esbd"].itertuples()]

    c = pd.read_csv(F / "cross_dataset.csv")
    cross = []
    for name in dict.fromkeys(c.model):
        row = [name]
        for sc in ["ssbd->esbd", "esbd->ssbd", "ssbd+esbd->external"]:
            d = c[(c.scenario == sc) & (c.model == name)].iloc[0]
            row += [pct(d.acc), pct(d.f1_macro)]
        row.append(pct(c[(c.scenario == "ssbd+esbd->external") & (c.model == name)].video_acc.item()))
        cross.append(row)
    data["cross"] = cross

    sc = pd.read_csv(F / "sensitivity_C.csv")
    sens = []
    for ds in ["esbd", "ssbd"]:
        for model in ["MRCC-Net (proposed)", "A1 VideoMAE-L baseline"]:
            d = sc[(sc.dataset == ds) & (sc.protocol == "grouped") & (sc.model == model)].sort_values("C")
            sens.append([ds.upper(), "MRCC-Net" if model.startswith("MRCC") else "VideoMAE-L"] +
                        [pct(v) for v in d.acc_mean])
    data["sens_C"] = sens
    data["sens_Cs"] = [str(v).replace(".", ",") for v in sorted(sc.C.unique())]
    sp = pd.read_csv(F / "sensitivity_pooling.csv")
    data["sens_pool"] = [[p] + [pct(sp[(sp.dataset == ds) & (sp.protocol == pr) & (sp.pooling == p)].acc_mean.item())
                                for ds in ["esbd", "ssbd"] for pr in ["grouped", "random"]]
                         for p in ["mean", "max", "mean+std"]]

    pc = pd.read_csv(F / "per_class.csv")
    data["per_class"] = [[r.dataset.upper(), r.cls, f"{r.precision:.3f}".replace(".", ","),
                          f"{r.recall:.3f}".replace(".", ","), f"{r.f1:.3f}".replace(".", ","), str(r.support)]
                         for r in pc[pc.protocol == "grouped"].itertuples()]

    co = pd.read_csv(F / "cost.csv")
    data["cost"] = co.fillna("").to_dict("records")
    data["ft_log"] = {}
    for mdl in ["B", "L"]:
        try:
            txt = open(f"D:/otizm_work/ft_{mdl}.log", encoding="utf8", errors="ignore").read()
            secs = [int(l.split("(")[-1].rstrip("s)\n")) for l in txt.splitlines() if "fold4 done" in l]
            data["ft_log"][mdl] = secs
        except FileNotFoundError:
            pass
    (OUT / "paper_data.json").write_text(json.dumps(data, ensure_ascii=False, indent=1), encoding="utf8")
    print("written", list(data))


if __name__ == "__main__":
    main()
