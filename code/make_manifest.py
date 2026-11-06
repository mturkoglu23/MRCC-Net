"""Clip manifest: clip, class, raw source id, merged group, evaluation unit and P1 fold per repeat.

Each clip is also linked to its row in the segment annotation sheet (dataset/{ds}.xlsx): YouTube id
and the annotated intervals. The dataset counts reported in the paper (Table 1) are read from this file. Outputs
results/final/manifest_{ds}.csv and manifest_counts.csv.
"""
import json
from urllib.parse import parse_qs, urlparse

import pandas as pd

from common import PROJECT, RESULTS, index_dataset, parse_name
from make_supplement import read_xlsx


def youtube_id(url):
    q = parse_qs(urlparse(url.strip()).query).get("v")
    return q[0] if q else url.strip()


def sheet_rows(ds, stem):
    """annotation rows for a clip stem (v_Cls_ID or v_Cls_ID_k); sheet names may carry an extra _k suffix"""
    rows = [r for r in read_xlsx(PROJECT / "dataset" / f"{ds}.xlsx")[1:] if r.get("D", "").startswith("v_")]
    return [r for r in rows if r["D"] == stem or r["D"].startswith(stem + "_")]


def main():
    counts = []
    for ds in ["esbd", "ssbd"]:
        rows = index_dataset(ds)
        split = json.loads((RESULTS / "splits" / f"{ds}.json").read_text())
        assert split["clips"] == [r["clip"] for r in rows]
        df = pd.DataFrame({"clip": [r["clip"] for r in rows], "label": [r["label"] for r in rows],
                           "source_id": ["_".join(parse_name(r["clip"])) for r in rows],
                           "group": [r["group"] for r in rows]})
        df["eval_unit"] = df.group + "|" + df.label
        stem = df["clip"].str.replace(r"_part.*", "", regex=True)
        info = {st: sheet_rows(ds, st) for st in stem.unique()}
        assert all(len({youtube_id(r["E"]) for r in v}) == 1 for v in info.values())
        df["youtube_id"] = stem.map(lambda st: youtube_id(info[st][0]["E"]))
        df["annotated_intervals_s"] = stem.map(lambda st: info[st][0].get("F", "").strip())
        # a shared YouTube id must never cross groups
        assert df.groupby("youtube_id").group.nunique().max() == 1
        for rep, folds in enumerate(split["grouped"]):
            for k, te in enumerate(folds):
                df.loc[te, f"fold_r{rep}"] = k
            df[f"fold_r{rep}"] = df[f"fold_r{rep}"].astype(int)
            assert df.groupby("group")[f"fold_r{rep}"].nunique().max() == 1
        df.to_csv(RESULTS / "final" / f"manifest_{ds}.csv", index=False)
        counts.append(dict(dataset=ds, clips=len(df), source_ids=df.source_id.nunique(),
                           unique_videos=df.youtube_id.nunique(), groups=df.group.nunique(), eval_units=df.eval_unit.nunique(),
                           merged=(df.source_id != df.group).groupby(df.source_id).any().sum()))
    pd.DataFrame(counts).to_csv(RESULTS / "final" / "manifest_counts.csv", index=False)
    print(pd.DataFrame(counts))


if __name__ == "__main__":
    main()
