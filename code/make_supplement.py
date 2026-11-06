"""Reviewer supplement S1: manifests, splits, segment annotations, audit decisions, model revisions,
out-of-fold predictions and the code that recomputes every reported comparison. No video files.
Output: paper/supplementary_S1/ and paper/Supplementary_S1.zip
"""
import re
import shutil
import zipfile

import pandas as pd

from common import PROJECT, RESULTS

OUT = PROJECT / "paper" / "supplementary_S1"


def read_xlsx(path):
    """Minimal xlsx reader (first sheet) so no openpyxl dependency is needed."""
    z = zipfile.ZipFile(path)
    ss = []
    if "xl/sharedStrings.xml" in z.namelist():
        ss = ["".join(re.findall(r"<t[^>]*>([^<]*)</t>", s))
              for s in re.findall(r"<si>(.*?)</si>", z.read("xl/sharedStrings.xml").decode("utf8"), re.S)]
    rows = []
    for r in re.findall(r"<row[^>]*>(.*?)</row>", z.read("xl/worksheets/sheet1.xml").decode("utf8"), re.S):
        vals = {}
        for attrs, body in re.findall(r"<c ([^>]*?)(?:/>|>(.*?)</c>)", r, re.S):
            col = re.match(r'r="([A-Z]+)', attrs).group(1)
            v = re.search(r"<v>([^<]*)</v>", body or "")
            v = v.group(1) if v else ""
            if 't="s"' in attrs and v:
                v = ss[int(v)]
            vals[col] = v
        rows.append(vals)
    return rows


def annotations(ds):
    rows = read_xlsx(PROJECT / "dataset" / f"{ds}.xlsx")
    cols = ["A", "C", "D", "E", "F", "G", "H", "I", "J"]
    names = ["source", "category", "video", "url", "intervals_s", "n_segments", "segment_lengths_s", "n_persons",
             "subtype_note"]
    df = pd.DataFrame([[r.get(c, "") for c in cols] for r in rows[1:]], columns=names)
    used = set(pd.read_csv(RESULTS / "final" / f"manifest_{ds}.csv")["clip"].str.extract(r"^(v_[A-Za-z]+_[A-Za-z]?\d+)")[0])
    df = df[df.video.isin(used)]
    return df


def main():
    if OUT.exists():
        shutil.rmtree(OUT)
    for sub in ["manifest", "splits", "annotations", "audit", "models", "predictions", "results", "code"]:
        (OUT / sub).mkdir(parents=True)
    F = RESULTS / "final"
    for f in ["manifest_esbd.csv", "manifest_ssbd.csv", "manifest_counts.csv"]:
        shutil.copy(F / f, OUT / "manifest" / f)
    for f in (RESULTS / "splits").glob("*.json"):
        if "before" not in f.name:
            shutil.copy(f, OUT / "splits" / f.name)
    for ds in ["esbd", "ssbd"]:
        annotations(ds).to_csv(OUT / "annotations" / f"segments_{ds}.csv", index=False)
    ext = PROJECT / "external_datasets"
    for f in ["clips_manifest.csv", "all_urls_status.csv"]:
        shutil.copy(ext / f, OUT / "annotations" / f"external_{f}")
    shutil.copy(ext / "clips_overlap_with_local" / "OVERLAP.csv", OUT / "audit" / "external_overlap_removed.csv")
    shutil.copy(RESULTS / "figures" / "source_audit" / "pairs.json", OUT / "audit" / "linked_pair_candidates.json")
    for f in ["tracking_audit.csv", "tracking_audit_manual.csv", "external_nn_check.csv"]:
        shutil.copy(F / f, OUT / "audit" / f)
    shutil.copy(RESULTS / "model_revisions.txt", OUT / "models" / "model_revisions.txt")
    for f in F.glob("pred_*.npz"):
        shutil.copy(f, OUT / "predictions" / f.name)
    for f in F.glob("*.csv"):
        if not f.name.startswith(("manifest_", "tracking_audit", "external_nn")):
            shutil.copy(f, OUT / "results" / f.name)
    for f in (PROJECT / "code").glob("*.py"):
        shutil.copy(f, OUT / "code" / f.name)
    shutil.copy(PROJECT / "paper" / "supplementary_README.md", OUT / "README.md")
    zp = PROJECT / "paper" / "Supplementary_S1.zip"
    with zipfile.ZipFile(zp, "w", zipfile.ZIP_DEFLATED) as z:
        for f in sorted(OUT.rglob("*")):
            if f.is_file():
                z.write(f, f.relative_to(OUT.parent))
    print(zp, round(zp.stat().st_size / 1e6, 1), "MB", sum(1 for f in OUT.rglob("*") if f.is_file()), "files")


if __name__ == "__main__":
    main()
