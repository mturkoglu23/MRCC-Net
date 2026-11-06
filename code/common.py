"""Shared paths and dataset indexing."""
import os
import re
from pathlib import Path

PROJECT = Path(__file__).resolve().parents[1]
DATA = PROJECT / "dataset"
EXTERNAL = PROJECT / "external_datasets" / "clips"
WORK = Path(os.environ.get("OTIZM_WORK", "D:/otizm_work"))
FEAT = WORK / "features"
RESULTS = PROJECT / "results"

CLASSES = {
    "esbd": ["ArmFlapping", "HandAction", "HeadBanging", "Spinning"],
    "ssbd": ["ArmFlapping", "HeadBanging", "Spinning"],
}

_NAME = re.compile(r"^v_(?P<cls>[A-Za-z]+)_(?P<src>[A-Za-z]?\d+)(?:_(?P<sub>\d+))?_part")


def parse_name(fname):
    """v_HeadBanging_47_2_part4.mp4 -> ('HeadBanging', '47').

    The optional `_2` sub-index is folded into the source id so that every clip
    cut from the same original video (and its continuation) lands in one group.
    """
    m = _NAME.match(fname)
    if m is None:
        raise ValueError(f"unexpected file name: {fname}")
    return m.group("cls"), m.group("src")


# Different source ids that are the same original video (verified frame by frame,
# dHash distance 0.000): merged into one group so they never cross folds.
# Linked recordings found by visual inspection of the most similar cross-group pairs
# (same child, same room/session): ArmFlapping_41 ~ ArmFlapping_37 and HandAction_05 ~
# ArmFlapping_59 (same person, different behaviours). Group = split unit; the
# video-level evaluation unit stays (group, class), see train_eval.metrics.
GROUP_ALIAS = {"esbd": {"HeadBanging_47": "HeadBanging_46", "ArmFlapping_41": "ArmFlapping_37",
                        "HandAction_05": "ArmFlapping_59"}}
# Same video present in both datasets (dHash distance 0.08): excluded from the
# test side of cross-dataset evaluation.
CROSS_DUPLICATES = {"esbd": {"Spinning_29"}, "ssbd": {"Spinning_22"}}


def index_dataset(name):
    """Return list of dicts: path, clip, label, group (class + source id)."""
    root = EXTERNAL / name.split("ext_")[1] if name.startswith("ext_") else DATA / name
    rows = []
    for f in sorted(os.listdir(root)):
        if not f.endswith(".mp4"):
            continue
        cls, src = parse_name(f)
        rows.append({"dataset": name, "path": str(root / f), "clip": f,
                     "label": cls, "group": GROUP_ALIAS.get(name, {}).get(f"{cls}_{src}", f"{cls}_{src}")})
    return rows
