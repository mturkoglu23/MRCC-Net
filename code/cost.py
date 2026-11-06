"""Computational cost of MRCC-Net vs. baselines (results/final/cost.csv).

Measured on one RTX 4090, fp16 inference, on N SSBD clips (wall-clock incl. decoding).
"""
import time

import numpy as np
import pandas as pd
import torch

import eval_v2
from common import FEAT, RESULTS, index_dataset
from extract_features import load_clip
from extract_features_v2 import DEV, crop_window, track_boxes, window_index
from final_experiments import PROPOSED

OUT = RESULTS / "final"
N = 12
SPECS = [("vmaeL_r4", 4, 16, 0.5, 224, 224, "L"), ("vmaeL_r8", 8, 16, 0.5, 224, 224, "L"),
         ("vmaeL_r16", 16, 16, 0.5, 224, 224, "L"), ("vjL_ssv2_r8", 8, 16, 0.5, 256, 292, "J")]


def sync():
    torch.cuda.synchronize()
    return time.perf_counter()


def main():
    from transformers import VJEPA2ForVideoClassification, VideoMAEForVideoClassification
    from ultralytics import YOLO
    vmL = VideoMAEForVideoClassification.from_pretrained("MCG-NJU/videomae-large-finetuned-kinetics",
                                                        dtype=torch.float16).to(DEV).eval()
    vjL = VJEPA2ForVideoClassification.from_pretrained("facebook/vjepa2-vitl-fpc16-256-ssv2",
                                                       dtype=torch.float16).to(DEV).eval()
    pose = YOLO(str(FEAT.parent / "weights" / "yolo11x-pose.pt"))
    nparam = lambda m: sum(p.numel() for p in m.parameters())
    p_vm = nparam(vmL.videomae) + nparam(vmL.fc_norm)
    p_vj = nparam(vjL.vjepa2.encoder) + nparam(vjL.pooler)  # predictor is not used at inference
    p_yolo = sum(p.numel() for p in pose.model.parameters())

    @torch.no_grad()
    def emb(kind, x):
        if kind == "L":
            return vmL.fc_norm(vmL.videomae(x[None].half()).last_hidden_state.mean(1))
        return vjL.pooler(vjL.vjepa2(pixel_values_videos=x[None].half()).last_hidden_state)

    rows = index_dataset("ssbd")[:N]
    t_dec, t_yolo, t_feat, dur = [], [], {"full": [], "all": []}, []
    for i, r in enumerate(rows):
        t0 = sync()
        frames, fps = load_clip(r["path"], short=400)
        t1 = sync()
        boxes = track_boxes(pose, frames)
        t2 = sync()
        gpu = torch.from_numpy(frames).to(DEV)
        fb = boxes.copy()
        fb[:, [0, 2]] = frames.shape[2] - boxes[:, [2, 0]]
        tf = 0.0
        for spec, rate, n, hop, out, short, kind in SPECS:
            for w in window_index(len(frames), rate, n, hop):
                for view in ("full", "person"):
                    for fr, bx in ((gpu, boxes), (gpu.flip(2), fb)):
                        emb(kind, crop_window(fr, w, bx if view == "person" else None, out, short))
        t3 = sync()
        if i == 0:  # warm-up
            continue
        t_dec.append(t1 - t0)
        t_yolo.append(t2 - t1)
        t_feat["all"].append(t3 - t2)
        dur.append(len(frames) / 16)
    # classifier: train on all ESBD, time fit + predict
    rows_e, F = eval_v2.load_desc("esbd", keys=PROPOSED)
    X = np.concatenate([F[k] for k in PROPOSED], 1)
    y = np.array([r["label"] for r in rows_e])
    t0 = time.perf_counter()
    clf = eval_v2.lr().fit(X, y)
    t_fit = time.perf_counter() - t0
    t0 = time.perf_counter()
    for _ in range(100):
        clf.predict_proba(X[:1])
    t_pred = (time.perf_counter() - t0) / 100
    n_lr = clf[-1].coef_.size + clf[-1].intercept_.size
    clip = np.mean(dur)
    out = [dict(component="YOLO11x-pose (child tracking)", params_M=p_yolo / 1e6, trainable_M=0,
                sec_per_clip=np.mean(t_yolo)),
           dict(component="VideoMAE-L encoder (frozen)", params_M=p_vm / 1e6, trainable_M=0),
           dict(component="V-JEPA 2-L encoder (frozen)", params_M=p_vj / 1e6, trainable_M=0),
           dict(component="8-stream feature extraction (both encoders, 2 views, 2 flips)",
                sec_per_clip=np.mean(t_feat["all"])),
           dict(component="video decoding", sec_per_clip=np.mean(t_dec)),
           dict(component="logistic regression (trained)", params_M=n_lr / 1e6, trainable_M=n_lr / 1e6,
                train_s_full_esbd=t_fit, sec_per_clip=t_pred),
           dict(component="MRCC-Net total", params_M=(p_yolo + p_vm + p_vj + n_lr) / 1e6, trainable_M=n_lr / 1e6,
                sec_per_clip=np.mean(t_dec) + np.mean(t_yolo) + np.mean(t_feat["all"]) + t_pred,
                train_s_full_esbd=t_fit, mean_clip_seconds=clip)]
    df = pd.DataFrame(out)
    df.to_csv(OUT / "cost.csv", index=False)
    print(df.to_string())


if __name__ == "__main__":
    main()
