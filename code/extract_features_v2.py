"""Second-stage features: child-centric views, multiple temporal rates and
stronger video foundation models.

Views
  full    : shortest-edge resize + centre crop of the whole frame (as v1)
  person  : square crop around the tracked child, fixed per window (the union of
            the child's boxes over the window, x1.25), so the motion inside the
            window is preserved while background / aspect-ratio effects are removed
Backbones / temporal sampling (16 frames per window unless stated)
  vmaeL_r{4,8,16}  VideoMAE-L K400 at 4, 8, 16 fps (4 s, 2 s, 1 s windows)
  vmaeH_r8         VideoMAE-H K400 at 8 fps
  vjL_ssv2_r8      V-JEPA 2 ViT-L fine-tuned on Something-Something-v2 (motion-centric)
  vjG_r16          V-JEPA 2 ViT-g self-supervised encoder, 64 frames @ 16 fps (4 s)
Every array is stored for the original and the horizontally flipped video: [2, W, D].
"""
import argparse

import numpy as np
import torch
import torch.nn.functional as F
from tqdm import tqdm

from common import FEAT, index_dataset
from extract_features import IMNET_MEAN, IMNET_STD, load_clip

DEV = "cuda"
OUT = FEAT.parent / "features_v2"
BASE = 16  # fps of the common timeline


def window_index(T, rate, n=16, hop_s=0.5):
    step = BASE // rate
    span = n * step
    hop = int(hop_s * BASE)
    starts = list(range(0, max(T - span, 0) + 1, hop)) or [0]
    return [(np.arange(s, s + span, step) % T) for s in starts]


def track_boxes(pose_model, frames):
    """Per-frame box of the tracked child (largest first, then nearest centre)."""
    T, h, w = frames.shape[:3]
    boxes = np.full((T, 4), np.nan, np.float32)
    prev = None
    for i in range(0, T, 32):
        for j, r in enumerate(pose_model(list(frames[i:i + 32][..., ::-1]), verbose=False, device=0)):
            if len(r.boxes) == 0:
                continue
            b = r.boxes.xyxy.cpu().numpy()
            c = (b[:, :2] + b[:, 2:]) / 2
            area = (b[:, 2] - b[:, 0]) * (b[:, 3] - b[:, 1])
            k = int(np.argmax(area)) if prev is None else int(np.argmin(
                np.linalg.norm(c - prev, axis=1) / max(h, w) - 0.5 * area / (h * w)))
            prev = c[k]
            boxes[i + j] = b[k]
    return boxes


def crop_window(frames, idx, boxes, out, short):
    """frames uint8 cuda [T,H,W,3]; returns normalised [n,3,out,out]."""
    x = frames[idx].permute(0, 3, 1, 2).float() / 255
    H, W = x.shape[-2:]
    if boxes is not None:
        b = boxes[idx]
        b = b[~np.isnan(b[:, 0])]
    if boxes is not None and len(b):
        x1, y1 = b[:, 0].min(), b[:, 1].min()
        x2, y2 = b[:, 2].max(), b[:, 3].max()
        cx, cy, s = (x1 + x2) / 2, (y1 + y2) / 2, 1.25 * max(x2 - x1, y2 - y1)
        s = max(s, 0.3 * min(H, W))
        l, t = int(round(cx - s / 2)), int(round(cy - s / 2))
        s = int(round(s))
        pad = [max(0, -l), max(0, l + s - W), max(0, -t), max(0, t + s - H)]
        x = F.pad(x, pad)  # zero-pad if the square leaves the frame
        x = x[..., t + pad[2]:t + pad[2] + s, l + pad[0]:l + pad[0] + s]
        x = F.interpolate(x, size=(out, out), mode="bilinear", antialias=True, align_corners=False)
    else:
        sc = short / min(H, W)
        x = F.interpolate(x, size=(max(out, round(H * sc)), max(out, round(W * sc))),
                          mode="bilinear", antialias=True, align_corners=False)
        h, w = x.shape[-2:]
        t, l = (h - out) // 2, (w - out) // 2
        x = x[..., t:t + out, l:l + out]
    return (x - IMNET_MEAN) / IMNET_STD


class Backbones:
    def __init__(self):
        from transformers import VideoMAEForVideoClassification, VJEPA2ForVideoClassification, VJEPA2Model
        from ultralytics import YOLO
        h = torch.float16
        self.vmL = VideoMAEForVideoClassification.from_pretrained("MCG-NJU/videomae-large-finetuned-kinetics", dtype=h).to(DEV).eval()
        self.vmH = VideoMAEForVideoClassification.from_pretrained("MCG-NJU/videomae-huge-finetuned-kinetics", dtype=h).to(DEV).eval()
        self.vjL = VJEPA2ForVideoClassification.from_pretrained("facebook/vjepa2-vitl-fpc16-256-ssv2", dtype=h).to(DEV).eval()
        self.vjG = VJEPA2Model.from_pretrained("facebook/vjepa2-vitg-fpc64-256", dtype=h).to(DEV).eval()
        self.pose = YOLO(str(FEAT.parent / "weights" / "yolo11x-pose.pt"))

    @torch.no_grad()
    def vmae(self, m, x):
        z = m.videomae(x[None].half()).last_hidden_state.mean(1)
        return m.fc_norm(z) if m.fc_norm is not None else z

    @torch.no_grad()
    def vjl(self, x):
        return self.vjL.pooler(self.vjL.vjepa2(pixel_values_videos=x[None].half()).last_hidden_state)

    @torch.no_grad()
    def vjg(self, x):
        return self.vjG(pixel_values_videos=x[None].half()).last_hidden_state.mean(1)


SPECS = [  # name, rate, n_frames, hop_s, crop, short, fn
    ("vmaeL_r4", 4, 16, 0.5, 224, 224, lambda B, x: B.vmae(B.vmL, x)),
    ("vmaeL_r8", 8, 16, 0.5, 224, 224, lambda B, x: B.vmae(B.vmL, x)),
    ("vmaeL_r16", 16, 16, 0.5, 224, 224, lambda B, x: B.vmae(B.vmL, x)),
    ("vmaeH_r8", 8, 16, 0.5, 224, 224, lambda B, x: B.vmae(B.vmH, x)),
    ("vjL_ssv2_r8", 8, 16, 0.5, 256, 292, lambda B, x: B.vjl(x)),
    ("vjG_r16", 16, 64, 1.0, 256, 292, lambda B, x: B.vjg(x)),
]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--datasets", nargs="+", default=["esbd", "ssbd", "ext_ssbd", "ext_ssbdplus", "ext_wei2023"])
    args = ap.parse_args()
    B = Backbones()
    for name in args.datasets:
        od = OUT / name
        od.mkdir(parents=True, exist_ok=True)
        for r in tqdm(index_dataset(name), desc=name):
            dst = od / (r["clip"][:-4] + ".npz")
            if dst.exists():
                continue
            frames, src_fps = load_clip(r["path"], short=400)
            boxes = track_boxes(B.pose, frames)
            gpu = torch.from_numpy(frames).to(DEV)
            Wd = frames.shape[2]
            fboxes = boxes.copy()
            fboxes[:, [0, 2]] = Wd - boxes[:, [2, 0]]
            feats = {"boxes": boxes, "src_fps": src_fps}
            for spec, rate, n, hop, out, short, fn in SPECS:
                wins = window_index(len(frames), rate, n, hop)
                for view in ("full", "person"):
                    per_flip = []
                    for fr, bx in ((gpu, boxes), (gpu.flip(2), fboxes)):
                        z = [fn(B, crop_window(fr, w, bx if view == "person" else None, out, short)) for w in wins]
                        per_flip.append(torch.cat(z).float().cpu().numpy())
                    feats[f"{spec}_{view}"] = np.stack(per_flip)
            np.savez_compressed(dst, **feats)
            del gpu
            torch.cuda.empty_cache()


if __name__ == "__main__":
    main()
