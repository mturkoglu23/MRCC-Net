"""Extract frozen-backbone features for every clip.

All clips are first re-timed to a common 16 fps timeline (nearest-frame
resampling) so that heterogeneous source frame rates (9-30 fps) do not leak into
the representation.

Per clip we store (for original and horizontally flipped video):
  vmae  [W, 1024]  VideoMAE-L (K400) window embeddings, 16 frames @ 8 fps, hop 0.5 s
  xclip [W, 512]   X-CLIP-B/16 (K600) window video embeddings, same windows
  dino  [T, 768]   DINOv2-B CLS per frame @ 16 fps
  pose  [T, 17, 3] YOLO11-pose keypoints (x, y, conf) of the tracked child @ 16 fps
"""
import argparse
import os

import cv2
import numpy as np
import torch
import torch.nn.functional as F
from tqdm import tqdm

from common import FEAT, index_dataset

FPS = 16
WIN, WIN_STEP_FRAMES, HOP_S = 16, 2, 0.5  # 16 frames, every 2nd frame @16fps -> 8 fps, 2 s
DEV = "cuda"
IMNET_MEAN = torch.tensor([0.485, 0.456, 0.406], device=DEV).view(1, 3, 1, 1)
IMNET_STD = torch.tensor([0.229, 0.224, 0.225], device=DEV).view(1, 3, 1, 1)


def load_clip(path, short=320):
    cap = cv2.VideoCapture(path)
    src_fps = cap.get(cv2.CAP_PROP_FPS) or 30.0
    frames = []
    while True:
        ok, f = cap.read()
        if not ok:
            break
        h, w = f.shape[:2]
        s = short / min(h, w)
        if s < 1:
            f = cv2.resize(f, (round(w * s), round(h * s)), interpolation=cv2.INTER_AREA)
        frames.append(cv2.cvtColor(f, cv2.COLOR_BGR2RGB))
    cap.release()
    n = len(frames)
    dur = n / src_fps
    t = np.arange(0, dur, 1 / FPS)
    idx = np.minimum((t * src_fps).round().astype(int), n - 1)
    return np.stack([frames[i] for i in idx]), src_fps


def windows(T):
    span = WIN * WIN_STEP_FRAMES
    hop = int(HOP_S * FPS)
    starts = list(range(0, max(T - span, 0) + 1, hop)) or [0]
    out = []
    for s in starts:
        ix = np.arange(s, s + span, WIN_STEP_FRAMES)
        out.append(ix % T)  # loop-pad very short clips
    return out


class Extractors:
    def __init__(self):
        from transformers import AutoModel, VideoMAEForVideoClassification, XCLIPModel
        from ultralytics import YOLO
        self.vm = VideoMAEForVideoClassification.from_pretrained(
            "MCG-NJU/videomae-large-finetuned-kinetics", dtype=torch.float16).to(DEV).eval()
        self.xc = XCLIPModel.from_pretrained("microsoft/xclip-base-patch16-kinetics-600").to(DEV).eval()
        self.dn = AutoModel.from_pretrained("facebook/dinov2-base", dtype=torch.float16).to(DEV).eval()
        self.pose = YOLO(str(FEAT.parent / "weights" / "yolo11x-pose.pt"))

    @staticmethod
    @torch.no_grad()
    def prep(frames, short, crop=224):
        """GPU equivalent of the HF processors: resize shortest edge, centre
        crop, ImageNet normalisation. frames: uint8 [N,H,W,3] cuda tensor."""
        x = frames.permute(0, 3, 1, 2).float() / 255
        h, w = x.shape[-2:]
        s = short / min(h, w)
        x = F.interpolate(x, size=(max(crop, round(h * s)), max(crop, round(w * s))),
                          mode="bilinear", antialias=True, align_corners=False)
        h, w = x.shape[-2:]
        t, l = (h - crop) // 2, (w - crop) // 2
        x = x[..., t:t + crop, l:l + crop]
        return (x - IMNET_MEAN) / IMNET_STD

    @torch.no_grad()
    def videomae(self, frames, wins):
        x = self.prep(frames, 224)
        out = [self.vm.videomae(x[w][None].half()).last_hidden_state.mean(1) for w in wins]
        h = torch.cat(out)
        return (self.vm.fc_norm(h) if self.vm.fc_norm is not None else h).float().cpu().numpy()

    @torch.no_grad()
    def xclip(self, frames, wins):
        x = self.prep(frames, 224)
        out = []
        for w in wins:
            v = self.xc.get_video_features(pixel_values=x[w[::2]][None])  # X-CLIP-B/16 uses 8 frames
            out.append(v if torch.is_tensor(v) else v.pooler_output)
        return torch.cat(out).float().cpu().numpy()

    @torch.no_grad()
    def dino(self, frames, bs=64):
        x = self.prep(frames, 256)
        out = [self.dn(pixel_values=x[i:i + bs].half()).pooler_output for i in range(0, len(x), bs)]
        return torch.cat(out).float().cpu().numpy()

    def keypoints(self, frames):
        """Track the most salient person: start from the largest box, then follow
        the nearest box centre frame to frame."""
        T = len(frames)
        kp = np.zeros((T, 17, 3), np.float32)
        h, w = frames.shape[1:3]
        prev = None
        for i in range(0, T, 32):
            res = self.pose(list(frames[i:i + 32][..., ::-1]), verbose=False, device=0)
            for j, r in enumerate(res):
                if r.keypoints is None or len(r.boxes) == 0:
                    continue
                b = r.boxes.xyxy.cpu().numpy()
                c = np.stack([(b[:, 0] + b[:, 2]) / 2 / w, (b[:, 1] + b[:, 3]) / 2 / h], 1)
                area = (b[:, 2] - b[:, 0]) * (b[:, 3] - b[:, 1]) / (w * h)
                k = int(np.argmax(area)) if prev is None else int(np.argmin(
                    np.linalg.norm(c - prev, axis=1) - 0.5 * area))
                prev = c[k]
                xy = r.keypoints.xy[k].cpu().numpy()
                cf = r.keypoints.conf[k].cpu().numpy() if r.keypoints.conf is not None else np.ones(17)
                kp[i + j] = np.concatenate([xy / [w, h], cf[:, None]], 1)
        return kp


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--datasets", nargs="+", default=["esbd", "ssbd", "ext_ssbd", "ext_ssbdplus", "ext_wei2023"])
    args = ap.parse_args()
    ex = Extractors()
    for name in args.datasets:
        od = FEAT / name
        od.mkdir(parents=True, exist_ok=True)
        for r in tqdm(index_dataset(name), desc=name):
            dst = od / (r["clip"][:-4] + ".npz")
            if dst.exists():
                continue
            frames, src_fps = load_clip(r["path"])
            wins = windows(len(frames))
            feats = {"src_fps": src_fps, "n_frames": len(frames)}
            gpu = torch.from_numpy(frames).to(DEV)
            for tag, fr in [("", gpu), ("_flip", gpu.flip(2))]:
                feats["vmae" + tag] = ex.videomae(fr, wins)
                feats["xclip" + tag] = ex.xclip(fr, wins)
                feats["dino" + tag] = ex.dino(fr)
            feats["pose"] = ex.keypoints(frames)
            del gpu
            np.savez_compressed(dst, **feats)


if __name__ == "__main__":
    main()
