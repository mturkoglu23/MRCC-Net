"""Feature loading, pose normalisation and temporal cropping."""
import numpy as np
import torch

from common import FEAT, index_dataset

FPS = 16
CROP = 64          # frames per crop (4 s @ 16 fps)
WIN_HOP = 8        # frames between window starts (0.5 s)
WIN_SPAN = 32      # frames covered by one window (2 s)

# COCO-17 left/right pairs for horizontal flipping
FLIP_PERM = [0, 2, 1, 4, 3, 6, 5, 8, 7, 10, 9, 12, 11, 14, 13, 16, 15]
# joint groups for the spectral descriptor
GROUPS = {"head": [0, 1, 2, 3, 4], "shoulder": [5, 6], "elbow": [7, 8],
          "wrist": [9, 10], "hip": [11, 12]}


def normalise_pose(kp, thr=0.3):
    """kp [T,17,3] in image-normalised coords -> body-centred, torso-scaled [T,17,2].
    Low-confidence joints are linearly interpolated over time."""
    xy = kp[..., :2].copy()
    ok = kp[..., 2] > thr
    T = len(xy)
    t = np.arange(T)
    for j in range(17):
        if ok[:, j].sum() >= 2:
            for d in range(2):
                xy[:, j, d] = np.interp(t, t[ok[:, j]], xy[ok[:, j], j, d])
        else:
            xy[:, j] = np.nan
    centre = np.nanmean(xy[:, [5, 6, 11, 12]], axis=1, keepdims=True)
    centre = np.where(np.isnan(centre), np.nanmean(xy, axis=1, keepdims=True), centre)
    shoulder = np.linalg.norm(xy[:, 5] - xy[:, 6], axis=-1)
    torso = np.linalg.norm(np.nanmean(xy[:, [5, 6]], 1) - np.nanmean(xy[:, [11, 12]], 1), axis=-1)
    scale = np.nanmedian(np.concatenate([shoulder * 2, torso]))
    if not np.isfinite(scale) or scale < 1e-3:
        scale = 0.3
    out = (xy - centre) / scale
    out = np.nan_to_num(out, nan=0.0)
    return np.clip(out, -5, 5).astype(np.float32), ok.mean(0).astype(np.float32)


def flip_pose(p):
    p = p[:, FLIP_PERM].copy()
    p[..., 0] *= -1
    return p


class Clip:
    def __init__(self, row):
        z = np.load(FEAT / row["dataset"] / (row["clip"][:-4] + ".npz"))
        self.label, self.group, self.clip = row["label"], row["group"], row["clip"]
        self.vmae = np.stack([z["vmae"], z["vmae_flip"]])
        self.xclip = np.stack([z["xclip"], z["xclip_flip"]])
        self.dino = np.stack([z["dino"], z["dino_flip"]])
        pose, self.pose_vis = normalise_pose(z["pose"])
        self.pose = np.stack([pose, flip_pose(pose)])
        self.T = self.dino.shape[1]
        self.src_fps = float(z["src_fps"])

    def crop(self, start, flip):
        """Return one CROP-frame view starting at frame `start` (loop-padded)."""
        fi = (start + np.arange(CROP)) % self.T
        W = self.vmae.shape[1]
        k0 = int(np.ceil(start / WIN_HOP))
        k1 = int(np.floor((start + CROP - WIN_SPAN) / WIN_HOP))
        wi = np.arange(k0, max(k1, k0) + 1) % W
        return {"vmae": self.vmae[flip, wi], "xclip": self.xclip[flip, wi],
                "dino": self.dino[flip, fi], "pose": self.pose[flip, fi]}

    def test_starts(self, stride=16):
        if self.T <= CROP:
            return [0]
        return list(range(0, self.T - CROP + 1, stride))


def load_dataset(name):
    return [Clip(r) for r in index_dataset(name)]


def collate(views):
    """Pad window sequences (variable length) and stack frame-level tensors."""
    Wmax = max(v["vmae"].shape[0] for v in views)
    B = len(views)
    vm = torch.zeros(B, Wmax, views[0]["vmae"].shape[-1])
    xc = torch.zeros(B, Wmax, views[0]["xclip"].shape[-1])
    mask = torch.ones(B, Wmax, dtype=torch.bool)  # True = padding
    for i, v in enumerate(views):
        n = v["vmae"].shape[0]
        vm[i, :n] = torch.from_numpy(v["vmae"])
        xc[i, :n] = torch.from_numpy(v["xclip"])
        mask[i, :n] = False
    dino = torch.from_numpy(np.stack([v["dino"] for v in views]))
    pose = torch.from_numpy(np.stack([v["pose"] for v in views]))
    return {"vmae": vm, "xclip": xc, "mask": mask, "dino": dino, "pose": pose}
