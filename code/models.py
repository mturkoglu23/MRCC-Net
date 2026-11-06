"""RePAF-Net: Repetition-aware Pose-Appearance Fusion Network.

Three branches over frozen-backbone features of a 4 s crop:
  * Appearance: VideoMAE-L + X-CLIP window tokens -> temporal Transformer.
  * Repetition: temporal self-similarity matrices (TSM) of DINOv2 frame
    embeddings and of normalised pose, as a 2-channel image -> small CNN.
    Repetitive stereotypies appear as periodic off-diagonal stripes in the TSM,
    independent of what the child looks like.
  * Kinematic spectrum: log power spectrum (0.25-8 Hz) of joint-group
    velocities -> MLP. Flapping, banging and spinning have distinct dominant
    frequencies and body parts.
Branch embeddings are fused by an input-conditioned gate (softmax weights over
branches) and classified. Modality dropout regularises the fusion.
"""
import torch
import torch.nn as nn
import torch.nn.functional as F

from data import CROP, GROUPS

D = 256


def tsm(x):
    """x [B,T,C] -> cosine self-similarity [B,T,T]."""
    x = F.normalize(x - x.mean(1, keepdim=True), dim=-1)
    return x @ x.transpose(1, 2)


def pose_spectrum(pose):
    """pose [B,T,17,2] -> [B, groups*2, T//2] log power of velocity spectra."""
    vel = pose[:, 1:] - pose[:, :-1]
    feats = []
    for idx in GROUPS.values():
        g = vel[:, :, idx].mean(2)                       # [B,T-1,2]
        g = g - g.mean(1, keepdim=True)
        win = torch.hann_window(g.shape[1], device=g.device)[None, :, None]
        p = torch.fft.rfft(g * win, dim=1).abs().pow(2)  # [B,F,2]
        feats.append(p[:, 1:].transpose(1, 2))           # drop DC
    s = torch.cat(feats, 1)
    s = s / (s.sum(-1, keepdim=True) + 1e-6)             # relative spectrum per channel
    return torch.log(s + 1e-4)


class AppearanceBranch(nn.Module):
    def __init__(self, d_vm=1024, d_xc=512, drop=0.3):
        super().__init__()
        self.proj = nn.Sequential(nn.LayerNorm(d_vm + d_xc), nn.Dropout(drop), nn.Linear(d_vm + d_xc, D))
        self.cls = nn.Parameter(torch.zeros(1, 1, D))
        self.pos = nn.Parameter(torch.zeros(1, 32, D))
        layer = nn.TransformerEncoderLayer(D, 4, 2 * D, drop, batch_first=True, norm_first=True)
        self.enc = nn.TransformerEncoder(layer, 2, enable_nested_tensor=False)
        nn.init.trunc_normal_(self.pos, std=0.02)

    def forward(self, vm, xc, mask):
        x = self.proj(torch.cat([vm, xc], -1))
        x = torch.cat([self.cls.expand(len(x), -1, -1), x], 1) + self.pos[:, : x.shape[1] + 1]
        m = torch.cat([torch.zeros_like(mask[:, :1]), mask], 1)
        return self.enc(x, src_key_padding_mask=m)[:, 0]


class RepetitionBranch(nn.Module):
    def __init__(self, drop=0.3):
        super().__init__()
        def block(i, o):
            return nn.Sequential(nn.Conv2d(i, o, 3, padding=1), nn.BatchNorm2d(o), nn.GELU(), nn.MaxPool2d(2))
        self.cnn = nn.Sequential(block(2, 32), block(32, 64), block(64, 128), block(128, 128),
                                 nn.AdaptiveAvgPool2d(1), nn.Flatten(), nn.Dropout(drop), nn.Linear(128, D))

    def forward(self, dino, pose):
        s = torch.stack([tsm(dino), tsm(pose.flatten(2))], 1)
        return self.cnn(s)


class SpectrumBranch(nn.Module):
    def __init__(self, n_ch=2 * len(GROUPS), n_f=(CROP - 1) // 2, drop=0.3):
        super().__init__()
        self.net = nn.Sequential(nn.Conv1d(n_ch, 64, 3, padding=1), nn.GELU(),
                                 nn.Conv1d(64, 64, 3, padding=1), nn.GELU(),
                                 nn.Flatten(), nn.Dropout(drop), nn.Linear(64 * n_f, D))

    def forward(self, pose):
        return self.net(pose_spectrum(pose))


class RePAFNet(nn.Module):
    BRANCHES = ("app", "rep", "spec")

    def __init__(self, n_cls, branches=BRANCHES, gate=True, drop=0.3, mod_drop=0.2):
        super().__init__()
        self.branches, self.gate_on, self.mod_drop = branches, gate, mod_drop
        self.app = AppearanceBranch(drop=drop) if "app" in branches else None
        self.rep = RepetitionBranch(drop) if "rep" in branches else None
        self.spec = SpectrumBranch(drop=drop) if "spec" in branches else None
        nb = len(branches)
        self.gate = nn.Sequential(nn.Linear(nb * D, 128), nn.GELU(), nn.Linear(128, nb)) if gate and nb > 1 else None
        self.head = nn.Sequential(nn.LayerNorm(D if self.gate is not None or nb == 1 else nb * D),
                                  nn.Dropout(drop),
                                  nn.Linear(D if self.gate is not None or nb == 1 else nb * D, n_cls))

    def forward(self, b, return_gate=False):
        z = []
        if self.app is not None:
            z.append(self.app(b["vmae"], b["xclip"], b["mask"]))
        if self.rep is not None:
            z.append(self.rep(b["dino"], b["pose"]))
        if self.spec is not None:
            z.append(self.spec(b["pose"]))
        z = torch.stack(z, 1)                                   # [B,nb,D]
        if self.training and self.mod_drop > 0 and z.shape[1] > 1:
            keep = (torch.rand(z.shape[:2], device=z.device) > self.mod_drop).float()
            keep[keep.sum(1) == 0, 0] = 1
            z = z * keep[..., None]
        if self.gate is not None:
            w = self.gate(z.flatten(1)).softmax(-1)
            h = (w[..., None] * z).sum(1)
        else:
            w = None
            h = z.flatten(1) if z.shape[1] > 1 else z[:, 0]
        out = self.head(h)
        return (out, w) if return_gate else out
