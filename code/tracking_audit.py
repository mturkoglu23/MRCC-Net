"""Audit of the child tracker (results/final/tracking_audit.csv + contact sheets).

Re-runs the exact tracking rule of extract_features_v2.track_boxes and records, per clip,
the detection rate, how often more than one person is visible, and the tracked box.
Clips with a second person in >= 10% of frames get a 3-frame strip (tracked box green,
other people red) for manual inspection in results/figures/tracking_audit/.
"""
import cv2
import numpy as np
import pandas as pd

from common import FEAT, RESULTS, index_dataset
from extract_features import load_clip

OUT = RESULTS / "figures" / "tracking_audit"
OUT.mkdir(parents=True, exist_ok=True)


def main():
    from ultralytics import YOLO
    pose = YOLO(str(FEAT.parent / "weights" / "yolo11x-pose.pt"))
    rows, strips = [], []
    for ds in ["esbd", "ssbd"]:
        for r in index_dataset(ds):
            frames, _ = load_clip(r["path"], short=400)
            T, h, w = frames.shape[:3]
            dets, boxes, prev = [], np.full((T, 4), np.nan, np.float32), None
            for i in range(0, T, 32):
                for j, res in enumerate(pose(list(frames[i:i + 32][..., ::-1]), verbose=False, device=0)):
                    b = res.boxes.xyxy.cpu().numpy() if len(res.boxes) else np.zeros((0, 4))
                    dets.append(b)
                    if not len(b):
                        continue
                    c = (b[:, :2] + b[:, 2:]) / 2
                    area = (b[:, 2] - b[:, 0]) * (b[:, 3] - b[:, 1])
                    k = int(np.argmax(area)) if prev is None else int(np.argmin(
                        np.linalg.norm(c - prev, axis=1) / max(h, w) - 0.5 * area / (h * w)))
                    prev = c[k]
                    boxes[i + j] = b[k]
            n = np.array([len(b) for b in dets])
            ok = ~np.isnan(boxes[:, 0])
            cen = (boxes[ok, :2] + boxes[ok, 2:]) / 2
            jumps = (np.linalg.norm(np.diff(cen, axis=0), axis=1) / max(h, w) > 0.25).sum() if ok.sum() > 1 else 0
            multi = (n >= 2).mean()
            rows.append(dict(dataset=ds, clip=r["clip"], label=r["label"], frames=T, det_rate=ok.mean(),
                             multi_person_rate=multi, max_persons=int(n.max()), jumps=int(jumps)))
            if multi >= 0.10:
                tiles = []
                for q in (0.25, 0.5, 0.75):
                    t = min(int(q * T), T - 1)
                    im = frames[t][..., ::-1].copy()
                    for b in dets[t]:
                        cv2.rectangle(im, tuple(map(int, b[:2])), tuple(map(int, b[2:])), (0, 0, 255), 2)
                    if ok[t]:
                        cv2.rectangle(im, tuple(map(int, boxes[t, :2])), tuple(map(int, boxes[t, 2:])), (0, 255, 0), 3)
                    tiles.append(cv2.resize(im, (int(240 * w / h), 240)))
                strip = np.concatenate(tiles, 1)
                cv2.putText(strip, f"{len(strips)} {ds}/{r['clip']}", (5, 22), cv2.FONT_HERSHEY_SIMPLEX, 0.7,
                            (255, 255, 0), 2)
                strips.append(strip)
            print(ds, r["clip"], f"det={ok.mean():.2f} multi={multi:.2f} jumps={jumps}", flush=True)
    df = pd.DataFrame(rows)
    df.to_csv(RESULTS / "final" / "tracking_audit.csv", index=False)
    W = 1100
    strips = [cv2.resize(s, (W, int(s.shape[0] * W / s.shape[1]))) for s in strips]
    for k in range(0, len(strips), 6):
        cv2.imwrite(str(OUT / f"sheet{k // 6:02d}.jpg"), np.concatenate(strips[k:k + 6], 0),
                    [cv2.IMWRITE_JPEG_QUALITY, 80])
    print(df.groupby("dataset")[["det_rate", "multi_person_rate", "jumps"]].describe().T)
    print("clips with second person >=10% frames:", len(strips))


if __name__ == "__main__":
    main()
