"""
Per-class IoU vs. class frequency chart (Lab 4 report figure).

Reads:
  outputs/cache_train_256x256.npz   -> pixel share of each class in the training masks
  outputs/task4_per_class_iou.txt   -> per-class validation IoU written by unet_lab.py (Task 4)
Writes:
  outputs/iou_vs_class_share.png

Run after unet_lab.py (locally or in Colab, from the lab folder):
    python plot_iou_chart.py
    python plot_iou_chart.py --iou results/per_class_iou_trained.txt --out results/iou_vs_class_share.png
"""
import argparse
import re
from pathlib import Path

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

HERE = Path(__file__).resolve().parent
OUT = HERE / "outputs"

ap = argparse.ArgumentParser()
ap.add_argument("--iou", default=str(OUT / "task4_per_class_iou.txt"), help="per-class IoU report from unet_lab.py")
ap.add_argument("--out", default=str(OUT / "iou_vs_class_share.png"), help="where to save the chart")
args = ap.parse_args()

CLASS_NAMES = [
    "unlabeled", "paved-area", "dirt", "grass", "gravel", "water", "rocks", "pool",
    "vegetation", "roof", "wall", "window", "door", "fence", "fence-pole", "person",
    "dog", "car", "bicycle", "tree", "bald-tree", "ar-marker", "obstacle", "conflicting",
]

# 1) Pixel share (%) of each class in the training masks
labels = np.load(OUT / "cache_train_256x256.npz")["labels"]
counts = np.bincount(labels.ravel(), minlength=256)[:len(CLASS_NAMES)]
share = counts / counts.sum() * 100

# 2) Per-class IoU from the Task 4 report, e.g. " 3        grass: 0.8126"
iou = {}
for line in Path(args.iou).read_text().splitlines():
    m = re.match(r"\s*(\d+)\s+\S+:\s+([\d.]+)(.*)", line)
    if m and "not present" not in m.group(3):
        iou[int(m.group(1))] = float(m.group(2))

# 3) Classes present, sorted by pixel share (largest first)
ids = sorted(iou, key=lambda c: -share[c])
names = [CLASS_NAMES[c] for c in ids]
shares = np.array([share[c] for c in ids])
ious = np.array([iou[c] for c in ids])
miou = ious.mean()
big, small = ious[shares >= 5], ious[shares < 1]

# 4) Plot: horizontal bars, blue = classes with >= 5% of pixels, dashed line = mIoU
fig, ax = plt.subplots(figsize=(9, 8.5))
y = np.arange(len(ids))
colors = ["#2a78d6" if s >= 5 else "#c4c3b8" for s in shares]
ax.barh(y, ious, color=colors, height=0.65)
ax.invert_yaxis()
ax.set_yticks(y, [f"{n}   {s:.2f}%" if s < 0.1 else f"{n}   {s:.1f}%" for n, s in zip(names, shares)])
ax.set_xlim(0, 1)
ax.set_xlabel("Validation IoU")

for yi, v in zip(y, ious):                       # value labels, nudged clear of the mIoU line
    x = v + 0.01
    if x < miou + 0.01 and v + 0.06 > miou:
        x = miou + 0.012
    ax.text(x, yi, f"{v:.2f}", va="center", fontsize=9)

ax.axvline(miou, color="grey", linestyle="--", linewidth=1)
ax.text(miou + 0.01, len(ids) - 0.2, f"mIoU {miou:.3f}", color="grey", fontsize=9)

for side in ("top", "right"):
    ax.spines[side].set_visible(False)
fig.suptitle(f"IoU follows class frequency: large classes average {big.mean():.2f}, "
             f"rare classes {small.mean():.2f}", x=0.02, ha="left", fontsize=13)
fig.text(0.02, 0.935, f"{len(big)} classes with ≥5% of training pixels average IoU {big.mean():.2f}; "
         f"{len(small)} classes under 1% average {small.mean():.2f}. Row labels: class, pixel share.",
         fontsize=9, color="dimgray")
fig.text(0.02, 0.01, "Blue = classes covering at least 5% of training pixels. "
         "Rows sorted by pixel share; absent classes omitted.", fontsize=8, color="dimgray")
fig.tight_layout(rect=(0, 0.03, 1, 0.92))

out_png = Path(args.out)
fig.savefig(out_png, dpi=150)
print(f"mIoU={miou:.4f}  large-class mean={big.mean():.3f}  rare-class mean={small.mean():.3f}")
print(f"saved -> {out_png}")
