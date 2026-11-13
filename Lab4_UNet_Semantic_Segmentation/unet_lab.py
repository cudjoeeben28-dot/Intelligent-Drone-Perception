"""
Lab 4 — Semantic segmentation with U-Net on the Semantic Drone Dataset
=====================================================================

Covers every item of the Lab Task:
  Task 1  Data sanity check   (split, colour map == class names, sizes, 3 random pairs, label range)
  Task 2  Dataloader check    (one batch: shapes / dtypes, where NUM_CLASSES is set)
  Task 3  Training loop       (Adam lr=1e-3, 4–10 epochs, per-epoch metrics, 3 curves)
  Task 4  Qualitative results (load best checkpoint, 6 Image|GT|Pred triplets, per-class IoU + mIoU)

Plus the extra items from the lab sheet (UNET_New.docx):
  * architecture parts marked in code (base conv block, encoder, bottleneck, decoder, skips)
  * pixel accuracy and Dice score
  * ReduceLROnPlateau scheduler + early stopping (patience)
  * saving the model the standard PyTorch way (state_dict)
  * running inference only, without training (--mode infer)

Usage
-----
    python split.py                         # once: makes train/ and valid/ folders
    python unet_lab.py --mode train         # Tasks 1-4: trains, saves my_best_unet.pth, evaluates it
    python unet_lab.py --mode infer         # Tasks 1, 2, 4 only — NO training, uses provided best_unet.pth
    python unet_lab.py --mode infer --ckpt my_best_unet.pth   # evaluate your own trained model

Google Colab: see COLAB_GUIDE.md. If the raw dataset folder is missing but
outputs/cache_train_256x256.npz and outputs/cache_valid_256x256.npz exist,
the script runs from those (~105 MB) instead of the 4 GB of original images.

All figures are saved in ./outputs (and also shown on screen unless --no-show).
"""

import argparse
import random
import time
from pathlib import Path

import numpy as np
from PIL import Image
import matplotlib
import matplotlib.pyplot as plt

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms
from tqdm.auto import tqdm

# =====================================================================
# Config
# =====================================================================
parser = argparse.ArgumentParser(description="U-Net lab on the Semantic Drone Dataset")
parser.add_argument("--mode", choices=["train", "infer"], default="train",
                    help="train = Tasks 1-4; infer = skip training and evaluate a saved checkpoint")
parser.add_argument("--data", default="semantic_drone_dataset",
                    help="folder that contains training_set/")
parser.add_argument("--epochs", type=int, default=6, help="Task 3 asks for 4-10")
parser.add_argument("--lr", type=float, default=1e-3, help="Task 3: Adam(lr=1e-3)")
parser.add_argument("--batch", type=int, default=8)
parser.add_argument("--size", type=int, default=256, help="images are resized to size x size")
parser.add_argument("--ckpt", default=None,
                    help="checkpoint to save (train, default my_best_unet.pth) / "
                         "load (infer, default best_unet.pth = the provided pretrained model)")
parser.add_argument("--patience", type=int, default=4, help="early-stopping patience (epochs)")
parser.add_argument("--no-show", action="store_true", help="save figures only, don't open windows")
args = parser.parse_args()

if args.no_show:
    matplotlib.use("Agg")

h_in = w_in = args.size
EPOCHS = args.epochs
LR = args.lr
BATCH_SIZE = args.batch
VAL_BATCH_SIZE = 4
NUM_WORKERS = 0
SEED = 42

HERE = Path(__file__).resolve().parent
OUT_DIR = HERE / "outputs"
OUT_DIR.mkdir(exist_ok=True)
if args.ckpt is None:
    args.ckpt = "my_best_unet.pth" if args.mode == "train" else "best_unet.pth"
CKPT_PATH = Path(args.ckpt) if Path(args.ckpt).is_absolute() else HERE / args.ckpt

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
print("Device:", device)

# Folder layout produced by split.py
data_dir = Path(args.data) / "training_set"
train_dir = data_dir / "train" / "images"
val_dir = data_dir / "valid" / "images"
lab_dir = data_dir / "gt" / "semantic" / "label_images"
class_csv = data_dir / "gt" / "semantic" / "class_dict.csv"

# Reproducibility
random.seed(SEED)
np.random.seed(SEED)
torch.manual_seed(SEED)
if torch.cuda.is_available():
    torch.cuda.manual_seed_all(SEED)


def show_or_save(fig, name):
    fig.savefig(OUT_DIR / name, dpi=110, bbox_inches="tight")
    print(f"  saved figure -> outputs/{name}")
    if args.no_show:
        plt.close(fig)
    else:
        plt.show()


# =====================================================================
# Classes + colour palette
# ---------------------------------------------------------------------
# Task 1: "Make sure colour map matches the class_names".
# The original code had 22 names but 24 colours ("dirt" and "door" were
# missing), so every name after "paved-area" was attached to the wrong
# colour. The dataset's class_dict.csv has 24 classes, in this order:
# =====================================================================
CLASS_NAMES = [
    "unlabeled", "paved-area", "dirt", "grass", "gravel", "water", "rocks", "pool",
    "vegetation", "roof", "wall", "window", "door", "fence", "fence-pole", "person",
    "dog", "car", "bicycle", "tree", "bald-tree", "ar-marker", "obstacle", "conflicting",
]
CLASS_COLORS = [
    (0, 0, 0), (128, 64, 128), (130, 76, 0), (0, 102, 0), (112, 103, 87),
    (28, 42, 168), (48, 41, 30), (0, 50, 89), (107, 142, 35), (70, 70, 70),
    (102, 102, 156), (254, 228, 12), (254, 148, 12), (190, 153, 153), (153, 153, 153),
    (255, 22, 96), (102, 51, 0), (9, 143, 150), (119, 11, 32), (51, 51, 0),
    (190, 250, 190), (112, 150, 146), (2, 135, 115), (255, 0, 0),
]
assert len(CLASS_NAMES) == len(CLASS_COLORS), "every class name needs exactly one colour"

# Cross-check against the dataset's own class_dict.csv (if present)
if class_csv.exists():
    rows = [l.split(",") for l in class_csv.read_text().strip().splitlines()[1:]]
    csv_map = {r[0].strip(): tuple(int(v) for v in r[1:4]) for r in rows}
    ours = dict(zip(CLASS_NAMES, CLASS_COLORS))
    assert csv_map == ours, f"colour map mismatch with class_dict.csv:\n{csv_map}\nvs\n{ours}"
    print(f"[Task 1] Colour map matches class_dict.csv ({len(csv_map)} classes)")

CLASS2ID = {rgb: idx for idx, rgb in enumerate(CLASS_COLORS)}
PALETTE = CLASS_COLORS

# Task 2: "where to instantiate num_classes" -> here, derived from the class list.
# It is used by the model's last layer (UNet(NUM_CLASSES)) and by all metrics.
NUM_CLASSES = len(CLASS_NAMES)   # 24
IGNORE_INDEX = 255               # any pixel colour not in the palette

# =====================================================================
# Transforms & helpers
# =====================================================================
MEAN, STD = (0.485, 0.456, 0.406), (0.229, 0.224, 0.225)
normalize = transforms.Normalize(mean=MEAN, std=STD)
inv_norm = transforms.Normalize(mean=[-m / s for m, s in zip(MEAN, STD)], std=[1 / s for s in STD])


def rgb_mask_to_ids(rgb_np: np.ndarray, lut=CLASS2ID, unknown_val=IGNORE_INDEX):
    """(H,W,3) RGB mask -> (H,W) integer class IDs."""
    h, w, _ = rgb_np.shape
    flat = rgb_np.reshape(-1, 3)
    unique_colors, inverse = np.unique(flat, axis=0, return_inverse=True)
    mapped = np.array([lut.get(tuple(int(v) for v in c), unknown_val) for c in unique_colors],
                      dtype=np.uint8)
    return mapped[inverse.reshape(-1)].reshape(h, w)


def ids_to_color(mask_ids: np.ndarray, palette=PALETTE):
    """(H,W) class IDs -> (H,W,3) RGB image (the colorizer)."""
    out = np.zeros((*mask_ids.shape, 3), dtype=np.uint8)
    for cls_id, rgb in enumerate(palette):
        out[mask_ids == cls_id] = rgb
    return out


# =====================================================================
# Dataset
# ---------------------------------------------------------------------
# The originals are 6000x4000 px, so decoding them every epoch is slow.
# Each split is resized once and cached to outputs/cache_*.npz.
# =====================================================================
class SemanticDroneDataset(Dataset):
    def __init__(self, image_dir, label_dir, cache_name):
        image_dir, label_dir = Path(image_dir), Path(label_dir)
        cache = OUT_DIR / f"cache_{cache_name}_{h_in}x{w_in}.npz"
        if not image_dir.exists() and cache.exists():
            # Cache-only mode (e.g. Google Colab): the raw 4 GB dataset is not
            # needed, the pre-resized arrays in outputs/ are enough.
            d = np.load(cache)
            self.names = [str(n) for n in d["names"]]
            self.images, self.labels = d["images"], d["labels"]
            print(f"  loaded {cache_name} set from {cache.name} (cache-only mode)")
            return
        exts = {".jpg", ".jpeg", ".png", ".bmp", ".tif", ".tiff"}
        imgs = {p.stem: p for p in image_dir.iterdir() if p.suffix.lower() in exts}
        labs = {p.stem: p for p in label_dir.iterdir() if p.suffix.lower() in exts}
        common = sorted(set(imgs) & set(labs))
        if not common:
            raise RuntimeError(f"No matching image/mask names in {image_dir} and {label_dir}")
        self.names = common

        if cache.exists() and list(np.load(cache)["names"]) == common:
            d = np.load(cache)
            self.images, self.labels = d["images"], d["labels"]
        else:
            self.images = np.zeros((len(common), h_in, w_in, 3), np.uint8)
            self.labels = np.zeros((len(common), h_in, w_in), np.uint8)
            for i, s in enumerate(tqdm(common, desc=f"Resizing {cache_name} set (one-off)")):
                img = Image.open(imgs[s]).convert("RGB")
                img.draft("RGB", (w_in * 2, h_in * 2))  # fast JPEG downscale on decode
                self.images[i] = np.array(img.resize((w_in, h_in), Image.BILINEAR))
                lab = Image.open(labs[s]).convert("RGB").resize((w_in, h_in), Image.NEAREST)
                self.labels[i] = rgb_mask_to_ids(np.array(lab))
            np.savez(cache, images=self.images, labels=self.labels, names=np.array(common))

    def __len__(self):
        return len(self.names)

    def __getitem__(self, idx):
        image = normalize(transforms.functional.to_tensor(self.images[idx]))  # float32 (3,H,W)
        label = torch.from_numpy(self.labels[idx].astype(np.int64))           # int64  (H,W)
        return image, label


# =====================================================================
# U-Net model
# =====================================================================
class UNet(nn.Module):
    def __init__(self, num_classes):
        super().__init__()
        # ---- ENCODER (left arm of the U) ----------------------------
        # 4 conv blocks; after each one a 2x2 max-pool halves H,W
        # while channels double: 3 -> 64 -> 128 -> 256 -> 512
        self.enc = nn.ModuleList([
            self.conv_block(3, 64), self.conv_block(64, 128),
            self.conv_block(128, 256), self.conv_block(256, 512),
        ])
        self.pool = nn.MaxPool2d(2)

        # ---- BOTTLENECK (bottom of the U) ---------------------------
        # smallest / deepest map: 1024 channels at H/16 x W/16 (16x16 for 256 input)
        self.mid = self.conv_block(512, 1024)

        # ---- DECODER (right arm of the U) ---------------------------
        # Transpose convolution (kernel 2, stride 2): every input pixel is
        # multiplied by a learned 2x2 kernel and "painted" into a 2x2 output
        # patch, so H,W double while channels halve. It is a *learned*
        # upsampling, unlike fixed bilinear interpolation.
        self.up = nn.ModuleList([
            nn.ConvTranspose2d(1024, 512, 2, 2),
            nn.ConvTranspose2d(512, 256, 2, 2),
            nn.ConvTranspose2d(256, 128, 2, 2),
            nn.ConvTranspose2d(128, 64, 2, 2),
        ])
        # input channels are doubled because the skip connection is concatenated
        self.dec = nn.ModuleList([
            self.conv_block(1024, 512),
            self.conv_block(512, 256),
            self.conv_block(256, 128),
            self.conv_block(128, 64),
        ])
        # ---- OUTPUT: 1x1 conv -> one score map (logit) per class ----
        self.out = nn.Conv2d(64, num_classes, 1)

    # ---- BASE CONVOLUTION BLOCK --------------------------------------
    # two x (3x3 Conv -> ReLU -> BatchNorm). (The lab sheet lists Conv->BN->ReLU;
    # this order is kept so the provided best_unet.pth loads — both work.)
    @staticmethod
    def conv_block(in_ch, out_ch):
        return nn.Sequential(
            nn.Conv2d(in_ch, out_ch, 3, 1, 1), nn.ReLU(inplace=True), nn.BatchNorm2d(out_ch),
            nn.Conv2d(out_ch, out_ch, 3, 1, 1), nn.ReLU(inplace=True), nn.BatchNorm2d(out_ch),
        )

    def forward(self, x):
        skips = []
        for enc_layer in self.enc:          # encoder
            x = enc_layer(x)
            skips.append(x)                 # keep for the SKIP CONNECTION
            x = self.pool(x)
        x = self.mid(x)                     # bottleneck
        for i in range(4):                  # decoder
            x = self.up[i](x)               # upsample x2
            x = torch.cat([x, skips[-(i + 1)]], dim=1)  # skip: "what" + "where"
            x = self.dec[i](x)
        return self.out(x)                  # (B, NUM_CLASSES, H, W)


# =====================================================================
# Metrics: confusion matrix -> pixel accuracy, IoU / mIoU, Dice
# =====================================================================
@torch.no_grad()
def compute_confusion_matrix(pred, target, num_classes, ignore_index=IGNORE_INDEX):
    mask = target != ignore_index
    k = num_classes * target[mask] + pred[mask]
    return torch.bincount(k, minlength=num_classes ** 2).reshape(num_classes, num_classes)


def iou_from_confmat(cm):
    """Per-class IoU = TP / (TP + FP + FN); mIoU over classes that appear."""
    tp = torch.diag(cm).float()
    fp = cm.sum(0).float() - tp
    fn = cm.sum(1).float() - tp
    denom = tp + fp + fn
    iou = torch.where(denom > 0, tp / denom.clamp_min(1), torch.zeros_like(tp))
    return iou, iou[denom > 0].mean()


def dice_from_confmat(cm):
    """Per-class Dice = 2*TP / (2*TP + FP + FN) = 2|P∩Y| / (|P|+|Y|), averaged.
    Multi-class version of: dice += (2*(preds*y).sum()) / ((preds+y).sum() + 1e-8)"""
    tp = torch.diag(cm).float()
    denom = cm.sum(0).float() + cm.sum(1).float()     # |P| + |Y|
    dice = 2 * tp / (denom + 1e-8)
    return dice[denom > 0].mean()


def pixel_accuracy(cm):
    """Accuracy = correctly classified pixels / total (non-ignored) pixels."""
    return torch.diag(cm).sum().float() / cm.sum().clamp_min(1)


@torch.no_grad()
def evaluate(model, loader, criterion, num_classes):
    """VALIDATION PART: no gradients, model.eval() (BatchNorm uses running stats)."""
    model.eval()
    total_loss = 0.0
    cm = torch.zeros((num_classes, num_classes), dtype=torch.long, device=device)
    for X, Y in loader:
        X, Y = X.to(device), Y.to(device)
        logits = model(X)
        total_loss += criterion(logits, Y).item()
        cm += compute_confusion_matrix(logits.argmax(1), Y, num_classes)
    _, miou = iou_from_confmat(cm)
    return total_loss / len(loader), pixel_accuracy(cm).item(), miou.item(), dice_from_confmat(cm).item(), cm


# =====================================================================
# TASK 1 — Data sanity check
# =====================================================================
print("\n=== Task 1: data sanity check ===")
has_cache = all((OUT_DIR / f"cache_{n}_{h_in}x{w_in}.npz").exists() for n in ("train", "valid"))
if not has_cache and (not train_dir.exists() or not val_dir.exists()):
    raise SystemExit(f"Missing {train_dir} / {val_dir}. Run `python split.py` first.")

train_ds = SemanticDroneDataset(train_dir, lab_dir, "train")
val_ds = SemanticDroneDataset(val_dir, lab_dir, "valid")
print(f"[Task 1] Train: {len(train_ds)} images, Val: {len(val_ds)} images")

# Label range check: must be in [0, NUM_CLASSES-1] or IGNORE_INDEX
all_ids = np.unique(np.concatenate([np.unique(train_ds.labels), np.unique(val_ds.labels)]))
bad = [int(v) for v in all_ids if not (0 <= v < NUM_CLASSES or v == IGNORE_INDEX)]
print(f"[Task 1] Label IDs present: {all_ids.tolist()}")
print(f"[Task 1] Valid range [0, {NUM_CLASSES - 1}] (+{IGNORE_INDEX} ignore): {'OK' if not bad else f'BAD {bad}'}")
assert not bad

# 3 random (image, mask) pairs: mask as grayscale IDs and colorized
fig, axs = plt.subplots(3, 3, figsize=(12, 12))
for row, idx in enumerate(random.sample(range(len(train_ds)), 3)):
    img, lab = train_ds[idx]
    axs[row, 0].imshow(inv_norm(img).permute(1, 2, 0).clamp(0, 1).numpy())
    axs[row, 0].set_title(f"Image {train_ds.names[idx]}")
    axs[row, 1].imshow(lab.numpy(), cmap="gray", vmin=0, vmax=NUM_CLASSES - 1)
    axs[row, 1].set_title("Mask (class IDs, grayscale)")
    axs[row, 2].imshow(ids_to_color(lab.numpy()))
    axs[row, 2].set_title("Mask (colorized)")
    for a in axs[row]:
        a.axis("off")
fig.tight_layout()
show_or_save(fig, "task1_samples.png")

# =====================================================================
# TASK 2 — Dataloader correctness
# =====================================================================
print("\n=== Task 2: dataloader check ===")
train_loader = DataLoader(train_ds, BATCH_SIZE, shuffle=True, num_workers=NUM_WORKERS)
val_loader = DataLoader(val_ds, VAL_BATCH_SIZE, shuffle=False, num_workers=NUM_WORKERS)
Xb, Yb = next(iter(train_loader))
print(f"[Task 2] train_loader: {len(train_loader)} batches of {BATCH_SIZE}; "
      f"val_loader: {len(val_loader)} batches of {VAL_BATCH_SIZE}")
print(f"[Task 2] X: shape={tuple(Xb.shape)} dtype={Xb.dtype}  (B, C, H, W), normalised floats")
print(f"[Task 2] Y: shape={tuple(Yb.shape)} dtype={Yb.dtype}  (B, H, W), integer class IDs")
print(f"[Task 2] NUM_CLASSES={NUM_CLASSES} (set from CLASS_NAMES; used by UNet's last layer + metrics)")

criterion = nn.CrossEntropyLoss(ignore_index=IGNORE_INDEX)

# =====================================================================
# TASK 3 — Training loop with metrics   (skipped in --mode infer)
# =====================================================================
if args.mode == "train":
    print(f"\n=== Task 3: training {EPOCHS} epochs, Adam lr={LR} ===")
    model = UNet(NUM_CLASSES).to(device)
    optimizer = optim.Adam(model.parameters(), lr=LR)
    # LR scheduling: if val loss hasn't improved for 2 epochs, LR *= 0.1
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode="min", patience=2, factor=0.1)

    keys = ["train_loss", "val_loss", "train_pix_acc", "val_pix_acc",
            "train_mIoU", "val_mIoU", "train_dice", "val_dice"]
    hist = {k: [] for k in keys}
    best_mIoU, epochs_no_improve = -1.0, 0

    for epoch in range(1, EPOCHS + 1):
        t0 = time.time()
        # ---- TRAINING PART ----
        model.train()
        run_loss = 0.0
        cm_train = torch.zeros((NUM_CLASSES, NUM_CLASSES), dtype=torch.long, device=device)
        for X, Y in tqdm(train_loader, desc=f"Epoch {epoch}/{EPOCHS}", leave=False):
            X, Y = X.to(device), Y.to(device)
            optimizer.zero_grad()
            out = model(X)
            loss = criterion(out, Y)
            loss.backward()
            optimizer.step()
            run_loss += loss.item()
            cm_train += compute_confusion_matrix(out.detach().argmax(1), Y, NUM_CLASSES)
        train_loss = run_loss / len(train_loader)
        train_pix = pixel_accuracy(cm_train).item()
        train_mIoU = iou_from_confmat(cm_train)[1].item()
        train_dice = dice_from_confmat(cm_train).item()

        # ---- VALIDATION PART ----
        val_loss, val_pix, val_mIoU, val_dice, _ = evaluate(model, val_loader, criterion, NUM_CLASSES)
        scheduler.step(val_loss)

        for k, v in zip(keys, [train_loss, val_loss, train_pix, val_pix,
                               train_mIoU, val_mIoU, train_dice, val_dice]):
            hist[k].append(v)
        print(f"Epoch {epoch:2d} | train loss {train_loss:.3f} acc {train_pix:.3f} mIoU {train_mIoU:.3f} "
              f"dice {train_dice:.3f} | val loss {val_loss:.3f} acc {val_pix:.3f} mIoU {val_mIoU:.3f} "
              f"dice {val_dice:.3f} | lr {optimizer.param_groups[0]['lr']:.1e} | {time.time() - t0:.0f}s")

        # ---- SAVE BEST MODEL (standard PyTorch: save the state_dict) ----
        # Monitor validation mIoU to decide which weights to keep.
        if val_mIoU > best_mIoU:
            best_mIoU, epochs_no_improve = val_mIoU, 0
            torch.save(model.state_dict(), CKPT_PATH)
            print(f"  -> new best val mIoU {best_mIoU:.3f}, saved {CKPT_PATH.name}")
        else:
            epochs_no_improve += 1
            # ---- EARLY STOPPING ----
            if epochs_no_improve >= args.patience:
                print(f"Early stopping: no val mIoU improvement for {args.patience} epochs")
                break

    # 3 curves: Loss, Pixel Acc, mIoU (train vs val)
    ep = np.arange(1, len(hist["train_loss"]) + 1)
    fig, axs = plt.subplots(1, 3, figsize=(15, 4))
    for a, (k, title) in zip(axs, [("loss", "Loss"), ("pix_acc", "Pixel accuracy"), ("mIoU", "mIoU")]):
        a.plot(ep, hist[f"train_{k}"], "o-", label="train")
        a.plot(ep, hist[f"val_{k}"], "o-", label="val")
        a.set_title(title); a.set_xlabel("epoch"); a.legend(); a.grid(alpha=0.3)
    fig.tight_layout()
    show_or_save(fig, "task3_curves.png")
else:
    print("\n=== Task 3 skipped (--mode infer): no training, loading saved weights ===")

# =====================================================================
# TASK 4 — Qualitative results with the best checkpoint (INFERENCE)
# ---------------------------------------------------------------------
# Inference = build the same architecture, load the state_dict, call
# model.eval() and run under torch.no_grad(). No optimizer, no backward.
# =====================================================================
print(f"\n=== Task 4: evaluating checkpoint {CKPT_PATH.name} ===")
best_model = UNet(NUM_CLASSES).to(device)
best_model.load_state_dict(torch.load(CKPT_PATH, map_location=device))
best_model.eval()

val_loss, val_pix, val_mIoU, val_dice, cm = evaluate(best_model, val_loader, criterion, NUM_CLASSES)
print(f"[Task 4] val loss={val_loss:.4f}  pixel acc={val_pix:.4f}  mIoU={val_mIoU:.4f}  Dice={val_dice:.4f}")

# 6 validation triplets: Image | Ground truth | Prediction
fig, axs = plt.subplots(6, 3, figsize=(12, 22))
with torch.no_grad():
    for row in range(6):
        X, Y = val_ds[row]
        pred = best_model(X.unsqueeze(0).to(device)).argmax(1)[0].cpu().numpy()
        axs[row, 0].imshow(inv_norm(X).permute(1, 2, 0).clamp(0, 1).numpy())
        axs[row, 1].imshow(ids_to_color(Y.numpy()))
        axs[row, 2].imshow(ids_to_color(pred))
        for a, t in zip(axs[row], [f"Image {val_ds.names[row]}", "Ground truth", "Prediction"]):
            a.set_title(t); a.axis("off")
fig.tight_layout()
show_or_save(fig, "task4_predictions.png")


# Per-class IoU
@torch.no_grad()
def per_class_iou(model, loader, num_classes):
    model.eval()
    cm = torch.zeros((num_classes, num_classes), dtype=torch.long, device=device)
    for X, Y in loader:
        X, Y = X.to(device), Y.to(device)
        cm += compute_confusion_matrix(model(X).argmax(1), Y, num_classes)
    iou, miou = iou_from_confmat(cm)
    present = (cm.sum(0) + cm.sum(1)) > 0
    return iou.cpu().numpy(), present.cpu().numpy(), miou.item()


ious, present, miou = per_class_iou(best_model, val_loader, NUM_CLASSES)
lines = ["Per-class IoU (validation set)", "-" * 34]
for cls_id, (name, iou) in enumerate(zip(CLASS_NAMES, ious)):
    lines.append(f"{cls_id:2d} {name:>12s}: {iou:.4f}" + ("" if present[cls_id] else "  (not present)"))
lines += ["-" * 34, f"Mean IoU: {miou:.4f}", f"Pixel accuracy: {val_pix:.4f}", f"Dice: {val_dice:.4f}"]
report = "\n".join(lines)
print("\n" + report)
(OUT_DIR / "task4_per_class_iou.txt").write_text(report)
print("  saved -> outputs/task4_per_class_iou.txt")
