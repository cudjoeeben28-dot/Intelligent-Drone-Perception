

"""
Semantic Drone Dataset — U-Net Training Script
Implements Tasks 1–4 (sanity check, dataloader, training, metrics, qualitative results).
"""

import os
from pathlib import Path
import random
from typing import Dict, Tuple

import numpy as np
from PIL import Image
import matplotlib.pyplot as plt

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms
from tqdm.auto import tqdm

# ----------------------------
# Config
# ----------------------------
h_in = 256
w_in = 256
EPOCHS = 20
LR = 4e-5
BATCH_SIZE = 8
VAL_BATCH_SIZE = 4
NUM_WORKERS = 0
SEED = 42

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
print("Device:", device)

data_dir = Path(r"D:\semantic_drone_dataset\training_set")
train_dir = data_dir / "train" / "images"
val_dir   = data_dir / "valid" / "images"
lab_dir   = data_dir / "masks"

# ----------------------------
# Reproducibility
# ----------------------------
random.seed(SEED)
np.random.seed(SEED)
torch.manual_seed(SEED)
if torch.cuda.is_available():
    torch.cuda.manual_seed_all(SEED)

# ----------------------------
# Classes + Palette
# ----------------------------
CLASS_NAMES = [
    "unlabeled", "paved-area", "grass", "gravel", "water", "rocks", "pool",
    "vegetation", "roof", "wall", "window", "fence", "fence-pole", "person",
    "dog", "car", "bicycle", "tree", "bald-tree", "ar-marker", "obstacle", "conflicting"
]

CLASS_COLORS = [
    (0, 0, 0), (128, 64, 128), (130, 76, 0), (0, 102, 0), (112, 103, 87),
    (28, 42, 168), (48, 41, 30), (0, 50, 89), (107, 142, 35), (70, 70, 70),
    (102, 102, 156), (254, 228, 12), (254, 148, 12), (190, 153, 153), (153, 153, 153),
    (255, 22, 96), (102, 51, 0), (9, 143, 150), (119, 11, 32), (51, 51, 0),
    (190, 250, 190), (112, 150, 146), (2, 135, 115), (255, 0, 0)
]

CLASS2ID = {rgb: idx for idx, rgb in enumerate(CLASS_COLORS)}
PALETTE = CLASS_COLORS
NUM_CLASSES = len(CLASS_NAMES)
IGNORE_INDEX = 255

# ----------------------------
# Transforms & helpers
# ----------------------------
img_tf = transforms.Compose([
    transforms.Resize((h_in, w_in)),
    transforms.ToTensor(),
    transforms.Normalize(mean=(0.485, 0.456, 0.406),
                         std=(0.229, 0.224, 0.225)),
])

inv_norm = transforms.Normalize(
    mean=(-0.485/0.229, -0.456/0.224, -0.406/0.225),
    std=(1/0.229, 1/0.224, 1/0.225),
)

def rgb_mask_to_ids(rgb_np: np.ndarray, lut=CLASS2ID, unknown_val=IGNORE_INDEX):
    h, w, _ = rgb_np.shape
    flat = rgb_np.reshape(-1, 3)
    unique_colors, inverse = np.unique(flat, axis=0, return_inverse=True)
    color2id = {tuple(c): lut.get(tuple(c), unknown_val) for c in unique_colors}
    mapped = np.array([color2id[tuple(c)] for c in unique_colors], dtype=np.uint8)
    return mapped[inverse].reshape(h, w)

def ids_to_color(mask_ids: np.ndarray, palette=PALETTE):
    h, w = mask_ids.shape
    out = np.zeros((h, w, 3), dtype=np.uint8)
    for cls_id, rgb in enumerate(palette):
        out[mask_ids == cls_id] = rgb
    return out

# ----------------------------
# Dataset
# ----------------------------
class SemanticDroneDataset(Dataset):
    def __init__(self, image_dir, label_dir):
        self.image_dir = Path(image_dir)
        self.label_dir = Path(label_dir)

        img_exts = {'.jpg', '.jpeg', '.png', '.bmp', '.tif', '.tiff'}
        imgs = {p.stem: p for p in self.image_dir.iterdir() if p.suffix.lower() in img_exts}
        labs = {p.stem: p for p in self.label_dir.iterdir() if p.suffix.lower() in img_exts}
        common = sorted(set(imgs) & set(labs))
        if not common:
            raise RuntimeError("No matching image/mask stems found. Check file names and folders.")
        self.pairs = [(imgs[s], labs[s]) for s in common]

    def __len__(self): return len(self.pairs)

    def __getitem__(self, idx):
        img_fp, lab_fp = self.pairs[idx]
        image = img_tf(Image.open(img_fp).convert('RGB'))
        lab_np = np.array(Image.open(lab_fp).convert('RGB').resize((w_in, h_in), Image.NEAREST))
        label = torch.from_numpy(rgb_mask_to_ids(lab_np, lut=CLASS2ID, unknown_val=IGNORE_INDEX).astype(np.int64))
        return image, label

# ----------------------------
# U-Net Model
# ----------------------------
class UNet(nn.Module):
    def __init__(self, num_classes):
        super().__init__()
        self.enc = nn.ModuleList([
            self.conv_block(3, 64), self.conv_block(64, 128),
            self.conv_block(128, 256), self.conv_block(256, 512)
        ])
        self.pool = nn.MaxPool2d(2)
        self.mid = self.conv_block(512, 1024)
        self.up = nn.ModuleList([
            nn.ConvTranspose2d(1024, 512, 2, 2),
            nn.ConvTranspose2d(512, 256, 2, 2),
            nn.ConvTranspose2d(256, 128, 2, 2),
            nn.ConvTranspose2d(128, 64, 2, 2)
        ])
        self.dec = nn.ModuleList([
            self.conv_block(1024, 512),
            self.conv_block(512, 256),
            self.conv_block(256, 128),
            self.conv_block(128, 64)
        ])
        self.out = nn.Conv2d(64, num_classes, 1)

    def conv_block(self, in_ch, out_ch):
        return nn.Sequential(
            nn.Conv2d(in_ch, out_ch, 3, 1, 1), nn.ReLU(inplace=True), nn.BatchNorm2d(out_ch),
            nn.Conv2d(out_ch, out_ch, 3, 1, 1), nn.ReLU(inplace=True), nn.BatchNorm2d(out_ch)
        )

    def forward(self, x):
        skips = []
        for enc_layer in self.enc:
            x = enc_layer(x)
            skips.append(x)
            x = self.pool(x)
        x = self.mid(x)
        for i in range(4):
            x = self.up[i](x)
            x = self.dec[i](torch.cat([x, skips[-(i+1)]], dim=1))
        return self.out(x)

# ----------------------------
# Metrics
# ----------------------------
@torch.no_grad()
def compute_confusion_matrix(pred, target, num_classes, ignore_index=IGNORE_INDEX):
    mask = target != ignore_index
    pred, target = pred[mask], target[mask]
    k = num_classes * target + pred
    cm = torch.bincount(k, minlength=num_classes**2).reshape(num_classes, num_classes)
    return cm

def iou_from_confmat(cm):
    tp = torch.diag(cm).float()
    fp = cm.sum(0).float() - tp
    fn = cm.sum(1).float() - tp
    denom = tp + fp + fn
    iou = torch.where(denom > 0, tp / denom, torch.zeros_like(tp))
    return iou, iou[denom > 0].mean()

def pixel_accuracy(cm): return torch.diag(cm).sum() / cm.sum().clamp_min(1)

@torch.no_grad()
def evaluate(model, loader, criterion, num_classes):
    model.eval(); total_loss = 0; cm = torch.zeros((num_classes, num_classes), device=device)
    for X, Y in loader:
        X, Y = X.to(device), Y.to(device)
        logits = model(X); loss = criterion(logits, Y)
        total_loss += loss.item()
        pred = torch.argmax(logits, 1)
        cm += compute_confusion_matrix(pred, Y, num_classes)
    val_loss = total_loss / len(loader)
    val_pix_acc = pixel_accuracy(cm).item()
    _, val_mIoU = iou_from_confmat(cm)
    return val_loss, val_pix_acc, val_mIoU.item(), cm

# ----------------------------
# Data sanity check
# ----------------------------
train_ds = SemanticDroneDataset(train_dir, lab_dir)
val_ds   = SemanticDroneDataset(val_dir,   lab_dir)
print(f"[Task 1] Train: {len(train_ds)}, Val: {len(val_ds)}")

def show_random_pairs(dataset, k=3):
    for idx in random.sample(range(len(dataset)), k=min(k, len(dataset))):
        img, lab = dataset[idx]
        img_np = inv_norm(img).permute(1,2,0).clamp(0,1).numpy()
        lab_np, lab_col = lab.numpy(), ids_to_color(lab.numpy())
        fig, axs = plt.subplots(1,3,figsize=(12,4))
        axs[0].imshow(img_np); axs[0].set_title("Image"); axs[0].axis('off')
        axs[1].imshow(lab_np, cmap='gray'); axs[1].set_title("Mask IDs"); axs[1].axis('off')
        axs[2].imshow(lab_col); axs[2].set_title("Mask Colorized"); axs[2].axis('off')
        plt.tight_layout(); plt.show()
show_random_pairs(train_ds)

# ----------------------------
#  Dataloader
# ----------------------------
train_loader = DataLoader(train_ds, BATCH_SIZE, True, num_workers=NUM_WORKERS)
val_loader   = DataLoader(val_ds, VAL_BATCH_SIZE, False, num_workers=NUM_WORKERS)
Xb, Yb = next(iter(train_loader))
print(f" Batch X: {Xb.shape}, Y: {Yb.shape}, num_classes={NUM_CLASSES}")

# ----------------------------
# Training Loop
# ----------------------------
model = UNet(NUM_CLASSES).to(device)
criterion = nn.CrossEntropyLoss(ignore_index=IGNORE_INDEX)
optimizer = optim.Adam(model.parameters(), lr=LR)

hist = {k: [] for k in ["train_loss","val_loss","train_mIoU","val_mIoU","train_pix","val_pix"]}
best_mIoU, best_path = -1, "best_unet.pth"

for epoch in range(1, EPOCHS+1):
    model.train(); run_loss = 0; cm_train = torch.zeros((NUM_CLASSES, NUM_CLASSES), device=device)
    for X, Y in tqdm(train_loader, desc=f"Epoch {epoch}/{EPOCHS}", leave=False):
        X, Y = X.to(device), Y.to(device)
        optimizer.zero_grad(); out = model(X); loss = criterion(out, Y)
        loss.backward(); optimizer.step()
        run_loss += loss.item()
        cm_train += compute_confusion_matrix(torch.argmax(out,1), Y, NUM_CLASSES)
    train_loss = run_loss / len(train_loader)
    train_pix = pixel_accuracy(cm_train).item()
    _, train_mIoU = iou_from_confmat(cm_train)
    val_loss, val_pix, val_mIoU, _ = evaluate(model, val_loader, criterion, NUM_CLASSES)
    hist["train_loss"].append(train_loss); hist["val_loss"].append(val_loss)
    hist["train_pix"].append(train_pix); hist["val_pix"].append(val_pix)
    hist["train_mIoU"].append(train_mIoU.item()); hist["val_mIoU"].append(val_mIoU)
    print(f"Epoch {epoch}: Train mIoU={train_mIoU:.3f}, Val mIoU={val_mIoU:.3f}")
    if val_mIoU > best_mIoU: best_mIoU = val_mIoU; torch.save(model.state_dict(), best_path)

# Plots
ep = np.arange(1, EPOCHS+1)
plt.figure(figsize=(14,4))
plt.subplot(131); plt.plot(ep,hist["train_loss"]); plt.plot(ep,hist["val_loss"]); plt.title("Loss")
plt.subplot(132); plt.plot(ep,hist["train_pix"]); plt.plot(ep,hist["val_pix"]); plt.title("Pixel Acc")
plt.subplot(133); plt.plot(ep,hist["train_mIoU"]); plt.plot(ep,hist["val_mIoU"]); plt.title("mIoU")
plt.tight_layout(); plt.show()

# ----------------------------
# Qualitative Results
# ----------------------------
# best_path = r"best_unet.pth"
# best_model = UNet(NUM_CLASSES).to(device)
# best_model.load_state_dict(torch.load(best_path, map_location=device))
# best_model.eval()
#
# # compute mIoU for the loaded model
# criterion = nn.CrossEntropyLoss(ignore_index=IGNORE_INDEX)
# val_loss, val_pix, val_mIoU, _ = evaluate(best_model, val_loader, criterion, NUM_CLASSES)
# print(f" Loaded best model (mIoU={val_mIoU:.3f})")
#
# # Display 6 samples
# with torch.no_grad():
#     shown = 0
#     for X, Y in val_loader:
#         X, Y = X.to(device), Y.to(device)
#         pred = torch.argmax(best_model(X),1)
#         for i in range(X.size(0)):
#             if shown >= 6: break
#             img = inv_norm(X[i]).permute(1,2,0).clamp(0,1).cpu().numpy()
#             gt, pr = ids_to_color(Y[i].cpu().numpy()), ids_to_color(pred[i].cpu().numpy())
#             fig, ax = plt.subplots(1,3,figsize=(15,5))
#             ax[0].imshow(img); ax[0].set_title("Image"); ax[0].axis('off')
#             ax[1].imshow(gt); ax[1].set_title("Ground Truth"); ax[1].axis('off')
#             ax[2].imshow(pr); ax[2].set_title("Prediction"); ax[2].axis('off')
#             plt.show(); shown += 1
#         if shown >= 6: break
#
# # Per-class IoU
# @torch.no_grad()
# def per_class_iou(model, loader, num_classes):
#     cm = torch.zeros((num_classes, num_classes), device=device)
#     for X, Y in loader:
#         X, Y = X.to(device), Y.to(device)
#         pred = torch.argmax(model(X),1)
#         cm += compute_confusion_matrix(pred, Y, num_classes)
#     iou, miou = iou_from_confmat(cm)
#     return iou.cpu().numpy(), miou.item()
#
# ious, miou = per_class_iou(best_model, val_loader, NUM_CLASSES)
# print("\n Per-class IoU:")
# for cls_id, iou in enumerate(ious):
#     cls_name = CLASS_NAMES[cls_id] if cls_id < len(CLASS_NAMES) else f"class_{cls_id}"
#     print(f"  {cls_id:2d} ({cls_name:>14s}): {iou:.4f}")
# print(f" Mean IoU: {miou:.4f}")
