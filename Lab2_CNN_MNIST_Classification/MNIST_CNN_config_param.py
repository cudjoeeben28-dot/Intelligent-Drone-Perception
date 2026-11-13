
from __future__ import annotations
import argparse, itertools, random, time
from typing import Tuple, List

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.optim.lr_scheduler import StepLR
from torch.utils.data import DataLoader
from torchvision import datasets, transforms, models

# ---------------------------------------------------------------------------
# 2.  ARCHITECTURES
# ---------------------------------------------------------------------------
class Net2Conv(nn.Module):
    def __init__(self, drop: float):
        super().__init__()
        self.conv1 = nn.Conv2d(1, 32, 3, padding=1)
        self.conv2 = nn.Conv2d(32, 64, 3, padding=1)
        self.fc1   = nn.Linear(64 * 14 * 14, 128)  # 64×14×14 = 12 544 in‑features
        self.fc2   = nn.Linear(128, 10)
        self.drop  = nn.Dropout(drop)
    def forward(self, x):
        x = F.relu(self.conv1(x))
        x = F.max_pool2d(F.relu(self.conv2(x)), 2)
        x = torch.flatten(x, 1)
        x = self.drop(F.relu(self.fc1(x)))
        return F.log_softmax(self.fc2(x), dim=1)

class Net3Conv(nn.Module):
    def __init__(self, drop: float):
        super().__init__()
        self.conv1 = nn.Conv2d(1, 32, 3, padding=1)
        self.conv2 = nn.Conv2d(32, 64, 3, padding=1)
        self.conv3 = nn.Conv2d(64, 128, 3, padding=1)
        self.fc    = nn.Linear(128 * 7 * 7, 10)     # 128×7×7 = 6 272 in‑features
        self.drop  = nn.Dropout(drop)
    def forward(self, x):
        x = F.relu(self.conv1(x))
        x = F.max_pool2d(F.relu(self.conv2(x)), 2)
        x = F.max_pool2d(F.relu(self.conv3(x)), 2)
        x = torch.flatten(x, 1)
        x = self.drop(x)
        return F.log_softmax(self.fc(x), dim=1)

ARCHS = {"2conv": Net2Conv, "3conv": Net3Conv}

# ---------------------------------------------------------------------------
# TRANSFORMS
# ---------------------------------------------------------------------------
MEAN, STD = 0.1307, 0.3081

def build_transforms(aug: bool, rgb: bool):
    train = []

    if aug:
        train.append(transforms.RandomAffine(
            10,
            translate=(0.1, 0.1)
        ))

    if rgb:
        train.append(transforms.Resize((32, 32)))
        train.append(transforms.Grayscale(num_output_channels=3))

    train += [transforms.ToTensor()]

    norm_rgb = transforms.Normalize(
        (0.5, 0.5, 0.5),
        (0.5, 0.5, 0.5)
    )

    norm_g = transforms.Normalize(
        (MEAN,),
        (STD,)
    )

    train.append(norm_rgb if rgb else norm_g)

    test = []

    if rgb:
        test.append(transforms.Resize((32, 32)))
        test.append(transforms.Grayscale(num_output_channels=3))

    test += [
        transforms.ToTensor(),
        norm_rgb if rgb else norm_g
    ]

    return transforms.Compose(train), transforms.Compose(test)

# ---------------------------------------------------------------------------
# TRAIN / EVAL LOOPS
# ---------------------------------------------------------------------------

def train_epoch(model, loader, opt, device):
    model.train()
    for x, y in loader:
        x, y = x.to(device), y.to(device)
        opt.zero_grad()
        loss = F.nll_loss(model(x), y)
        loss.backward()
        opt.step()


def evaluate(model, loader, device):
    model.eval(); correct = 0
    with torch.no_grad():
        for x, y in loader:
            x, y = x.to(device), y.to(device)
            pred = model(x).argmax(1)
            correct += pred.eq(y).sum().item()
    return 100 * correct / len(loader.dataset)

# ---------------------------------------------------------------------------
# EXPERIMENT DRIVER
# ---------------------------------------------------------------------------

def run_exp(args, device):
    rgb = args.pretrained
    tf_train, tf_test = build_transforms(args.aug, rgb)
    train_set = datasets.MNIST("../data", train=True, download=True, transform=tf_train)
    test_set  = datasets.MNIST("../data", train=False, transform=tf_test)
    train_ld  = DataLoader(train_set, batch_size=args.batch, shuffle=True, num_workers=2)
    test_ld   = DataLoader(test_set,  batch_size=1000, shuffle=False, num_workers=2)

    if args.pretrained:
        model = models.mobilenet_v2(weights="IMAGENET1K_V1")
        model.classifier[1] = nn.Linear(model.last_channel, 10)
        for p in model.features.parameters():
            p.requires_grad = False
    else:
        model = ARCHS[args.arch](args.drop)

    model.to(device)
    params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"Trainable parameters: {params:,}")

    if args.optim == "adam":
        opt = optim.Adam(model.parameters(), lr=args.lr, weight_decay=args.wd)
    elif args.optim == "sgd":
        opt = optim.SGD(model.parameters(), lr=args.lr, momentum=0.9, weight_decay=args.wd)
    else:
        opt = optim.Adadelta(model.parameters(), lr=args.lr, weight_decay=args.wd)
    sched = StepLR(opt, 1, gamma=args.gamma)

    best = 0
    for ep in range(1, args.epochs+1):
        train_epoch(model, train_ld, opt, device)
        acc = evaluate(model, test_ld, device)
        print(f"Epoch {ep:02d}: {acc:.2f} %")
        best = max(best, acc)
        sched.step()
    return best

# ---------------------------------------------------------------------------
# GRID SEARCH (item 4)
# ---------------------------------------------------------------------------

def grid_search(args, device):
    space = {'lr':[1e-3,5e-4], 'batch':[64,128], 'drop':[0.0,0.25]}
    best, best_cfg = 0, None
    for lr, bs, dr in itertools.product(*space.values()):
        print(f"lr={lr} batch={bs} drop={dr}")
        args.lr, args.batch, args.drop = lr, bs, dr
        acc = run_exp(args, device)
        if acc>best: best,best_cfg=acc,(lr,bs,dr)
    print("Best", best_cfg, "→", best)

# ---------------------------------------------------------------------------
# ARG PARSE & ITEM PRESETS
# ---------------------------------------------------------------------------

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--item', type=int, default=0, help='0‑6 select lab demo')
    # common tunables (can override presets)
    ap.add_argument('--epochs', type=int, default=1)
    ap.add_argument('--batch',  type=int, default=64)
    ap.add_argument('--lr',     type=float, default=1e-3)
    ap.add_argument('--gamma',  type=float, default=0.7)
    ap.add_argument('--arch',   choices=list(ARCHS), default='2conv')
    ap.add_argument('--optim',  choices=['adam','sgd','adadelta'], default='adadelta')
    ap.add_argument('--drop',   type=float, default=0.25)
    ap.add_argument('--wd',     type=float, default=0.0)
    ap.add_argument('--aug',    action='store_true')
    ap.add_argument('--pretrained', action='store_true')
    args = ap.parse_args()

    # ---- presets per item -----------------------------------------
    if args.item == 0:      # baseline
        args.arch='2conv'; args.optim='adadelta'; args.lr=1.0; args.aug=False; args.pretrained=False
    elif args.item == 1:    # architecture demo
        args.arch='3conv'
    elif args.item == 2:    # optimiser demo
        args.optim='adam'; args.lr=1e-3
    elif args.item == 3:    # regularisation demo
        args.drop=0.5; args.wd=1e-4; args.optim='adam' #wd = weight decay
    elif args.item == 4:    # grid search
        device=torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        grid_search(args, device)
        return
    elif args.item == 5:    # data augmentation
        args.aug=True
    elif args.item == 6:    # transfer learning
        args.pretrained=True; args.aug=True; args.optim='adam'; args.lr=1e-3
    else:
        raise SystemExit("item must be 0‑6")

    # ---------------------------------------------------------------
    torch.manual_seed(42); random.seed(42)
    device=torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    best = run_exp(args, device)
    print(f"Best accuracy: {best:.2f} %")

if __name__=='__main__':
    t0=time.time(); main(); print(f"Finished in {time.time()-t0:.1f} s")

# python name of the file.py --epochs 10  --lr 0.5  