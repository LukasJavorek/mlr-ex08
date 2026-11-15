"""Ablation study on top of the ex08 notebook pipeline (Task 2 / Task 3).

Nothing here is a rewrite: `LeafDataset`, `LeafNetBN`, the transforms, the loaders and the
training loop are the notebook's own code, copied so this file runs headless. Every variant
changes exactly ONE thing versus V0 and that change is named in VARIANTS below; the training
loop is a single flat function whose `if cfg[...]` lines each say which variant they serve.

Fairness: same train split, same batch order (DataLoader generator seeded), same seed for
numpy / torch / model init, and the SAME fixed `val_camera` set (built once with seed 42)
for every variant.

Run:
    .venv/Scripts/python.exe ablation.py                    # all variants, seed 0
    .venv/Scripts/python.exe ablation.py --only V0,V3       # subset
    .venv/Scripts/python.exe ablation.py --only V0,V9 --seeds 0,1,2
Results are appended to ablation_results.csv after every run, model weights to runs/.
"""

import argparse
import copy
import csv
import json
import random
import time
import traceback
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
from PIL import Image
from torch.utils.data import DataLoader
from torchvision import datasets, models, transforms

HERE = Path(__file__).parent
DATA = HERE / "leaves"                    # notebook has the same folder as data/leaves
RESULTS_CSV = HERE / "ablation_results.csv"
RUNS = HERE / "runs"

# ---------------------------------------------------------------- notebook setup --

mean = [0.485, 0.456, 0.406]              # ImageNet statistics, as in the notebook
std = [0.229, 0.224, 0.225]
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

transform = transforms.Compose([          # notebook cell: validation / clean transform
    transforms.ToTensor(),
    transforms.Normalize(mean, std),
])

aug_transform = transforms.Compose([      # notebook cell 60: existing geometric augmentation
    transforms.RandomResizedCrop(96, scale=(0.6, 1.0)),
    transforms.RandomRotation(20),
    transforms.RandomHorizontalFlip(),
    transforms.ToTensor(),
    transforms.Normalize(mean, std),
])

# V1 / V8 / V9: the same geometric augmentation plus colour jitter.
aug_color_transform = transforms.Compose([
    transforms.RandomResizedCrop(96, scale=(0.6, 1.0)),
    transforms.RandomRotation(20),
    transforms.RandomHorizontalFlip(),
    transforms.ColorJitter(brightness=0.3, contrast=0.3, saturation=0.3, hue=0.08),
    transforms.ToTensor(),
    transforms.Normalize(mean, std),
])


class LeafDataset(torch.utils.data.Dataset):   # verbatim from notebook cell 54
    def __init__(self, folder, transform):
        self.transform = transform
        folder_path = Path(folder)
        self.classes = sorted([d.name for d in folder_path.iterdir() if d.is_dir()])
        class_to_idx = {cls_name: i for i, cls_name in enumerate(self.classes)}
        self.samples = []
        for cls_name in self.classes:
            cls_dir = folder_path / cls_name
            for file_path in sorted(cls_dir.glob("*")):
                if file_path.is_file():
                    self.samples.append((str(file_path), class_to_idx[cls_name]))

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, i):
        img_path, target = self.samples[i]
        img = Image.open(img_path).convert("RGB")
        if self.transform is not None:
            img = self.transform(img)
        return img, target


class LeafNetBN(nn.Module):                    # notebook cell 60, plus the two V2 lines
    """LeafNet with batch normalization after every convolution."""

    def __init__(self, gap_dropout=False):     # V2: AdaptiveAvgPool2d(1) + Dropout(0.2)
        super().__init__()
        self.gap_dropout = gap_dropout
        self.conv1 = nn.Conv2d(3, 16, kernel_size=3, padding=1)
        self.bn1 = nn.BatchNorm2d(16)
        self.conv2 = nn.Conv2d(16, 32, kernel_size=3, padding=1)
        self.bn2 = nn.BatchNorm2d(32)
        self.conv3 = nn.Conv2d(32, 64, kernel_size=3, padding=1)
        self.bn3 = nn.BatchNorm2d(64)
        self.conv4 = nn.Conv2d(64, 64, kernel_size=3, padding=1)
        self.bn4 = nn.BatchNorm2d(64)
        self.pool = nn.MaxPool2d(kernel_size=2, stride=2)
        self.gap = nn.AdaptiveAvgPool2d(1)                 # V2
        self.dropout = nn.Dropout(0.2)                     # V2
        # V2 needs a smaller fc1: after global pooling there is one value per channel (64),
        # not the 64*6*6 map that the baseline flattens.
        self.fc1 = nn.Linear(64, 64) if gap_dropout else nn.Linear(64 * 6 * 6, 64)
        self.fc2 = nn.Linear(64, 4)

    def forward(self, x):
        x = self.pool(torch.relu(self.bn1(self.conv1(x))))
        x = self.pool(torch.relu(self.bn2(self.conv2(x))))
        x = self.pool(torch.relu(self.bn3(self.conv3(x))))
        x = self.pool(torch.relu(self.bn4(self.conv4(x))))
        if self.gap_dropout:                               # V2 instead of x.flatten(1)
            x = self.gap(x).flatten(1)
            x = torch.relu(self.fc1(x))
            return self.fc2(self.dropout(x))
        x = x.flatten(1)
        x = torch.relu(self.fc1(x))
        return self.fc2(x)


# ------------------------------------------------------------------ fixed seeds --

def set_seed(seed):
    """seed = 0 (or 1, 2) for python, numpy and torch, including the model init."""
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


# ------------------------------------------------- camera-like validation set --

camera_augment = transforms.Compose([       # built ONCE with seed 42, shared by all variants
    transforms.RandomResizedCrop(96, scale=(0.5, 1.0)),
    transforms.RandomRotation(30),
    transforms.ColorJitter(brightness=0.3, contrast=0.3, saturation=0.3, hue=0.08),
])


def build_val_camera(val_ds, seed=42):
    """The validation images seen once through camera-like distortion, fixed for every run."""
    images, labels = [], []
    for i, (path, label) in enumerate(val_ds.samples):
        torch.manual_seed(seed * 1000 + i)          # a fixed distortion per image
        images.append(camera_augment(Image.open(path).convert("RGB")))
        labels.append(label)
    x = torch.stack([transform(img) for img in images])
    return x, torch.tensor(labels)


# ------------------------------------------------------------------- evaluation --

@torch.no_grad()
def accuracy(model, x, y):
    model.eval()
    correct = 0
    for i in range(0, len(x), 64):
        correct += (model(x[i:i + 64].to(device)).argmax(dim=1).cpu() == y[i:i + 64]).sum().item()
    return correct / len(x)


@torch.no_grad()
def accuracy_loader(model, loader):
    model.eval()
    correct = total = 0
    for images, labels in loader:
        correct += (model(images.to(device)).argmax(dim=1).cpu() == labels).sum().item()
        total += len(labels)
    return correct / total


# ------------------------------------------------------------------ the variants --

def base_cfg(**changes):
    """One dict per variant. Defaults are the notebook's Task 3 settings."""
    cfg = dict(arch="leafnetbn", augment="geom", gap_dropout=False, optimizer="adam",
               label_smoothing=0.0, best_epoch=False, freeze_backbone=False,
               cosine=False, epochs=15, lr=1e-3)
    cfg.update(changes)
    return cfg


VARIANTS = {
    "V0": ("baseline: Task 3 LeafNetBN + geometric augmentation (notebook)", base_cfg()),
    "V1": ("+ ColorJitter(0.3,0.3,0.3,0.08) in the train transform",
           base_cfg(augment="geom_color")),
    "V2": ("+ AdaptiveAvgPool2d(1) and Dropout(0.2) instead of Flatten",
           base_cfg(gap_dropout=True)),
    "V3": ("+ AdamW with weight_decay=1e-2 instead of Adam",
           base_cfg(optimizer="adamw")),
    "V4": ("+ CrossEntropyLoss(label_smoothing=0.1)",
           base_cfg(label_smoothing=0.1)),
    "V5": ("+ best-epoch checkpoint (deepcopy) instead of last-epoch weights",
           base_cfg(best_epoch=True)),
    "V6": ("transfer: pretrained MobileNetV3-Small, backbone frozen",
           base_cfg(arch="mobilenet", augment="none", freeze_backbone=True)),
    "V7": ("transfer: full fine-tuning of MobileNetV3-Small, lr=3e-4",
           base_cfg(arch="mobilenet", augment="none", lr=3e-4)),
    "V8": ("transfer + geometric & ColorJitter + CosineAnnealingLR + 30 epochs + best epoch",
           base_cfg(arch="mobilenet", augment="geom_color", cosine=True, epochs=30,
                    lr=3e-4, best_epoch=True)),
    "V9": ("best combined: V8 + AdamW(weight_decay=1e-2) + label_smoothing=0.1",
           base_cfg(arch="mobilenet", augment="geom_color", cosine=True, epochs=30,
                    lr=3e-4, best_epoch=True, optimizer="adamw", label_smoothing=0.1)),
}


def build_model(cfg, seed):
    set_seed(seed)                                   # same init for every variant
    if cfg["arch"] == "leafnetbn":
        model = LeafNetBN(gap_dropout=cfg["gap_dropout"])
    else:
        model = models.mobilenet_v3_small(weights="DEFAULT")
        model.classifier[3] = nn.Linear(model.classifier[3].in_features, 4)
        if cfg["freeze_backbone"]:                   # V6: train only the new head
            for p in model.parameters():
                p.requires_grad = False
            for p in model.classifier.parameters():
                p.requires_grad = True
    return model.to(device)


def make_loaders(cfg, seed):
    train_transform = {"none": transform, "geom": aug_transform,
                       "geom_color": aug_color_transform}[cfg["augment"]]
    gen = torch.Generator().manual_seed(seed)        # identical batch order for every variant
    train_loader = DataLoader(datasets.ImageFolder(DATA / "train", transform=train_transform),
                              batch_size=32, shuffle=True, generator=gen)
    clean_train_loader = DataLoader(datasets.ImageFolder(DATA / "train", transform=transform),
                                    batch_size=64)
    return train_loader, clean_train_loader


# --------------------------------------------------------------------- training --

def train_variant(name, what_changed, cfg, seed, val_x, val_y, cam_x, cam_y):
    print(f"\n===== {name} (seed {seed}): {what_changed}", flush=True)
    train_loader, clean_train_loader = make_loaders(cfg, seed)
    model = build_model(cfg, seed)
    criterion = nn.CrossEntropyLoss(label_smoothing=cfg["label_smoothing"])   # V4
    trainable = [p for p in model.parameters() if p.requires_grad]            # V6 head only
    if cfg["optimizer"] == "adamw":                                            # V3
        optimizer = torch.optim.AdamW(trainable, lr=cfg["lr"], weight_decay=1e-2)
    else:
        optimizer = torch.optim.Adam(trainable, lr=cfg["lr"])
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=cfg["epochs"]) \
        if cfg["cosine"] else None                                             # V8

    best_state, best_val = None, -1.0
    t0 = time.time()
    for epoch in range(cfg["epochs"]):
        model.train()
        loss_sum = 0.0
        for images, labels in train_loader:
            images, labels = images.to(device), labels.to(device)
            optimizer.zero_grad()
            loss = criterion(model(images), labels)
            loss.backward()
            optimizer.step()
            loss_sum += loss.item() * len(images)
        if scheduler is not None:
            scheduler.step()

        val_acc = accuracy(model, val_x, val_y)
        # V5: remember the weights of the best epoch instead of trusting the last one.
        if val_acc >= best_val:
            best_val = val_acc
            best_state = copy.deepcopy(model.state_dict())
        print(f"epoch {epoch + 1:2d}  loss {loss_sum / len(train_loader.dataset):.4f}  "
              f"val {val_acc:.3f}", flush=True)

    if cfg["best_epoch"] and best_state is not None:
        model.load_state_dict(best_state)

    RUNS.mkdir(exist_ok=True)
    torch.save(model.state_dict(), RUNS / f"{name}_seed{seed}.pt")
    row = dict(variant_name=name if seed == 0 else f"{name}-s{seed}",
               what_changed=what_changed, epochs=cfg["epochs"], seed=seed,
               train_acc=round(accuracy_loader(model, clean_train_loader), 4),
               val_acc=round(accuracy(model, val_x, val_y), 4),
               camera_val_acc=round(accuracy(model, cam_x, cam_y), 4),
               hist_val_acc="", seconds=round(time.time() - t0, 1))
    print(f"{name} seed{seed}: train {row['train_acc']:.3f}  val {row['val_acc']:.3f}  "
          f"camera_val {row['camera_val_acc']:.3f}  ({row['seconds']}s)", flush=True)
    return row


# --------------------------------------------------------------------------- csv --

FIELDS = ["variant_name", "what_changed", "epochs", "seed", "train_acc", "val_acc",
          "camera_val_acc", "hist_val_acc", "seconds"]


def append_csv(row):
    new = not RESULTS_CSV.exists()
    with open(RESULTS_CSV, "a", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=FIELDS)
        if new:
            writer.writeheader()
        writer.writerow(row)


# -------------------------------------------------------------------------- main --

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--only", default="", help="comma separated variants, e.g. V0,V3")
    parser.add_argument("--seeds", default="0", help="comma separated seeds, e.g. 0,1,2")
    args = parser.parse_args()

    wanted = [v.strip() for v in args.only.split(",") if v.strip()] or list(VARIANTS)
    seeds = [int(s) for s in args.seeds.split(",") if s.strip()]
    print("torch", torch.__version__, "| device:", device, "| variants:", wanted, "| seeds:", seeds)

    val_ds = datasets.ImageFolder(DATA / "val", transform=transform)
    set_seed(0)
    val_x, val_y = torch.stack([transform(Image.open(p).convert("RGB")) for p, _ in val_ds.samples]), \
        torch.tensor([l for _, l in val_ds.samples])
    cam_x, cam_y = build_val_camera(val_ds, seed=42)        # once, shared by every variant
    print("val:", len(val_y), "| val_camera:", len(cam_y), "| classes:", val_ds.classes)

    for name in wanted:
        what_changed, cfg = VARIANTS[name]
        for seed in seeds:
            try:    # one crashing variant must not throw away the rest of the sweep
                append_csv(train_variant(name, what_changed, cfg, seed, val_x, val_y, cam_x, cam_y))
            except Exception:
                print(f"{name} seed{seed}: FAILED", flush=True)
                traceback.print_exc()

    print("\ndone, results in", RESULTS_CSV)
