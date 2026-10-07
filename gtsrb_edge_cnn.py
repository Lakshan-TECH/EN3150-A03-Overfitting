#!/usr/bin/env python3
"""
EN3150 Assignment 03 - Resource-constrained CNN for edge image classification
Dataset : GTSRB (German Traffic Sign Recognition Benchmark), 43 classes, resized to 64x64
Frameworks: PyTorch + torchvision + scikit-learn + matplotlib

What this script does (one run produces everything the report needs):
  1. Data preparation   : download GTSRB, resize to IMG x IMG, stratified 70/15/15 split
  2. Custom models      : Model A (standard CNN) and Model B (depthwise-separable CNN, <100k params)
  3. Optimizer study    : Adam vs SGD vs SGD+Momentum (+ momentum sweep) on Model B
  4. Train / evaluate   : >=20 epochs, loss curves, accuracy, confusion matrix, precision, recall
  5. SOTA fine-tuning   : MobileNetV2 and SqueezeNet1.1 (ImageNet pre-trained), same splits
  6. Comparison tables  : written to <out>/results_tables.md and <out>/results.json

Usage
  python gtsrb_edge_cnn.py                       # full run (GPU recommended, e.g. Colab T4)
  python gtsrb_edge_cnn.py --epochs 2 --fake-data --skip-sota   # quick smoke test, no download

Outputs are written to ./outputs (change with --out).
"""
import argparse
import copy
import csv
import json
import math
import os
import random
import time

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
from PIL import Image
from sklearn.metrics import accuracy_score, confusion_matrix, precision_recall_fscore_support
from sklearn.model_selection import train_test_split

NUM_CLASSES = 43
CLASS_NAMES = [
    "Speed limit 20", "Speed limit 30", "Speed limit 50", "Speed limit 60", "Speed limit 70",
    "Speed limit 80", "End of speed limit 80", "Speed limit 100", "Speed limit 120", "No passing",
    "No passing (>3.5t)", "Right-of-way at next junction", "Priority road", "Yield", "Stop",
    "No vehicles", "Vehicles >3.5t prohibited", "No entry", "General caution", "Dangerous curve left",
    "Dangerous curve right", "Double curve", "Bumpy road", "Slippery road", "Road narrows (right)",
    "Road work", "Traffic signals", "Pedestrians", "Children crossing", "Bicycles crossing",
    "Ice/snow", "Wild animals crossing", "End of all limits", "Turn right ahead", "Turn left ahead",
    "Ahead only", "Go straight or right", "Go straight or left", "Keep right", "Keep left",
    "Roundabout mandatory", "End of no passing", "End of no passing (>3.5t)",
]


# --------------------------------------------------------------------------------------
# Utilities
# --------------------------------------------------------------------------------------
def set_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def sync(device):
    if device.type == "cuda":
        torch.cuda.synchronize()


def count_params(model):
    """Total TRAINABLE parameters (BatchNorm running mean/var are buffers, not counted)."""
    return sum(p.numel() for p in model.parameters() if p.requires_grad)


def count_macs(model, img):
    """Multiply-accumulate operations for one forward pass of one image (Conv2d + Linear)."""
    macs = [0]

    def conv_hook(m, i, o):
        macs[0] += (o.shape[2] * o.shape[3] * m.out_channels *
                    (m.in_channels // m.groups) * m.kernel_size[0] * m.kernel_size[1])

    def lin_hook(m, i, o):
        macs[0] += m.in_features * m.out_features

    hooks = []
    for m in model.modules():
        if isinstance(m, nn.Conv2d):
            hooks.append(m.register_forward_hook(conv_hook))
        elif isinstance(m, nn.Linear):
            hooks.append(m.register_forward_hook(lin_hook))
    was_training = model.training
    dev = next(model.parameters()).device
    model.eval()
    with torch.no_grad():
        model(torch.zeros(1, 3, img, img, device=dev))
    for h in hooks:
        h.remove()
    model.train(was_training)
    return macs[0]


def model_file_size_kb(model, path):
    """Size of the saved fp32 state_dict on disk, in KB."""
    torch.save(model.state_dict(), path)
    kb = os.path.getsize(path) / 1024.0
    return kb


@torch.no_grad()
def cpu_latency_ms(model, img, runs=100, warmup=10):
    """Single-image, single-thread CPU inference latency (rough proxy for an edge CPU)."""
    m = copy.deepcopy(model).cpu().eval()
    old_threads = torch.get_num_threads()
    torch.set_num_threads(1)
    x = torch.randn(1, 3, img, img)
    for _ in range(warmup):
        m(x)
    t0 = time.perf_counter()
    for _ in range(runs):
        m(x)
    ms = (time.perf_counter() - t0) / runs * 1000.0
    torch.set_num_threads(old_threads)
    return ms


# --------------------------------------------------------------------------------------
# 1. Data preparation
# --------------------------------------------------------------------------------------
def load_gtsrb(root, img, fake=False, seed=0):
    """Return (x uint8 N x H x W x 3, y int N). Train+test official splits are merged and
    then re-split 70/15/15 so that the assignment's split ratios are respected."""
    if fake:  # random data for smoke-testing the pipeline only
        rng = np.random.default_rng(seed)
        x = rng.integers(0, 256, size=(3000, img, img, 3), dtype=np.uint8)
        y = rng.integers(0, NUM_CLASSES, size=3000)
        return x, y
    os.makedirs(root, exist_ok=True)
    cache = os.path.join(root, f"gtsrb_{img}.npz")
    if os.path.exists(cache):
        d = np.load(cache)
        return d["x"], d["y"]
    from torchvision.datasets import GTSRB
    xs, ys = [], []
    for split in ("train", "test"):
        ds = GTSRB(root=root, split=split, download=True)
        print(f"  loading GTSRB {split}: {len(ds)} images")
        for i in range(len(ds)):
            im, lab = ds[i]
            xs.append(np.asarray(im.convert("RGB").resize((img, img), Image.BILINEAR)))
            ys.append(int(lab))
    x, y = np.stack(xs), np.array(ys)
    np.savez_compressed(cache, x=x, y=y)
    return x, y


def split_data(y, seed):
    """Stratified 70 / 15 / 15 split -> index arrays."""
    idx = np.arange(len(y))
    tr, tmp = train_test_split(idx, test_size=0.30, stratify=y, random_state=seed)
    va, te = train_test_split(tmp, test_size=0.50, stratify=y[tmp], random_state=seed)
    return tr, va, te


def channel_stats(x_u8, chunk=4096):
    """Per-channel mean/std (0-1 range) of a uint8 N x 3 x H x W tensor."""
    s = torch.zeros(3, dtype=torch.float64)
    s2 = torch.zeros(3, dtype=torch.float64)
    n = 0
    for i in range(0, len(x_u8), chunk):
        b = x_u8[i:i + chunk].float().div(255).cpu().double()
        s += b.sum(dim=(0, 2, 3))
        s2 += (b ** 2).sum(dim=(0, 2, 3))
        n += b.shape[0] * b.shape[2] * b.shape[3]
    mean = s / n
    std = (s2 / n - mean ** 2).sqrt()
    return mean.float().view(1, 3, 1, 1), std.float().view(1, 3, 1, 1)


def prep(xb_u8, mean, std, resize=None):
    x = xb_u8.float().div(255)
    x = (x - mean) / std
    if resize is not None and resize != x.shape[-1]:
        x = F.interpolate(x, size=(resize, resize), mode="bilinear", align_corners=False)
    return x


def random_affine(x, max_deg=10, max_shift=0.1, max_scale=0.1):
    """Light GPU-side augmentation (rotation / translation / zoom). No flips: signs are
    direction-sensitive (e.g. 'turn left' vs 'turn right')."""
    b = x.size(0)
    dev = x.device
    ang = (torch.rand(b, device=dev) * 2 - 1) * math.radians(max_deg)
    sc = 1 + (torch.rand(b, device=dev) * 2 - 1) * max_scale
    tx = (torch.rand(b, device=dev) * 2 - 1) * max_shift * 2
    ty = (torch.rand(b, device=dev) * 2 - 1) * max_shift * 2
    cos, sin = torch.cos(ang) / sc, torch.sin(ang) / sc
    theta = torch.stack([torch.stack([cos, -sin, tx], 1), torch.stack([sin, cos, ty], 1)], 1)
    grid = F.affine_grid(theta, x.size(), align_corners=False)
    return F.grid_sample(x, grid, padding_mode="border", align_corners=False)


def iterate(x, y, bs, shuffle):
    n = len(y)
    order = torch.randperm(n, device=x.device) if shuffle else torch.arange(n, device=x.device)
    for i in range(0, n, bs):
        j = order[i:i + bs]
        yield x[j], y[j]


# --------------------------------------------------------------------------------------
# 2. Custom architectures
# --------------------------------------------------------------------------------------
class ModelA(nn.Module):
    """Standard CNN: [Conv3x3 -> BN -> ReLU -> MaxPool2] x 4  ->  FC128 -> FC43.
    Channels 3 -> 16 -> 32 -> 64 -> 128, spatial 64 -> 32 -> 16 -> 8 -> 4."""

    def __init__(self, nc=NUM_CLASSES, widths=(16, 32, 64, 128), fc=128, img=64):
        super().__init__()
        layers, cin = [], 3
        for w in widths:
            layers += [nn.Conv2d(cin, w, 3, padding=1, bias=False), nn.BatchNorm2d(w),
                       nn.ReLU(inplace=True), nn.MaxPool2d(2)]
            cin = w
        self.features = nn.Sequential(*layers)
        side = img // (2 ** len(widths))
        self.classifier = nn.Sequential(nn.Flatten(), nn.Linear(cin * side * side, fc),
                                        nn.ReLU(inplace=True), nn.Dropout(0.3), nn.Linear(fc, nc))

    def forward(self, x):
        return self.classifier(self.features(x))


def dw_sep_block(cin, cout):
    """Depthwise 3x3 (groups=cin) -> BN -> ReLU6 -> Pointwise 1x1 -> BN -> ReLU6."""
    return nn.Sequential(
        nn.Conv2d(cin, cin, 3, padding=1, groups=cin, bias=False), nn.BatchNorm2d(cin), nn.ReLU6(inplace=True),
        nn.Conv2d(cin, cout, 1, bias=False), nn.BatchNorm2d(cout), nn.ReLU6(inplace=True))


class ModelB(nn.Module):
    """Lightweight CNN with depthwise-separable convolutions (66,827 trainable params).
    Stem Conv3x3(3->16)+Pool -> DS(16->32)+Pool -> DS(32->64)+Pool -> DS(64->128)+Pool
    -> DS(128->256) -> GlobalAvgPool -> FC64 -> FC43."""

    def __init__(self, nc=NUM_CLASSES):
        super().__init__()
        self.features = nn.Sequential(
            nn.Conv2d(3, 16, 3, padding=1, bias=False), nn.BatchNorm2d(16), nn.ReLU6(inplace=True),
            nn.MaxPool2d(2),                       # 64 -> 32
            dw_sep_block(16, 32), nn.MaxPool2d(2),  # 32 -> 16
            dw_sep_block(32, 64), nn.MaxPool2d(2),  # 16 -> 8
            dw_sep_block(64, 128), nn.MaxPool2d(2),  # 8 -> 4
            dw_sep_block(128, 256),                 # 4 x 4
            nn.AdaptiveAvgPool2d(1))
        self.classifier = nn.Sequential(nn.Flatten(), nn.Linear(256, 64), nn.ReLU6(inplace=True),
                                        nn.Dropout(0.2), nn.Linear(64, nc))

    def forward(self, x):
        return self.classifier(self.features(x))


def build_sota(name, nc, pretrained=True):
    from torchvision import models
    if name == "mobilenet_v2":
        w = models.MobileNet_V2_Weights.IMAGENET1K_V1 if pretrained else None
        m = models.mobilenet_v2(weights=w)
        m.classifier[1] = nn.Linear(m.last_channel, nc)
    elif name == "squeezenet1_1":
        w = models.SqueezeNet1_1_Weights.IMAGENET1K_V1 if pretrained else None
        m = models.squeezenet1_1(weights=w)
        m.classifier[1] = nn.Conv2d(512, nc, kernel_size=1)
    else:
        raise ValueError(name)
    return m


# --------------------------------------------------------------------------------------
# 3/4. Training and evaluation
# --------------------------------------------------------------------------------------
@torch.no_grad()
def predict(model, x, y, mean, std, resize=None, bs=512):
    model.eval()
    logits = torch.cat([model(prep(x[i:i + bs], mean, std, resize)) for i in range(0, len(y), bs)])
    loss = F.cross_entropy(logits, y).item()
    return loss, logits.argmax(1).cpu().numpy()


def train_model(name, model, opt, D, mean, std, epochs, bs, augment, device, resize=None):
    """Train for `epochs`, track train/val loss+acc, keep the best-validation-accuracy weights."""
    model.to(device)
    hist = {"train_loss": [], "val_loss": [], "train_acc": [], "val_acc": [], "epoch_time": []}
    best_acc, best_state = -1.0, None
    for ep in range(1, epochs + 1):
        model.train()
        sync(device)
        t0 = time.time()
        tl, tc, n = 0.0, 0, 0
        for xb, yb in iterate(D["xtr"], D["ytr"], bs, True):
            xb = prep(xb, mean, std, resize)
            if augment:
                xb = random_affine(xb)
            out = model(xb)
            loss = F.cross_entropy(out, yb)
            opt.zero_grad(set_to_none=True)
            loss.backward()
            opt.step()
            tl += loss.item() * len(yb)
            tc += (out.argmax(1) == yb).sum().item()
            n += len(yb)
        sync(device)
        ep_time = time.time() - t0  # training-pass time only
        vl, vpred = predict(model, D["xva"], D["yva"], mean, std, resize)
        va = float((vpred == D["yva"].cpu().numpy()).mean())
        for k, v in zip(hist.keys(), [tl / n, vl, tc / n, va, ep_time]):
            hist[k].append(v)
        if va > best_acc:
            best_acc, best_state = va, copy.deepcopy(model.state_dict())
        print(f"[{name}] ep {ep:02d}/{epochs}  train_loss {tl / n:.4f}  val_loss {vl:.4f}  "
              f"train_acc {tc / n:.4f}  val_acc {va:.4f}  ({ep_time:.1f}s)")
    model.load_state_dict(best_state)
    return hist


def evaluate_test(name, model, D, mean, std, out, resize=None):
    _, pred = predict(model, D["xte"], D["yte"], mean, std, resize)
    yt = D["yte"].cpu().numpy()
    labels = list(range(NUM_CLASSES))
    cm = confusion_matrix(yt, pred, labels=labels)
    pm, rm, fm, _ = precision_recall_fscore_support(yt, pred, average="macro", zero_division=0)
    pw, rw, fw, _ = precision_recall_fscore_support(yt, pred, average="weighted", zero_division=0)
    pc, rc, fc, sc = precision_recall_fscore_support(yt, pred, labels=labels, zero_division=0)
    np.savetxt(os.path.join(out, f"confusion_{name}.csv"), cm, fmt="%d", delimiter=",")
    with open(os.path.join(out, f"per_class_{name}.csv"), "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["class_id", "name", "precision", "recall", "f1", "support"])
        for i in labels:
            w.writerow([i, CLASS_NAMES[i], f"{pc[i]:.4f}", f"{rc[i]:.4f}", f"{fc[i]:.4f}", int(sc[i])])
    # confusion-matrix heat-map (row-normalised)
    cmn = cm / np.maximum(cm.sum(1, keepdims=True), 1)
    plt.figure(figsize=(8, 7))
    plt.imshow(cmn, cmap="Blues", vmin=0, vmax=1)
    plt.colorbar(label="fraction of true class")
    plt.xlabel("Predicted class")
    plt.ylabel("True class")
    plt.title(f"Confusion matrix (test) - {name}")
    plt.tight_layout()
    plt.savefig(os.path.join(out, f"confusion_{name}.png"), dpi=150)
    plt.close()
    # most frequent confusions
    off = cm.copy()
    np.fill_diagonal(off, 0)
    top = np.dstack(np.unravel_index(np.argsort(-off.ravel())[:8], off.shape))[0]
    confusions = [(CLASS_NAMES[a], CLASS_NAMES[b], int(off[a, b])) for a, b in top if off[a, b] > 0]
    return {"test_acc": float(accuracy_score(yt, pred)),
            "precision_macro": float(pm), "recall_macro": float(rm), "f1_macro": float(fm),
            "precision_weighted": float(pw), "recall_weighted": float(rw), "f1_weighted": float(fw),
            "top_confusions(true,pred,count)": confusions}


def plot_curves(hists, path, title):
    """Training (dashed) and validation (solid) loss for every model in `hists`."""
    plt.figure(figsize=(7, 4.5))
    for i, (name, h) in enumerate(hists.items()):
        c = f"C{i}"
        e = range(1, len(h["train_loss"]) + 1)
        plt.plot(e, h["train_loss"], "--", color=c, label=f"{name} train")
        plt.plot(e, h["val_loss"], "-", color=c, label=f"{name} val")
    plt.xlabel("Epoch")
    plt.ylabel("Cross-entropy loss")
    plt.title(title)
    plt.grid(alpha=0.3)
    plt.legend()
    plt.tight_layout()
    plt.savefig(path, dpi=150)
    plt.close()


def plot_val(hists, path, title):
    fig, ax = plt.subplots(1, 2, figsize=(11, 4.2))
    for name, h in hists.items():
        e = range(1, len(h["val_loss"]) + 1)
        ax[0].plot(e, h["val_loss"], label=name)
        ax[1].plot(e, h["val_acc"], label=name)
    ax[0].set_ylabel("Validation loss")
    ax[1].set_ylabel("Validation accuracy")
    for a in ax:
        a.set_xlabel("Epoch")
        a.grid(alpha=0.3)
    ax[0].set_yscale("log")
    ax[1].legend(fontsize=8)
    fig.suptitle(title)
    fig.tight_layout()
    fig.savefig(path, dpi=150)
    plt.close(fig)


def summarise_hist(h):
    e90 = next((i + 1 for i, v in enumerate(h["val_acc"]) if v >= 0.90), None)
    return {"best_val_acc": max(h["val_acc"]), "final_val_acc": h["val_acc"][-1],
            "min_val_loss": min(h["val_loss"]), "final_val_loss": h["val_loss"][-1],
            "epochs_to_90pct_val": e90, "mean_epoch_time_s": float(np.mean(h["epoch_time"]))}


def dataset_figures(x, y, tr, va, te, out):
    counts = np.bincount(y, minlength=NUM_CLASSES)
    plt.figure(figsize=(10, 3.5))
    plt.bar(range(NUM_CLASSES), counts)
    plt.xlabel("Class id")
    plt.ylabel("Images")
    plt.title("GTSRB class distribution (all images)")
    plt.tight_layout()
    plt.savefig(os.path.join(out, "fig_class_distribution.png"), dpi=150)
    plt.close()
    fig, axs = plt.subplots(5, 9, figsize=(11, 6.5))
    for c, a in enumerate(axs.ravel()):
        a.axis("off")
        if c < NUM_CLASSES:
            a.imshow(x[np.where(y == c)[0][0]])
            a.set_title(str(c), fontsize=8)
    fig.suptitle("One sample per class (resized)")
    fig.tight_layout()
    fig.savefig(os.path.join(out, "fig_samples.png"), dpi=150)
    plt.close(fig)
    print(f"Dataset: {len(y)} images | train {len(tr)} | val {len(va)} | test {len(te)}")


# --------------------------------------------------------------------------------------
# Main
# --------------------------------------------------------------------------------------
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data", default="./data")
    ap.add_argument("--out", default="./outputs")
    ap.add_argument("--img", type=int, default=64, help="image side (<=64 per assignment)")
    ap.add_argument("--epochs", type=int, default=20)
    ap.add_argument("--batch", type=int, default=128)
    ap.add_argument("--lr", type=float, default=1e-3, help="Adam learning rate (custom models)")
    ap.add_argument("--sgd-lr", type=float, default=1e-2)
    ap.add_argument("--sota-lr", type=float, default=5e-4)
    ap.add_argument("--sota-img", type=int, default=None,
                    help="optional up-sampling size fed to the SOTA nets (e.g. 96/128); default = --img")
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--no-augment", action="store_true")
    ap.add_argument("--skip-opt", action="store_true", help="skip optimizer comparison")
    ap.add_argument("--skip-sota", action="store_true", help="skip SOTA fine-tuning")
    ap.add_argument("--fake-data", action="store_true", help="random data, smoke test only")
    args = ap.parse_args()
    assert args.img <= 64, "assignment: maximum resolution is 64x64"
    os.makedirs(args.out, exist_ok=True)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print("device:", device)
    set_seed(args.seed)
    augment = not args.no_augment

    # ---- 1. data ----
    x, y = load_gtsrb(args.data, args.img, args.fake_data, args.seed)
    tr, va, te = split_data(y, args.seed)
    dataset_figures(x, y, tr, va, te, args.out)
    xt = torch.from_numpy(x).permute(0, 3, 1, 2).contiguous()
    yt = torch.from_numpy(y).long()
    D = {k: v.to(device) for k, v in {
        "xtr": xt[tr], "ytr": yt[tr], "xva": xt[va], "yva": yt[va], "xte": xt[te], "yte": yt[te]}.items()}
    mean, std = channel_stats(D["xtr"])
    mean, std = mean.to(device), std.to(device)
    imn_mean = torch.tensor([0.485, 0.456, 0.406], device=device).view(1, 3, 1, 1)
    imn_std = torch.tensor([0.229, 0.224, 0.225], device=device).view(1, 3, 1, 1)
    print("train-set mean/std:", mean.flatten().tolist(), std.flatten().tolist())

    results, hists = {}, {}

    # ---- 2+4. custom models A and B (Adam) ----
    def run_custom(name, model):
        n_params = count_params(model)
        macs = count_macs(model.to(device), args.img)
        print(f"\n=== {name}: {n_params:,} trainable params, {macs / 1e6:.2f} M MACs ===")
        opt = torch.optim.Adam(model.parameters(), lr=args.lr)
        h = train_model(name, model, opt, D, mean, std, args.epochs, args.batch, augment, device)
        r = evaluate_test(name, model, D, mean, std, args.out)
        r.update({"params": n_params, "macs": macs,
                  "size_kb_fp32_file": model_file_size_kb(model, os.path.join(args.out, f"{name}.pt")),
                  "size_kb_int8_estimate": n_params / 1024.0,
                  "epoch_time_s": float(np.mean(h["epoch_time"])),
                  "cpu_latency_ms": cpu_latency_ms(model, args.img), **summarise_hist(h)})
        model.to(device)
        results[name], hists[name] = r, h
        print(json.dumps({k: v for k, v in r.items() if k != "top_confusions(true,pred,count)"}, indent=1))
        print("top confusions:", r["top_confusions(true,pred,count)"])

    set_seed(args.seed)
    run_custom("ModelA", ModelA(img=args.img))
    set_seed(args.seed)
    run_custom("ModelB", ModelB())
    plot_curves({"Model A": hists["ModelA"], "Model B": hists["ModelB"]},
                os.path.join(args.out, "fig_loss_curves_AB.png"), "Training / validation loss (Adam, lr=1e-3)")
    for nme in ("ModelA", "ModelB"):
        plot_curves({nme: hists[nme]}, os.path.join(args.out, f"fig_loss_curves_{nme}.png"),
                    f"{nme}: training / validation loss")

    # ---- 3. optimizer comparison on Model B ----
    opt_summ = {}
    if not args.skip_opt:
        opt_hists = {f"Adam lr={args.lr:g}": hists["ModelB"]}
        configs = {f"SGD lr={args.sgd_lr:g} (m=0)": 0.0,
                   f"SGD lr={args.sgd_lr:g} (m=0.5)": 0.5,
                   f"SGD lr={args.sgd_lr:g} (m=0.9)": 0.9,
                   f"SGD lr={args.sgd_lr:g} (m=0.99)": 0.99}
        for label, mom in configs.items():
            set_seed(args.seed)
            m = ModelB().to(device)
            opt = torch.optim.SGD(m.parameters(), lr=args.sgd_lr, momentum=mom)
            print(f"\n=== Optimizer study: {label} ===")
            opt_hists[label] = train_model(label, m, opt, D, mean, std, args.epochs, args.batch, augment, device)
            _, pred = predict(m, D["xte"], D["yte"], mean, std)
            opt_hists[label]["_test_acc"] = float((pred == D["yte"].cpu().numpy()).mean())
        plot_val({k: v for k, v in opt_hists.items() if "m=0.5" not in k and "m=0.99" not in k},
                 os.path.join(args.out, "fig_optimizers.png"), "Model B: Adam vs SGD vs SGD+Momentum")
        plot_val({k: v for k, v in opt_hists.items() if k.startswith("SGD")},
                 os.path.join(args.out, "fig_momentum_sweep.png"), "Model B: effect of momentum (SGD)")
        for k, h in opt_hists.items():
            s = summarise_hist(h)
            s["test_acc"] = results["ModelB"]["test_acc"] if k.startswith("Adam") else h["_test_acc"]
            opt_summ[k] = s
        results["optimizer_study"] = opt_summ

    # ---- 5. SOTA fine-tuning ----
    if not args.skip_sota:
        s_img = args.sota_img or args.img
        resize = s_img if s_img != args.img else None
        for name in ("mobilenet_v2", "squeezenet1_1"):
            set_seed(args.seed)
            model = build_sota(name, NUM_CLASSES, pretrained=not args.fake_data).to(device)
            n_params = count_params(model)
            macs = count_macs(model, s_img)
            print(f"\n=== {name}: {n_params:,} params, {macs / 1e6:.1f} M MACs @ {s_img}px ===")
            opt = torch.optim.Adam(model.parameters(), lr=args.sota_lr)
            h = train_model(name, model, opt, D, imn_mean, imn_std, args.epochs, args.batch, augment, device, resize)
            r = evaluate_test(name, model, D, imn_mean, imn_std, args.out, resize)
            size_kb = model_file_size_kb(model, os.path.join(args.out, f"{name}.pt"))
            r.update({"params": n_params, "macs": macs, "input_px": s_img,
                      "size_kb_fp32_file": size_kb, "size_mb_fp32_file": size_kb / 1024.0,
                      "size_kb_int8_estimate": n_params / 1024.0,
                      "epoch_time_s": float(np.mean(h["epoch_time"])),
                      "cpu_latency_ms": cpu_latency_ms(model, s_img), **summarise_hist(h)})
            results[name], hists[name] = r, h
            print(json.dumps({k: v for k, v in r.items() if k != "top_confusions(true,pred,count)"}, indent=1))
        plot_curves({k: hists[k] for k in ("mobilenet_v2", "squeezenet1_1")},
                    os.path.join(args.out, "fig_loss_curves_sota.png"), "SOTA fine-tuning: loss curves")

    # ---- save everything ----
    with open(os.path.join(args.out, "history.json"), "w") as f:
        json.dump({k: {kk: vv for kk, vv in v.items() if not kk.startswith("_")} for k, v in hists.items()}, f)
    if not args.skip_opt:
        with open(os.path.join(args.out, "history_optimizers.json"), "w") as f:
            json.dump({k: {kk: vv for kk, vv in v.items() if not kk.startswith("_")} for k, v in opt_hists.items()}, f)
    with open(os.path.join(args.out, "results.json"), "w") as f:
        json.dump(results, f, indent=2)
    write_tables(results, args)
    print(f"\nDone. See {args.out}/results_tables.md, results.json and the fig_*.png files.")


def write_tables(R, args):
    L = []
    A, B = R["ModelA"], R["ModelB"]
    L.append("## Table 1 - Model A vs Model B (test set)\n")
    L.append("| Metric | Model A (standard) | Model B (depthwise-separable) |\n|---|---|---|")
    rows = [("Trainable parameters", f"{A['params']:,}", f"{B['params']:,}"),
            ("MACs / image (M)", f"{A['macs'] / 1e6:.2f}", f"{B['macs'] / 1e6:.2f}"),
            ("Model size on disk, fp32 (KB)", f"{A['size_kb_fp32_file']:.1f}", f"{B['size_kb_fp32_file']:.1f}"),
            ("Estimated size, int8 (KB)", f"{A['size_kb_int8_estimate']:.1f}", f"{B['size_kb_int8_estimate']:.1f}"),
            ("Training time / epoch (s)", f"{A['epoch_time_s']:.1f}", f"{B['epoch_time_s']:.1f}"),
            ("CPU latency, 1 thread (ms)", f"{A['cpu_latency_ms']:.2f}", f"{B['cpu_latency_ms']:.2f}"),
            ("Test accuracy", f"{A['test_acc']:.4f}", f"{B['test_acc']:.4f}"),
            ("Precision (macro)", f"{A['precision_macro']:.4f}", f"{B['precision_macro']:.4f}"),
            ("Recall (macro)", f"{A['recall_macro']:.4f}", f"{B['recall_macro']:.4f}"),
            ("F1 (macro)", f"{A['f1_macro']:.4f}", f"{B['f1_macro']:.4f}")]
    L += [f"| {a} | {b} | {c} |" for a, b, c in rows]
    if "optimizer_study" in R:
        L.append("\n## Table 2 - Optimizer comparison on Model B\n")
        L.append("| Optimizer | Best val acc | Final val acc | Min val loss | Epochs to 90% val acc | Test acc |\n|---|---|---|---|---|---|")
        for k, s in R["optimizer_study"].items():
            L.append(f"| {k} | {s['best_val_acc']:.4f} | {s['final_val_acc']:.4f} | {s['min_val_loss']:.4f} | "
                     f"{s['epochs_to_90pct_val']} | {s['test_acc']:.4f} |")
    if "mobilenet_v2" in R:
        L.append("\n## Table 3 - Custom Model B vs fine-tuned lightweight SOTA (test set)\n")
        L.append("| Metric | Model B | MobileNetV2 | SqueezeNet1.1 |\n|---|---|---|---|")
        names = ["ModelB", "mobilenet_v2", "squeezenet1_1"]
        spec = [("Trainable parameters", lambda r: f"{r['params']:,}"),
                ("Model size fp32 (MB)", lambda r: f"{r['size_kb_fp32_file'] / 1024:.3f}"),
                ("Estimated size int8 (MB)", lambda r: f"{r['size_kb_int8_estimate'] / 1024:.3f}"),
                ("MACs / image (M)", lambda r: f"{r['macs'] / 1e6:.2f}"),
                ("CPU latency, 1 thread (ms)", lambda r: f"{r['cpu_latency_ms']:.2f}"),
                ("Training time / epoch (s)", lambda r: f"{r['epoch_time_s']:.1f}"),
                ("Test accuracy", lambda r: f"{r['test_acc']:.4f}"),
                ("Precision (macro)", lambda r: f"{r['precision_macro']:.4f}"),
                ("Recall (macro)", lambda r: f"{r['recall_macro']:.4f}"),
                ("F1 (macro)", lambda r: f"{r['f1_macro']:.4f}")]
        for lab, fn in spec:
            L.append(f"| {lab} | " + " | ".join(fn(R[n]) for n in names) + " |")
    with open(os.path.join(args.out, "results_tables.md"), "w") as f:
        f.write("\n".join(L) + "\n")
    print("\n".join(L))


if __name__ == "__main__":
    main()
