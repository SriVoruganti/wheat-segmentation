"""
RF Feature Ablation Experiment (EWS Wheat Segmentation).

Trains a Random Forest pixel classifier using different handcrafted feature sets
and evaluates on a chosen split (default: val). This is used to quantify how much
HSV and ExG help compared to RGB alone.

Feature modes:
- rgb (3D): R,G,B
- rgb_exg (4D): RGB + Excess Green (ExG = 2G - R - B)
- rgb_hsv (6D): RGB + HSV
- rgb_hsv_exg (7D): RGB + HSV + ExG

Outputs:
- JSON: results/rf_full/ablation/rf_feature_ablation_<split>.json
- Plot: results/rf_full/ablation/rf_feature_ablation_<split>.png
- Prints a Markdown table to paste into the report.

Usage (PowerShell):
    python .\experiments\rf_feature_ablation.py
    python .\experiments\rf_feature_ablation.py --split val
    python .\experiments\rf_feature_ablation.py --split test
    python .\experiments\rf_feature_ablation.py --n_estimators 200 --max_pixels_per_image 5000
    python .\experiments\rf_feature_ablation.py --no_plots
"""

import os
import sys
import json
import time
import argparse
from pathlib import Path

import numpy as np
import cv2

# Allow imports from repo root when running as a script
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from sklearn.ensemble import RandomForestClassifier
from data.dataset import EWSDatasetRF


# Args

FEATURE_MODES = ["rgb", "rgb_exg", "rgb_hsv", "rgb_hsv_exg"]


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument(
        "--data_root",
        type=str,
        default=os.path.join("data", "EWS-Dataset"),
        help="Path to EWS-Dataset root (contains train/val/test).",
    )
    p.add_argument(
        "--split",
        type=str,
        default="val",
        choices=["train", "val", "test"],
        help="Split to evaluate on. Use val while iterating; test only for final reporting.",
    )
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--max_pixels_per_image", type=int, default=5000)
    p.add_argument("--n_estimators", type=int, default=200)
    p.add_argument("--max_depth", type=int, default=None)
    p.add_argument("--n_jobs", type=int, default=-1)
    p.add_argument(
        "--out_dir",
        type=str,
        default=os.path.join("results", "rf_full", "ablation"),
        help="Output directory for JSON + plots.",
    )
    p.add_argument("--no_plots", action="store_true", help="Skip saving plots (still saves JSON).")
    return p.parse_args()


# IO helpers

def load_image_rgb_u8(p: Path) -> np.ndarray:
    bgr = cv2.imread(str(p), cv2.IMREAD_COLOR)
    if bgr is None:
        raise FileNotFoundError(f"Could not read image: {p}")
    return cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB)


def load_gt_mask01(p: Path) -> np.ndarray:
    m = cv2.imread(str(p), cv2.IMREAD_UNCHANGED)
    if m is None:
        raise FileNotFoundError(f"Could not read mask: {p}")
    if m.ndim == 3:
        m = m[:, :, 0]
    return (m == 0).astype(np.uint8)  # plant=1 where ch0==0


# Features

def build_features(img_rgb_u8: np.ndarray, feature_mode: str):
    """
    Returns:
        X: (H*W, C) float32
        (H, W)
    """
    rgb = (img_rgb_u8.astype(np.float32) / 255.0)
    feats = [rgb]

    if feature_mode in ("rgb_exg", "rgb_hsv_exg"):
        R, G, B = rgb[:, :, 0], rgb[:, :, 1], rgb[:, :, 2]
        exg = (2.0 * G - R - B)[:, :, None].astype(np.float32)
        feats.append(exg)

    if feature_mode in ("rgb_hsv", "rgb_hsv_exg"):
        hsv = cv2.cvtColor(img_rgb_u8, cv2.COLOR_RGB2HSV).astype(np.float32)
        hsv[:, :, 0] /= 179.0
        hsv[:, :, 1] /= 255.0
        hsv[:, :, 2] /= 255.0
        feats.append(hsv)

    feat_img = np.concatenate(feats, axis=2)
    H, W, C = feat_img.shape
    X = feat_img.reshape(-1, C).astype(np.float32)
    return X, (H, W)


# Metrics

def metrics_from_masks(pred01: np.ndarray, gt01: np.ndarray):
    pred = pred01.astype(bool)
    gt = gt01.astype(bool)

    tp = np.logical_and(pred, gt).sum()
    fp = np.logical_and(pred, ~gt).sum()
    fn = np.logical_and(~pred, gt).sum()

    precision = tp / (tp + fp + 1e-8)
    recall = tp / (tp + fn + 1e-8)
    f1 = 2 * precision * recall / (precision + recall + 1e-8)
    iou = tp / (tp + fp + fn + 1e-8)
    return float(precision), float(recall), float(f1), float(iou)


def eval_split_full_images(clf, feature_mode: str, data_root: Path, split: str):
    img_dir = data_root / split / "images"
    msk_dir = data_root / split / "masks"

    files = sorted([p.name for p in img_dir.glob("*.png")])
    if len(files) == 0:
        raise FileNotFoundError(f"No .png images found in {img_dir}")

    Ps, Rs, F1s, IoUs, times = [], [], [], [], []

    for img_file in files:
        stem = Path(img_file).stem
        img = load_image_rgb_u8(img_dir / img_file)
        gt = load_gt_mask01(msk_dir / f"{stem}_mask.png")

        X, (H, W) = build_features(img, feature_mode)

        t0 = time.perf_counter()
        pred = clf.predict(X).reshape(H, W).astype(np.uint8)
        t1 = time.perf_counter()
        times.append((t1 - t0) * 1000.0)

        p, r, f1, iou = metrics_from_masks(pred, gt)
        Ps.append(p)
        Rs.append(r)
        F1s.append(f1)
        IoUs.append(iou)

    return {
        "precision": float(np.mean(Ps)),
        "recall": float(np.mean(Rs)),
        "f1": float(np.mean(F1s)),
        "iou": float(np.mean(IoUs)),
        "ms_img": float(np.mean(times)),
        "n_images": int(len(files)),
    }


# Plot

def save_plot(rows, out_path: Path, split: str):
    import matplotlib.pyplot as plt

    labels = [r["model"] for r in rows]
    ious = [r["iou"] for r in rows]
    f1s = [r["f1"] for r in rows]
    x = np.arange(len(labels))

    plt.figure(figsize=(9, 4))
    plt.plot(x, ious, marker="o", label="IoU")
    plt.plot(x, f1s, marker="s", label="F1-Score")
    plt.xticks(x, labels, rotation=15, ha="right")
    plt.ylim(0, 1)
    plt.ylabel("Score")
    plt.title(f"Handcrafted Feature Ablation — Random Forest ({split})")
    plt.grid(True, alpha=0.3)
    plt.legend()
    plt.tight_layout()
    plt.savefig(out_path, dpi=200, bbox_inches="tight")
    plt.close()


# Main

def main():
    args = parse_args()

    data_root = Path(args.data_root)
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    configs = [
        ("RF (RGB)", "rgb"),
        ("RF (RGB+ExG)", "rgb_exg"),
        ("RF (RGB+HSV)", "rgb_hsv"),
        ("RF (RGB+HSV+ExG)", "rgb_hsv_exg"),
    ]

    rows = []
    for label, mode in configs:
        print(f"\n=== {label} ===")

        # Train on TRAIN split (pixel subsampling handled by EWSDatasetRF)
        ds = EWSDatasetRF(
            root=str(data_root),
            split="train",
            max_pixels_per_image=args.max_pixels_per_image,
            seed=args.seed,
            feature_mode=mode,
        )
        X_train, y_train = ds.load()
        print(f"Training pixels: {X_train.shape[0]:,} | feature_mode={mode}")

        clf = RandomForestClassifier(
            n_estimators=args.n_estimators,
            max_depth=args.max_depth,
            n_jobs=args.n_jobs,
            random_state=args.seed,
            class_weight="balanced",
        )

        t0 = time.perf_counter()
        clf.fit(X_train, y_train)
        t1 = time.perf_counter()

        m = eval_split_full_images(clf, mode, data_root=data_root, split=args.split)
        m["model"] = label
        m["feature_mode"] = mode
        m["train_time_s"] = float(t1 - t0)
        rows.append(m)

        print(f"IoU {m['iou']:.4f} | F1 {m['f1']:.4f} | ms/img {m['ms_img']:.1f} | train {m['train_time_s']:.1f}s")

    # Save JSON
    json_path = out_dir / f"rf_feature_ablation_{args.split}.json"
    with open(json_path, "w") as f:
        json.dump({"args": vars(args), "results": rows}, f, indent=2)
    print("\nSaved:", json_path)

    # Print markdown table
    print("\n| Model | Precision | Recall | F1 | IoU | ms/img |")
    print("|---|---:|---:|---:|---:|---:|")
    for r in rows:
        print(
            f"| {r['model']} | {r['precision']:.3f} | {r['recall']:.3f} | "
            f"{r['f1']:.3f} | {r['iou']:.3f} | {r['ms_img']:.1f} |"
        )

    # Plot
    if not args.no_plots:
        fig_path = out_dir / f"rf_feature_ablation_{args.split}.png"
        save_plot(rows, fig_path, split=args.split)
        print("Saved:", fig_path)


if __name__ == "__main__":
    main()
