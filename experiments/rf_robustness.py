"""
RF Robustness Experiment (EWS Wheat Segmentation).

Evaluates a trained Random Forest pixel classifier under common image distortions
(contrast/brightness changes, noise, blur, occlusion, JPEG compression) and reports
mean IoU/F1 over a chosen dataset split.

This script is intended as an *analysis* experiment (not training). It loads a saved
RF model (.pkl), applies distortions to each image, predicts a mask, and compares
to the ground-truth mask.

Outputs:
- JSON metrics: results/rf_full/robustness/rf_robustness_<split>_<feature_mode>.json
- Plot (IoU bar chart): results/rf_full/robustness/rf_robustness_<split>_<feature_mode>.png

Usage (PowerShell):
    python .\experiments\rf_robustness.py `
      --model_path .\results\rf_full\rf_model_rgb_hsv_exg.pkl `
      --feature_mode rgb_hsv_exg `
      --split val

    python .\experiments\rf_robustness.py --split test --seed 0
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

from models.random_forest import load_model


# Args

FEATURE_MODES = ["rgb", "rgb_exg", "rgb_hsv", "rgb_hsv_exg"]


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument(
        "--data_root",
        type=str,
        default=os.path.join("data", "EWS-Dataset"),
        help="Path to EWS-Dataset root (contains train/val/test folders).",
    )
    p.add_argument(
        "--split",
        type=str,
        default="val",
        choices=["train", "val", "test"],
        help="Which split to evaluate on. Use val while iterating; test only for final reporting.",
    )
    p.add_argument(
        "--model_path",
        type=str,
        default=os.path.join("results", "rf_full", "rf_model_rgb_hsv_exg.pkl"),
        help="Path to trained RF model .pkl",
    )
    p.add_argument(
        "--feature_mode",
        type=str,
        default="rgb_hsv_exg",
        choices=FEATURE_MODES,
        help="Feature mode used to train the model (must match at inference).",
    )
    p.add_argument(
        "--out_dir",
        type=str,
        default=os.path.join("results", "rf_full", "robustness"),
        help="Output directory for JSON + plots.",
    )
    p.add_argument("--seed", type=int, default=0, help="Seed controlling stochastic distortions.")
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


# Feature extraction (kept consistent with your RF feature modes)

def build_features(img_rgb_u8: np.ndarray, feature_mode: str):
    """
    Returns:
        X: (H*W, C) float32
        (H, W)
    """
    rgb = (img_rgb_u8.astype(np.float32) / 255.0)
    feats = [rgb]  # always include RGB

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


def predict_mask01(clf, img_rgb_u8: np.ndarray, feature_mode: str) -> np.ndarray:
    X, (H, W) = build_features(img_rgb_u8, feature_mode)
    pred = clf.predict(X).reshape(H, W).astype(np.uint8)
    return pred


# Metrics

def metrics_iou_f1(gt01: np.ndarray, pred01: np.ndarray):
    gt = gt01.astype(bool)
    pr = pred01.astype(bool)

    tp = np.logical_and(pr, gt).sum()
    fp = np.logical_and(pr, ~gt).sum()
    fn = np.logical_and(~pr, gt).sum()

    iou = tp / (tp + fp + fn + 1e-8)
    precision = tp / (tp + fp + 1e-8)
    recall = tp / (tp + fn + 1e-8)
    f1 = 2 * precision * recall / (precision + recall + 1e-8)
    return float(iou), float(f1)


# Distortions

def clip_u8(x):
    return np.clip(x, 0, 255).astype(np.uint8)


def low_contrast(img, alpha=0.6):
    # pull values toward mid-gray
    return clip_u8(128 + alpha * (img.astype(np.float32) - 128))


def low_brightness(img, beta=-40):
    return clip_u8(img.astype(np.int16) + int(beta))


def gaussian_noise(img, sigma=10, rng=None):
    if rng is None:
        rng = np.random.default_rng(0)
    noise = rng.normal(0, sigma, img.shape).astype(np.float32)
    return clip_u8(img.astype(np.float32) + noise)


def blur(img, k=5):
    k = int(k)
    if k % 2 == 0:
        k += 1
    return cv2.GaussianBlur(img, (k, k), 0)


def occlusion(img, n=8, size=30, rng=None):
    if rng is None:
        rng = np.random.default_rng(0)
    out = img.copy()
    H, W, _ = out.shape
    size = int(size)
    for _ in range(int(n)):
        y = int(rng.integers(0, max(1, H - size)))
        x = int(rng.integers(0, max(1, W - size)))
        out[y : y + size, x : x + size] = 0
    return out


def jpeg_compress(img, quality=20):
    encode_param = [int(cv2.IMWRITE_JPEG_QUALITY), int(quality)]
    ok, enc = cv2.imencode(".jpg", cv2.cvtColor(img, cv2.COLOR_RGB2BGR), encode_param)
    if not ok:
        return img
    dec = cv2.imdecode(enc, cv2.IMREAD_COLOR)
    return cv2.cvtColor(dec, cv2.COLOR_BGR2RGB)


def make_distortions(seed=0):
    rng = np.random.default_rng(seed)
    return [
        ("clean", lambda x: x),
        ("low_contrast", lambda x: low_contrast(x, alpha=0.6)),
        ("low_brightness", lambda x: low_brightness(x, beta=-40)),
        ("gaussian_noise_mild", lambda x: gaussian_noise(x, sigma=10, rng=rng)),
        ("gaussian_noise_strong", lambda x: gaussian_noise(x, sigma=25, rng=rng)),
        ("occlusion", lambda x: occlusion(x, n=8, size=30, rng=rng)),
        ("blur_mild", lambda x: blur(x, k=5)),
        ("blur_strong", lambda x: blur(x, k=11)),
        ("jpeg_compression", lambda x: jpeg_compress(x, quality=20)),
    ]


# Plot

def save_plot(results, out_path: Path, title: str):
    import matplotlib.pyplot as plt

    clean_iou = [r["iou"] for r in results if r["distortion"] == "clean"][0]
    labels = [r["distortion"] for r in results]
    vals = [r["iou"] for r in results]

    plt.figure(figsize=(10, 4.5))
    plt.bar(range(len(vals)), vals, color="#4C78A8")
    plt.axhline(clean_iou, linestyle="--", color="black", linewidth=1, label=f"Clean IoU ({clean_iou:.3f})")
    plt.ylim(0, 1)
    plt.ylabel("IoU")
    plt.title(title)
    plt.xticks(range(len(labels)), labels, rotation=25, ha="right")
    plt.legend()
    plt.tight_layout()
    plt.savefig(out_path, dpi=200)
    plt.close()


# Main

def main():
    args = parse_args()

    data_root = Path(args.data_root)
    img_dir = data_root / args.split / "images"
    msk_dir = data_root / args.split / "masks"

    if not img_dir.exists() or not msk_dir.exists():
        raise FileNotFoundError(f"Missing images/ or masks/ for split='{args.split}' at: {data_root}")

    image_files = sorted([p.name for p in img_dir.glob("*.png")])
    print(f"Found {len(image_files)} {args.split} images in {img_dir}")
    if len(image_files) == 0:
        raise FileNotFoundError(f"No .png images found in {img_dir}")

    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    model_path = Path(args.model_path)
    if not model_path.exists():
        raise FileNotFoundError(f"Model not found: {model_path}")

    clf = load_model(str(model_path))

    # Optional: sanity check expected feature dimension
    if hasattr(clf, "n_features_in_"):
        expected = int(clf.n_features_in_)
        mode_dims = {"rgb": 3, "rgb_exg": 4, "rgb_hsv": 6, "rgb_hsv_exg": 7}
        got = mode_dims.get(args.feature_mode, None)
        print("Model expects n_features_in_ =", expected)
        if got is not None and got != expected:
            print(
                f"WARNING: feature_mode='{args.feature_mode}' gives {got} dims, "
                f"but model expects {expected}. Update --feature_mode or retrain the model."
            )

    distortions = make_distortions(seed=args.seed)

    results = []
    for name, fn in distortions:
        ious, f1s = [], []
        t0 = time.perf_counter()

        for img_file in image_files:
            stem = Path(img_file).stem
            img = load_image_rgb_u8(img_dir / img_file)
            gt = load_gt_mask01(msk_dir / f"{stem}_mask.png")

            img_d = fn(img)
            pred = predict_mask01(clf, img_d, feature_mode=args.feature_mode)

            iou_v, f1_v = metrics_iou_f1(gt, pred)
            ious.append(iou_v)
            f1s.append(f1_v)

        t1 = time.perf_counter()

        row = {
            "distortion": name,
            "iou": float(np.mean(ious)),
            "f1": float(np.mean(f1s)),
            "std_iou": float(np.std(ious)),
            "ms_img": float((t1 - t0) * 1000 / len(image_files)),
            "feature_mode": args.feature_mode,
            "split": args.split,
        }
        results.append(row)
        print(f"{name:>18s} | IoU {row['iou']:.4f} | F1 {row['f1']:.4f} | ms/img {row['ms_img']:.1f}")

    # Save JSON
    json_path = out_dir / f"rf_robustness_{args.split}_{args.feature_mode}.json"
    with open(json_path, "w") as f:
        json.dump({"args": vars(args), "results": results}, f, indent=2)
    print("Saved:", json_path)

    # Plot
    if not args.no_plots:
        fig_path = out_dir / f"rf_robustness_{args.split}_{args.feature_mode}.png"
        title = f"RF Robustness — {args.split} ({args.feature_mode})"
        save_plot(results, fig_path, title)
        print("Saved:", fig_path)


if __name__ == "__main__":
    main()
