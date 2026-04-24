"""
Save RF qualitative panels: worst / median / best (by IoU), plus a combined image.

PowerShell example:
  python -m scripts.rf_save_panels --data_root .\data\EWS-Dataset --split test `
    --model_path .\results\rf_model_rgb_hsv_exg.pkl --feature_mode rgb_hsv_exg `
    --out_dir .\results\rf_full\panels
"""

import argparse
from pathlib import Path

import numpy as np
import joblib

from scripts.rf_common import (
    load_pairs_ews,
    read_rgb,
    read_mask_ews,
    iou,
    segment_rf,
    save_panel,
    combine_panels,
)


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--data_root", type=str, default="./data/EWS-Dataset")
    p.add_argument("--split", type=str, default="test")
    p.add_argument("--model_path", type=str, default="./results/rf_model_rgb_hsv_exg.pkl")
    p.add_argument("--feature_mode", type=str, default="rgb_hsv_exg")
    p.add_argument("--out_dir", type=str, default="./results/rf_full/panels")
    return p.parse_args()


def main():
    args = parse_args()

    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    # Load model
    model_path = Path(args.model_path)
    if not model_path.exists():
        raise FileNotFoundError(f"Model not found: {model_path}")
    rf = joblib.load(str(model_path))

    # Load (image, mask) pairs
    pairs, missing = load_pairs_ews(args.data_root, split=args.split)
    print(f"Split={args.split} pairs={len(pairs)} missing_masks={missing}")
    if len(pairs) == 0:
        raise RuntimeError("No pairs found. Check --data_root and dataset folder structure.")

    # Score each image by IoU
    scores = []
    cache = []  # (img_path, mask_path, pred01, iou)
    for img_path, mask_path in pairs:
        img = read_rgb(img_path)
        gt = read_mask_ews(mask_path)
        pred = segment_rf(img, rf, feature_mode=args.feature_mode)
        score = iou(pred, gt)

        scores.append(score)
        cache.append((img_path, mask_path, pred, score))

    scores = np.asarray(scores, dtype=np.float64)
    order = np.argsort(scores)

    picks = [order[0], order[len(order) // 2], order[-1]]
    tags = ["worst", "median", "best"]

    panel_paths = []
    for idx, tag in zip(picks, tags):
        img_path, mask_path, pred, score = cache[idx]
        img = read_rgb(img_path)
        gt = read_mask_ews(mask_path)

        stem = Path(img_path).stem
        out_path = out_dir / f"{tag}_{stem}_iou_{score:.3f}.png"

        save_panel(img, gt, pred, str(out_path), title=f"{tag} example — IoU={score:.3f}")
        panel_paths.append(str(out_path))

    combined_out = out_dir / "rf_examples_combined.png"
    combine_panels(panel_paths, str(combined_out), layout="vertical")

    print("Saved panels to:", out_dir)
    print("Saved combined panel to:", combined_out)


if __name__ == "__main__":
    main()
