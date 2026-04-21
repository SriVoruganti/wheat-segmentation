"""
Data Scarcity Experiment — Random Forest.

Trains the model with varying fractions of the training set
and plots how performance changes across feature modes.

Usage:
    python experiments/data_scarcity_RF.py
    python experiments/data_scarcity_RF.py --feature_mode rgb_hsv_exg
    python experiments/data_scarcity_RF.py --feature_mode rgb_hsv_exg --label_noise 0.1
    python experiments/data_scarcity_RF.py --fractions 0.01 0.02 0.05 0.10 0.15 0.20 0.25
"""

import os
import sys
import json
import argparse
import time
import random

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from data.dataset import EWSDatasetRF
from models.random_forest import build_model
from sklearn.metrics import f1_score, jaccard_score

# ------------------------------------------------------------------
# Paths
# ------------------------------------------------------------------

DATA_ROOT  = os.path.join(os.path.dirname(__file__), "..", "data", "EWS-Dataset")
OUTPUT_DIR = os.path.join(os.path.dirname(__file__), "..", "results", "data_scarcity")


# ------------------------------------------------------------------
# Args
# ------------------------------------------------------------------

def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--feature_mode",         type=str,   default="rgb",
                        choices=["rgb", "rgb_exg", "rgb_hsv", "rgb_hsv_exg"],
                        help="Pixel feature representation to use")
    parser.add_argument("--label_noise",          type=float, default=0.0,
                        help="Fraction of mask pixels to flip (simulates noisy annotation)")
    parser.add_argument("--max_pixels_per_image", type=int,   default=5000)
    parser.add_argument("--seed",                 type=int,   default=42)
    parser.add_argument("--fractions",            type=float, nargs="+",
                        default=[0.25, 0.50, 0.75, 1.00],
                        help="Training fractions to evaluate, e.g. --fractions 0.01 0.05 0.10 0.25")
    return parser.parse_args()


# ------------------------------------------------------------------
# Main
# ------------------------------------------------------------------

def main():
    args = parse_args()
    os.makedirs(OUTPUT_DIR, exist_ok=True)

    fractions   = sorted(args.fractions)
    all_results = {}

    print(f"Feature mode : {args.feature_mode}")
    print(f"Fractions    : {fractions}")

    # Load val set once — reused across all fractions
    print("\nLoading validation data...")
    val_ds = EWSDatasetRF(
        root=DATA_ROOT,
        split="val",
        max_pixels_per_image=args.max_pixels_per_image,
        feature_mode=args.feature_mode,
    )
    X_val, y_val = val_ds.load()

    for frac in fractions:
        label = f"{int(frac * 100)}%"
        print(f"\n{'='*50}")
        print(f"Training with {label} of data (feature_mode={args.feature_mode}, label_noise={args.label_noise})")

        train_ds = EWSDatasetRF(
            root=DATA_ROOT,
            split="train",
            max_pixels_per_image=args.max_pixels_per_image,
            label_noise=args.label_noise,
            seed=args.seed,
            feature_mode=args.feature_mode,
        )

        # Manually apply subset
        rng = random.Random(args.seed)
        n   = max(1, int(len(train_ds.image_files) * frac))
        idx = sorted(rng.sample(range(len(train_ds.image_files)), n))
        train_ds.image_files = [train_ds.image_files[i] for i in idx]
        train_ds.mask_files  = [train_ds.mask_files[i]  for i in idx]

        X_train, y_train = train_ds.load()
        print(f"  Training samples : {X_train.shape[0]:,}")

        # Train
        t0  = time.time()
        clf = build_model()
        clf.fit(X_train, y_train)
        elapsed = time.time() - t0

        # Evaluate on val set
        y_pred = clf.predict(X_val)
        metrics = {
            "f1":  f1_score(y_val, y_pred),
            "iou": jaccard_score(y_val, y_pred),
        }

        all_results[label] = {
            **metrics,
            "train_samples": X_train.shape[0],
            "time_s":        round(elapsed, 1),
        }
        print(f"  IoU={metrics['iou']:.4f}  F1={metrics['f1']:.4f}  ({elapsed:.1f}s)")

    # ------------------------------------------------------------------
    # Save JSON — filename includes feature mode
    # ------------------------------------------------------------------
    json_path = os.path.join(OUTPUT_DIR, f"data_scarcity_RF_{args.feature_mode}.json")
    with open(json_path, "w") as f:
        json.dump({"args": vars(args), "results": all_results}, f, indent=2)
    print(f"\nResults saved → {json_path}")

    # ------------------------------------------------------------------
    # Plot
    # ------------------------------------------------------------------
    try:
        import matplotlib.pyplot as plt

        labels = list(all_results.keys())
        ious   = [all_results[l]["iou"] for l in labels]
        f1s    = [all_results[l]["f1"]  for l in labels]
        x      = range(len(labels))

        fig, ax = plt.subplots(figsize=(9, 5))
        ax.plot(x, ious, "o-", color="#2196F3", linewidth=2, markersize=8, label="IoU")
        ax.plot(x, f1s,  "s-", color="#4CAF50", linewidth=2, markersize=8, label="F1-Score")

        for xi, (iou, f1) in enumerate(zip(ious, f1s)):
            ax.annotate(f"{iou:.3f}", (xi, iou), textcoords="offset points",
                        xytext=(0, 10),  ha="center", fontsize=10, color="#2196F3")
            ax.annotate(f"{f1:.3f}",  (xi, f1),  textcoords="offset points",
                        xytext=(0, -15), ha="center", fontsize=10, color="#4CAF50")

        ax.set_xticks(x)
        ax.set_xticklabels(labels, fontsize=12)
        ax.set_xlabel("Training Data Used", fontsize=12)
        ax.set_ylabel("Score", fontsize=12)
        ax.set_ylim(0, 1.05)

        title = f"Data Scarcity Analysis — Random Forest ({args.feature_mode})"
        if args.label_noise > 0:
            title += f" (label noise={args.label_noise})"
        ax.set_title(title, fontsize=13, fontweight="bold")
        ax.legend(fontsize=11)
        ax.grid(alpha=0.3)
        plt.tight_layout()

        fig_path = os.path.join(OUTPUT_DIR, f"data_scarcity_RF_{args.feature_mode}.png")
        plt.savefig(fig_path, dpi=150, bbox_inches="tight")
        print(f"Plot saved → {fig_path}")

    except ImportError:
        print("matplotlib not installed — skipping plot.")


if __name__ == "__main__":
    main()
