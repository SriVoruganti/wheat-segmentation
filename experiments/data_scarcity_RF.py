"""
Data Scarcity Experiment — Random Forest (Combined Final).

Trains Random Forest pixel classifiers with varying fractions of the TRAIN set,
evaluates on the VAL set, and plots performance curves.

Supports:
- Multiple feature modes (rgb, rgb_exg, rgb_hsv, rgb_hsv_exg)
- Custom training fractions
- Optional label-noise injection (flips a fraction of mask pixels during training)
- Saves per-feature-mode JSON + PNG, and an optional combined comparison plot.

Usage examples:
    python experiments/data_scarcity_RF.py
    python experiments/data_scarcity_RF.py --feature_mode rgb_hsv_exg
    python experiments/data_scarcity_RF.py --feature_mode all
    python experiments/data_scarcity_RF.py --feature_mode all --label_noise 0.1
    python experiments/data_scarcity_RF.py --fractions 0.01 0.02 0.05 0.10 0.25 0.50 1.0
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


# Paths

DATA_ROOT = os.path.join(os.path.dirname(__file__), "..", "data", "EWS-Dataset")
OUTPUT_DIR = os.path.join(os.path.dirname(__file__), "..", "results", "data_scarcity")


# Args

FEATURE_MODES = ["rgb", "rgb_exg", "rgb_hsv", "rgb_hsv_exg"]


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument(
        "--feature_mode",
        type=str,
        default="rgb",
        choices=FEATURE_MODES + ["all"],
        help="Pixel feature representation to use, or 'all' to run all modes",
    )
    p.add_argument(
        "--label_noise",
        type=float,
        default=0.0,
        help="Fraction of mask pixels to flip during training (simulates noisy annotation)",
    )
    p.add_argument("--max_pixels_per_image", type=int, default=5000)
    p.add_argument("--seed", type=int, default=42)
    p.add_argument(
        "--fractions",
        type=float,
        nargs="+",
        default=[0.25, 0.50, 0.75, 1.00],
        help="Training fractions to evaluate, e.g. --fractions 0.01 0.05 0.10 0.25",
    )
    p.add_argument(
        "--no_plots",
        action="store_true",
        help="Skip saving plots (still saves JSON results).",
    )
    return p.parse_args()


# Core experiment

def _subset_dataset_inplace(ds, frac, seed):
    """Subsample the dataset file lists in-place to a fraction of images."""
    rng = random.Random(seed)
    n_total = len(ds.image_files)
    n = max(1, int(n_total * frac))
    idx = sorted(rng.sample(range(n_total), n))
    ds.image_files = [ds.image_files[i] for i in idx]
    ds.mask_files  = [ds.mask_files[i]  for i in idx]


def run_for_feature_mode(args, feature_mode, X_val, y_val, fractions):
    """Run scarcity experiment for a single feature mode; returns results dict."""
    results = {}

    for frac in fractions:
        label = f"{int(round(frac * 100))}%"
        print(f"\n{'='*60}")
        print(f"[{feature_mode}] Training with {label} of TRAIN (label_noise={args.label_noise})")

        train_ds = EWSDatasetRF(
            root=DATA_ROOT,
            split="train",  # IMPORTANT: train on train split
            max_pixels_per_image=args.max_pixels_per_image,
            label_noise=args.label_noise,
            seed=args.seed,
            feature_mode=feature_mode,
        )

        _subset_dataset_inplace(train_ds, frac, args.seed)

        X_train, y_train = train_ds.load()
        print(f"  Training samples (pixels): {X_train.shape[0]:,}")

        # Train
        t0 = time.time()
        clf = build_model()
        clf.fit(X_train, y_train)
        elapsed = time.time() - t0

        # Eval on val
        y_pred = clf.predict(X_val)
        metrics = {
            "f1": float(f1_score(y_val, y_pred)),
            "iou": float(jaccard_score(y_val, y_pred)),
        }

        results[label] = {
            **metrics,
            "train_samples": int(X_train.shape[0]),
            "time_s": round(elapsed, 1),
        }

        print(f"  IoU={metrics['iou']:.4f}  F1={metrics['f1']:.4f}  ({elapsed:.1f}s)")

    return results


# Plotting

def plot_single_mode(results, feature_mode, label_noise, out_path):
    import matplotlib.pyplot as plt

    labels = list(results.keys())
    ious = [results[l]["iou"] for l in labels]
    f1s  = [results[l]["f1"]  for l in labels]
    x = range(len(labels))

    fig, ax = plt.subplots(figsize=(9, 5))
    ax.plot(x, ious, "o-", linewidth=2, markersize=7, label="IoU")
    ax.plot(x, f1s,  "s-", linewidth=2, markersize=7, label="F1")

    ax.set_xticks(list(x))
    ax.set_xticklabels(labels, fontsize=11)
    ax.set_xlabel("Training Data Used")
    ax.set_ylabel("Score")
    ax.set_ylim(0, 1.05)

    title = f"Data Scarcity — RF ({feature_mode})"
    if label_noise > 0:
        title += f" (label noise={label_noise})"
    ax.set_title(title)
    ax.grid(alpha=0.3)
    ax.legend()
    plt.tight_layout()
    plt.savefig(out_path, dpi=150, bbox_inches="tight")
    plt.close(fig)


def plot_compare_modes(all_mode_results, fractions, label_noise, out_path):
    import matplotlib.pyplot as plt

    # Use IoU curves for comparison (cleaner); you can add F1 similarly if you want.
    labels = [f"{int(round(f * 100))}%" for f in fractions]
    x = range(len(labels))

    fig, ax = plt.subplots(figsize=(10, 5))
    for mode, results in all_mode_results.items():
        ious = [results[l]["iou"] for l in labels]
        ax.plot(x, ious, marker="o", linewidth=2, label=mode)

    ax.set_xticks(list(x))
    ax.set_xticklabels(labels, fontsize=11)
    ax.set_xlabel("Training Data Used")
    ax.set_ylabel("IoU")
    ax.set_ylim(0, 1.05)

    title = "Data Scarcity — RF (IoU comparison)"
    if label_noise > 0:
        title += f" (label noise={label_noise})"
    ax.set_title(title)
    ax.grid(alpha=0.3)
    ax.legend()
    plt.tight_layout()
    plt.savefig(out_path, dpi=150, bbox_inches="tight")
    plt.close(fig)


# Main

def main():
    args = parse_args()
    os.makedirs(OUTPUT_DIR, exist_ok=True)

    fractions = sorted(set(args.fractions))
    if fractions[0] <= 0 or fractions[-1] > 1.0:
        raise ValueError("fractions must be in (0, 1]. Example: 0.25 0.5 0.75 1.0")

    modes = FEATURE_MODES if args.feature_mode == "all" else [args.feature_mode]

    print(f"DATA_ROOT   : {DATA_ROOT}")
    print(f"OUTPUT_DIR  : {OUTPUT_DIR}")
    print(f"Fractions   : {fractions}")
    print(f"Modes       : {modes}")
    print(f"Label noise : {args.label_noise}")

    # Load val once (per mode, because feature_mode changes X)
    all_mode_results = {}

    for mode in modes:
        print(f"\n{'#'*70}")
        print(f"Running feature_mode={mode}")

        print("Loading validation data...")
        val_ds = EWSDatasetRF(
            root=DATA_ROOT,
            split="val",
            max_pixels_per_image=args.max_pixels_per_image,
            feature_mode=mode,
        )
        X_val, y_val = val_ds.load()

        results = run_for_feature_mode(args, mode, X_val, y_val, fractions)
        all_mode_results[mode] = results

        # Save JSON per mode
        json_path = os.path.join(OUTPUT_DIR, f"data_scarcity_RF_{mode}.json")
        payload = {
            "args": {**vars(args), "feature_mode": mode, "fractions": fractions},
            "results": results,
        }
        with open(json_path, "w") as f:
            json.dump(payload, f, indent=2)
        print(f"\nSaved JSON → {json_path}")

        # Plot per mode
        if not args.no_plots:
            fig_path = os.path.join(OUTPUT_DIR, f"data_scarcity_RF_{mode}.png")
            plot_single_mode(results, mode, args.label_noise, fig_path)
            print(f"Saved plot → {fig_path}")

    # Combined comparison plot (only if multiple modes)
    if (not args.no_plots) and len(modes) > 1:
        comp_path = os.path.join(OUTPUT_DIR, "data_scarcity_RF_compare_iou.png")
        plot_compare_modes(all_mode_results, fractions, args.label_noise, comp_path)
        print(f"\nSaved comparison plot → {comp_path}")


if __name__ == "__main__":
    main()
