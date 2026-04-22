"""
Compare the noise-robust superpixel variant against the baseline superpixel
method on both clean and noise-distorted EWS images.

Reports per-image IoU scores, and an average Precision / Recall / F1 / IoU
summary for each method-condition combination.

Usage
-----
From the project root:

    python -m experiments.noise_robustness_eval \\
        --indices 0 1 2 3 4 5 6 7 8 9 10 11 12
"""

import argparse
import sys
from pathlib import Path

import numpy as np

sys.path.append(str(Path(__file__).resolve().parents[1]))

from data.classical_loader import get_split_path, load_image_and_mask
from models.advanced_segmentation import (
    segment_wheat_superpixel,
    segment_wheat_superpixel_noise_robust,
)
from utils.classical_distortions import add_noise
from utils.classical_metrics import calculate_all_metrics


def evaluate_noise_robustness(folder_path, indices):
    """Per-image IoU table + full Precision/Recall/F1/IoU summary."""

    print("\n--- Per-image IoU values "
          "(baseline vs. noise-robust superpixel, on clean and noisy images) ---")
    print(f"{'img':<6} {'method':<28} {'Original':>10} {'Noise':>10}")
    print("=" * 60)

    # per-(method, condition): list of metric dicts, one per image
    metrics_orig_on_clean   = []   # baseline superpixel on clean image
    metrics_orig_on_noise   = []   # baseline superpixel on noisy image
    metrics_robust_on_clean = []   # noise-robust variant on clean image
    metrics_robust_on_noise = []   # noise-robust variant on noisy image

    for idx in indices:
        img_orig, true_mask, _ = load_image_and_mask(folder_path, idx)
        img_noise = add_noise(img_orig)

        m_o_clean = calculate_all_metrics(true_mask, segment_wheat_superpixel(img_orig))
        m_o_noise = calculate_all_metrics(true_mask, segment_wheat_superpixel(img_noise))
        m_r_clean = calculate_all_metrics(true_mask, segment_wheat_superpixel_noise_robust(img_orig))
        m_r_noise = calculate_all_metrics(true_mask, segment_wheat_superpixel_noise_robust(img_noise))

        metrics_orig_on_clean.append(m_o_clean)
        metrics_orig_on_noise.append(m_o_noise)
        metrics_robust_on_clean.append(m_r_clean)
        metrics_robust_on_noise.append(m_r_noise)

        print(f"{idx:<6} {'Superpixel':<28}"
              f"{m_o_clean['iou']:>10.4f}{m_o_noise['iou']:>10.4f}")
        print(f"{'':<6} {'Superpixel_noise_robust':<28}"
              f"{m_r_clean['iou']:>10.4f}{m_r_noise['iou']:>10.4f}")
        print("-" * 60)

    # --- IoU summary (compatible with original output) ---
    print("\n=== Average IoU values ===")
    print(f"{'AVG':<6} {'method':<28} {'Original':>10} {'Noise':>10}")
    print("-" * 60)

    avg_iou_orig_clean   = np.mean([m["iou"] for m in metrics_orig_on_clean])
    avg_iou_orig_noise   = np.mean([m["iou"] for m in metrics_orig_on_noise])
    avg_iou_robust_clean = np.mean([m["iou"] for m in metrics_robust_on_clean])
    avg_iou_robust_noise = np.mean([m["iou"] for m in metrics_robust_on_noise])

    print(f"{'':<6} {'Superpixel':<28}"
          f"{avg_iou_orig_clean:>10.4f}{avg_iou_orig_noise:>10.4f}")
    print(f"{'':<6} {'Superpixel_noise_robust':<28}"
          f"{avg_iou_robust_clean:>10.4f}{avg_iou_robust_noise:>10.4f}")
    diff_orig  = avg_iou_robust_clean - avg_iou_orig_clean
    diff_noise = avg_iou_robust_noise - avg_iou_orig_noise
    print(f"{'':<6} {'Δ (robust-orig)':<28}{diff_orig:>+10.4f}{diff_noise:>+10.4f}")

    # --- Full Precision/Recall/F1/IoU summary 
    print("\n=== Average metrics across all images ===")
    print("-" * 72)
    print(f"{'Method':<25} {'Condition':<10} {'Precision':>10} "
          f"{'Recall':>10} {'F1':>10} {'IoU':>10}")
    print("-" * 72)

    def _summary(method_label, condition_label, metric_list):
        avg_p   = np.mean([m["precision"] for m in metric_list])
        avg_r   = np.mean([m["recall"]    for m in metric_list])
        avg_f1  = np.mean([m["f1"]        for m in metric_list])
        avg_iou = np.mean([m["iou"]       for m in metric_list])
        print(f"{method_label:<25} {condition_label:<10} "
              f"{avg_p:>10.4f} {avg_r:>10.4f} {avg_f1:>10.4f} {avg_iou:>10.4f}")

    _summary("Superpixel",              "Original", metrics_orig_on_clean)
    _summary("Superpixel",              "Noise",    metrics_orig_on_noise)
    _summary("Superpixel_noise_robust", "Original", metrics_robust_on_clean)
    _summary("Superpixel_noise_robust", "Noise",    metrics_robust_on_noise)
    print("=" * 72)


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--split", choices=["train", "val", "test"], default="train")
    p.add_argument("--data_root", type=str, default=None)
    p.add_argument(
        "--indices", type=int, nargs="+",
        default=[0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12],
    )
    args = p.parse_args()

    folder_path = get_split_path(args.split, data_root=args.data_root)
    print(f"Using split folder: {folder_path}")

    evaluate_noise_robustness(folder_path, indices=args.indices)


if __name__ == "__main__":
    main()