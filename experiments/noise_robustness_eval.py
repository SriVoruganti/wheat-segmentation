"""
Compare the noise-robust superpixel variant against the baseline superpixel
method on both clean and noise-distorted EWS images.

Reports per-image and mean IoU scores, as well as the absolute improvement
(Δ) of the noise-robust variant on both clean and noisy inputs.

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
from utils.classical_metrics import calculate_iou


def evaluate_noise_robustness(folder_path, indices):
    """Per-image table + averaged deltas for the noise-robust variant."""

    print(f"{'img':<6} {'method':<28} {'Original':>10} {'Noise':>10}")
    print("=" * 60)

    scores_orig   = {"Original": [], "Noise": []}
    scores_robust = {"Original": [], "Noise": []}

    for idx in indices:
        img_orig, true_mask, _ = load_image_and_mask(folder_path, idx)
        img_noise = add_noise(img_orig)

        iou_o_orig  = calculate_iou(true_mask, segment_wheat_superpixel(img_orig))
        iou_o_noise = calculate_iou(true_mask, segment_wheat_superpixel(img_noise))
        iou_r_orig  = calculate_iou(true_mask, segment_wheat_superpixel_noise_robust(img_orig))
        iou_r_noise = calculate_iou(true_mask, segment_wheat_superpixel_noise_robust(img_noise))

        scores_orig["Original"].append(iou_o_orig)
        scores_orig["Noise"].append(iou_o_noise)
        scores_robust["Original"].append(iou_r_orig)
        scores_robust["Noise"].append(iou_r_noise)

        print(f"{idx:<6} {'Superpixel':<28}{iou_o_orig:>10.4f}{iou_o_noise:>10.4f}")
        print(f"{'':<6} {'Superpixel_noise_robust':<28}{iou_r_orig:>10.4f}{iou_r_noise:>10.4f}")
        print("-" * 60)

    print("\n" + "=" * 60)
    print(f"{'AVG':<6} {'method':<28} {'Original':>10} {'Noise':>10}")
    print("-" * 60)
    print(f"{'':<6} {'Superpixel':<28}"
          f"{np.mean(scores_orig['Original']):>10.4f}"
          f"{np.mean(scores_orig['Noise']):>10.4f}")
    print(f"{'':<6} {'Superpixel_noise_robust':<28}"
          f"{np.mean(scores_robust['Original']):>10.4f}"
          f"{np.mean(scores_robust['Noise']):>10.4f}")
    diff_orig  = np.mean(scores_robust["Original"]) - np.mean(scores_orig["Original"])
    diff_noise = np.mean(scores_robust["Noise"])    - np.mean(scores_orig["Noise"])
    print(f"{'':<6} {'Δ (robust-orig)':<28}"
          f"{diff_orig:>+10.4f}{diff_noise:>+10.4f}")

    return scores_orig, scores_robust


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