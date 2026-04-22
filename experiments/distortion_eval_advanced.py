"""
Robustness evaluation of the classical advanced-segmentation methods
under simulated image distortions:
  - Gaussian blur
  - Gaussian noise
  - Low brightness / contrast ("dark")
  - Partial occlusion

The baseline `segment_wheat_superpixel` method is applied to the original
image and to each of the four distorted variants, and Precision, Recall,
F1 and IoU against the ground-truth mask are reported.

Usage
-----
From the project root:

    # Single-image qualitative visualisation
    python -m experiments.distortion_eval_advanced --mode single --index 10

    # Batch evaluation (Precision / Recall / F1 / IoU per distortion)
    python -m experiments.distortion_eval_advanced --mode batch \\
        --indices 0 1 2 3 4 5 6 7 8 9 10 11 12
"""

import argparse
import sys
from pathlib import Path

import numpy as np
import matplotlib.pyplot as plt

sys.path.append(str(Path(__file__).resolve().parents[1]))

from data.classical_loader import get_split_path, load_image_and_mask
from models.advanced_segmentation import segment_wheat_superpixel
from utils.classical_distortions import (
    add_blur,
    add_noise,
    adjust_brightness_contrast,
    add_partial_occlusion,
)
from utils.classical_metrics import calculate_iou, calculate_all_metrics


# ---------------------------------------------------------------------------
# Single-image qualitative plot
# ---------------------------------------------------------------------------

def evaluate_robustness(folder_path, index: int = 10, save_path: str | None = None):
    """Plot a 2x5 grid: distorted images (row 1) and predicted masks (row 2)."""
    img_orig, true_mask, name = load_image_and_mask(folder_path, index)

    # distorting
    img_blur  = add_blur(img_orig)
    img_noise = add_noise(img_orig)
    img_dark  = adjust_brightness_contrast(img_orig)
    img_occ   = add_partial_occlusion(img_orig)

    # apply superpixel on all
    mask_orig  = segment_wheat_superpixel(img_orig)
    mask_blur  = segment_wheat_superpixel(img_blur)
    mask_noise = segment_wheat_superpixel(img_noise)
    mask_dark  = segment_wheat_superpixel(img_dark)
    mask_occ   = segment_wheat_superpixel(img_occ)

    # IoU
    iou_orig  = calculate_iou(true_mask, mask_orig)
    iou_blur  = calculate_iou(true_mask, mask_blur)
    iou_noise = calculate_iou(true_mask, mask_noise)
    iou_dark  = calculate_iou(true_mask, mask_dark)
    iou_occ   = calculate_iou(true_mask, mask_occ)

    # plotting
    fig, axes = plt.subplots(2, 5, figsize=(20, 8))

    images = [img_orig, img_blur, img_noise, img_dark, img_occ]
    masks  = [mask_orig, mask_blur, mask_noise, mask_dark, mask_occ]
    ious   = [iou_orig, iou_blur, iou_noise, iou_dark, iou_occ]
    titles = ["Original", "Blur", "Noise", "Low Brightness", "Occlusion"]

    for i in range(5):
        axes[0, i].imshow(images[i])
        axes[0, i].set_title(titles[i])
        axes[0, i].axis("off")
        axes[1, i].imshow(masks[i] * 255, cmap="gray", vmin=0, vmax=255)
        axes[1, i].set_title(f"IoU: {ious[i]:.4f}")
        axes[1, i].axis("off")

    plt.suptitle(f"Robustness Evaluation for Image: {name}")
    plt.tight_layout()

    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches="tight")
        print(f"Figure saved to {save_path}")
    plt.show()


# ---------------------------------------------------------------------------
# Batch evaluation (Precision / Recall / F1 / IoU per distortion)
# ---------------------------------------------------------------------------

def evaluate_distortions_batch(folder_path, indices=None):
    """Print per-image IoUs and a full Precision/Recall/F1/IoU summary."""
    if indices is None:
        indices = [0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12]

    distortion_names = ["orig", "blur", "noise", "dark", "occ"]

    # per-distortion: list of metric dicts, one per image
    all_metrics = {d: [] for d in distortion_names}

    # --- Per-image IoU table ---
    print("\n--- Per-image IoU values (Superpixel under different distortions) ---")
    print(f"{'img(index)':<12} {'Original':<10} {'Blur':<10} "
          f"{'Noise':<10} {'Dark':<10} {'Occlusion':<10}")
    print("-" * 65)

    for i in indices:
        img_orig, true_mask, _ = load_image_and_mask(folder_path, i)

        variants = {
            "orig":  img_orig,
            "blur":  add_blur(img_orig),
            "noise": add_noise(img_orig),
            "dark":  adjust_brightness_contrast(img_orig),
            "occ":   add_partial_occlusion(img_orig),
        }

        row_ious = {}
        for key, variant in variants.items():
            pred = segment_wheat_superpixel(variant)
            m = calculate_all_metrics(true_mask, pred)
            all_metrics[key].append(m)
            row_ious[key] = m["iou"]

        print(f"{i:<12} {row_ious['orig']:<10.4f} {row_ious['blur']:<10.4f} "
              f"{row_ious['noise']:<10.4f} {row_ious['dark']:<10.4f} "
              f"{row_ious['occ']:<10.4f}")

    print("-" * 65)
    avg_line = f"{'Average:':<12} "
    for key in distortion_names:
        avg_iou = np.mean([m["iou"] for m in all_metrics[key]])
        avg_line += f"{avg_iou:<10.4f} "
    print(avg_line)

    # --- Full metrics summary ---
    print("\n=== Average metrics across all images (Superpixel method) ===")
    print("-" * 62)
    print(f"{'Distortion':<14} {'Precision':>10} {'Recall':>10} {'F1':>10} {'IoU':>10}")
    print("-" * 62)
    pretty = {"orig": "Original", "blur": "Blur", "noise": "Noise",
              "dark": "Dark", "occ": "Occlusion"}
    for key in distortion_names:
        avg_p   = np.mean([m["precision"] for m in all_metrics[key]])
        avg_r   = np.mean([m["recall"]    for m in all_metrics[key]])
        avg_f1  = np.mean([m["f1"]        for m in all_metrics[key]])
        avg_iou = np.mean([m["iou"]       for m in all_metrics[key]])
        print(f"{pretty[key]:<14} {avg_p:>10.4f} {avg_r:>10.4f} "
              f"{avg_f1:>10.4f} {avg_iou:>10.4f}")
    print("=" * 62)

    return all_metrics


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--mode", choices=["single", "batch"], default="batch")
    p.add_argument("--split", choices=["train", "val", "test"], default="train")
    p.add_argument("--data_root", type=str, default=None)
    p.add_argument("--index", type=int, default=10,
                   help="Image index for --mode single")
    p.add_argument(
        "--indices", type=int, nargs="+",
        default=[0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12],
    )
    p.add_argument("--save", type=str, default=None,
                   help="Optional path to save the figure (--mode single only)")
    args = p.parse_args()

    folder_path = get_split_path(args.split, data_root=args.data_root)
    print(f"Using split folder: {folder_path}")

    if args.mode == "single":
        evaluate_robustness(folder_path, index=args.index, save_path=args.save)
    else:
        evaluate_distortions_batch(folder_path, indices=args.indices)


if __name__ == "__main__":
    main()