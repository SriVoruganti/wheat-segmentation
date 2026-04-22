"""
Robustness evaluation of the classical advanced-segmentation methods
under simulated image distortions:
  - Gaussian blur
  - Gaussian noise
  - Low brightness / contrast ("dark")
  - Partial occlusion

The baseline `segment_wheat_superpixel` method is applied to the original
image and to each of the four distorted variants, and the IoU against the
ground-truth mask is reported.

Usage
-----
From the project root:

    # Single-image qualitative visualisation
    python -m experiments.distortion_eval_advanced --mode single --index 10

    # Batch evaluation (mean IoU per distortion)
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
from utils.classical_metrics import calculate_iou


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
# Batch evaluation (text table with mean IoUs)
# ---------------------------------------------------------------------------

def evaluate_distortions_batch(folder_path, indices=None):
    """Print a per-image table and the mean IoU for each distortion."""
    if indices is None:
        indices = [0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12]

    print(f"{'img(index)':<12} {'Original':<10} {'Blur':<10} "
          f"{'Noise':<10} {'Dark':<10} {'Occlusion':<10}")
    print("-" * 65)

    ious = {"orig": [], "blur": [], "noise": [], "dark": [], "occ": []}

    for i in indices:
        img_orig, true_mask, _ = load_image_and_mask(folder_path, i)

        # distortions
        img_blur  = add_blur(img_orig)
        img_noise = add_noise(img_orig)
        img_dark  = adjust_brightness_contrast(img_orig)
        img_occ   = add_partial_occlusion(img_orig)

        # superpixel
        mask_orig  = segment_wheat_superpixel(img_orig)
        mask_blur  = segment_wheat_superpixel(img_blur)
        mask_noise = segment_wheat_superpixel(img_noise)
        mask_dark  = segment_wheat_superpixel(img_dark)
        mask_occ   = segment_wheat_superpixel(img_occ)

        # IoU
        i_orig  = calculate_iou(true_mask, mask_orig)
        i_blur  = calculate_iou(true_mask, mask_blur)
        i_noise = calculate_iou(true_mask, mask_noise)
        i_dark  = calculate_iou(true_mask, mask_dark)
        i_occ   = calculate_iou(true_mask, mask_occ)

        ious["orig"].append(i_orig)
        ious["blur"].append(i_blur)
        ious["noise"].append(i_noise)
        ious["dark"].append(i_dark)
        ious["occ"].append(i_occ)

        print(f"{i:<12} {i_orig:<10.4f} {i_blur:<10.4f} "
              f"{i_noise:<10.4f} {i_dark:<10.4f} {i_occ:<10.4f}")

    print("-" * 65)

    avg_orig  = np.mean(ious["orig"])
    avg_blur  = np.mean(ious["blur"])
    avg_noise = np.mean(ious["noise"])
    avg_dark  = np.mean(ious["dark"])
    avg_occ   = np.mean(ious["occ"])

    print(f"{'Average:':<12} {avg_orig:<10.4f} {avg_blur:<10.4f} "
          f"{avg_noise:<10.4f} {avg_dark:<10.4f} {avg_occ:<10.4f}")

    return ious


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