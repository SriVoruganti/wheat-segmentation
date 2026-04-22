"""
Evaluate the classical advanced-segmentation methods
(Watershed and Superpixel) on the EWS dataset.

Usage
-----
From the project root (directory containing `data/`, `models/`, ...):

    # Single-image qualitative visualisation
    python -m experiments.evaluate_advanced --mode single --index 1

    # Batch evaluation (prints average IoU over a list of indices)
    python -m experiments.evaluate_advanced --mode batch \\
        --indices 0 1 2 3 4 5 6 7 8 9 10 11 12

    # Use a custom dataset location
    python -m experiments.evaluate_advanced --mode batch \\
        --data_root /path/to/EWS-Dataset
"""

import argparse
import sys
from pathlib import Path

import numpy as np
import matplotlib.pyplot as plt

# Make the project root importable even when called as a plain script
sys.path.append(str(Path(__file__).resolve().parents[1]))

from data.classical_loader import get_split_path, load_image_and_mask
from models.advanced_segmentation import (
    segment_wheat_watershed,
    segment_wheat_superpixel,
)
from utils.classical_metrics import calculate_iou


# ---------------------------------------------------------------------------
# Single-image qualitative visualisation
# ---------------------------------------------------------------------------

def evaluate_all_methods(folder_path, index: int = 0, save_path: str | None = None):
    """Plot Original / GroundTruth / Watershed / Superpixel for a single image."""
    img, true_mask, name = load_image_and_mask(folder_path, index)

    # generate mask
    m_water = segment_wheat_watershed(img)
    m_super = segment_wheat_superpixel(img)

    # calculate IoUs
    iou_water = calculate_iou(true_mask, m_water)
    iou_super = calculate_iou(true_mask, m_super)

    # plotting
    fig, ax = plt.subplots(1, 4, figsize=(16, 4))

    ax[0].imshow(img)
    ax[0].set_title("Original")
    ax[0].axis("off")

    ax[1].imshow(true_mask * 255, cmap="gray", vmin=0, vmax=255)
    ax[1].set_title("Ground Truth")
    ax[1].axis("off")

    ax[2].imshow(m_water * 255, cmap="gray", vmin=0, vmax=255)
    ax[2].set_title(f"Watershed (IoU: {iou_water:.2f})")
    ax[2].axis("off")

    ax[3].imshow(m_super * 255, cmap="gray", vmin=0, vmax=255)
    ax[3].set_title(f"Superpixel (IoU: {iou_super:.2f})")
    ax[3].axis("off")

    plt.suptitle(f"Image: {name}")
    plt.tight_layout()

    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches="tight")
        print(f"Figure saved to {save_path}")
    plt.show()


# ---------------------------------------------------------------------------
# Batch evaluation (text-only table)
# ---------------------------------------------------------------------------

def test_multiple_images_text(folder_path, indices):
    """Print a table of IoU scores for Watershed and Superpixel on each index."""
    methods = {
        "Watershed":  segment_wheat_watershed,
        "Superpixel": segment_wheat_superpixel,
    }
    # collect IoU-values per method
    scores = {name: [] for name in methods}

    print(f"{'img(index)':<10} {'Watershed':>12} {'Superpixel':>12}")
    print("-" * 38)

    for idx in indices:
        image_rgb, true_mask, _ = load_image_and_mask(folder_path, idx)
        row = f"{idx:<10}"
        for name, fn in methods.items():
            iou = calculate_iou(true_mask, fn(image_rgb))
            scores[name].append(iou)
            row += f"{iou:>12.4f}"
        print(row)

    print("-" * 38)
    avg_row = f"{'Average:':<10}"
    for name in methods:
        avg_row += f"{np.mean(scores[name]):>12.4f}"
    print(avg_row)

    return scores


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--mode", choices=["single", "batch"], default="batch")
    p.add_argument("--split", choices=["train", "val", "test"], default="train")
    p.add_argument("--data_root", type=str, default=None,
                   help="Path to the EWS-Dataset root folder "
                        "(default: ./data/EWS-Dataset or $EWS_DATA_ROOT).")
    p.add_argument("--index", type=int, default=1,
                   help="Image index for --mode single")
    p.add_argument(
        "--indices", type=int, nargs="+",
        default=[0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12],
        help="Image indices for --mode batch",
    )
    p.add_argument("--save", type=str, default=None,
                   help="Optional path to save the figure (--mode single only)")
    args = p.parse_args()

    folder_path = get_split_path(args.split, data_root=args.data_root)
    print(f"Using split folder: {folder_path}")

    if args.mode == "single":
        evaluate_all_methods(folder_path, index=args.index, save_path=args.save)
    else:
        test_multiple_images_text(folder_path, indices=args.indices)


if __name__ == "__main__":
    main()