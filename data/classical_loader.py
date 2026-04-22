"""
Helper functions for loading EWS images and ground-truth masks for the
classical (non-deep-learning) advanced segmentation methods.

Expects the ORIGINAL EWS-Dataset folder layout, where every image and its
ground-truth mask sit next to each other in a single split folder:

    EWS-Dataset/
        train/
            FPWW0220032_RGB1_20180411_113950_6.png
            FPWW0220032_RGB1_20180411_113950_6_mask.png
            ...
        val/
            ...
        test/
            ...

The default dataset root is `./data/EWS-Dataset/`, but can be overridden via
the `EWS_DATA_ROOT` environment variable or by passing a `data_root` argument
from a CLI script.
"""

import os
from pathlib import Path
from typing import Tuple

import cv2
import numpy as np


# ---------------------------------------------------------------------------
# Dataset-root resolution
# ---------------------------------------------------------------------------

# Default: <project-root>/data/EWS-Dataset
_DEFAULT_ROOT = Path(__file__).resolve().parents[1] / "data" / "EWS-Dataset"


def get_data_root(data_root: str | os.PathLike | None = None) -> Path:
    """
    Resolve the EWS dataset root folder.

    Priority:
      1. Explicit `data_root` argument
      2. Environment variable `EWS_DATA_ROOT`
      3. Default: <project-root>/data/EWS-Dataset
    """
    if data_root is not None:
        root = Path(data_root)
    elif "EWS_DATA_ROOT" in os.environ:
        root = Path(os.environ["EWS_DATA_ROOT"])
    else:
        root = _DEFAULT_ROOT

    if not root.exists():
        raise FileNotFoundError(
            f"EWS dataset not found at '{root}'.\n"
            "Please download EWS-Dataset.zip from\n"
            "  https://www.research-collection.ethz.ch/entities/researchdata/"
            "165d22fc-6b0f-4fc3-a441-20d8bdc50a70\n"
            "and extract it so that the folder structure is:\n"
            f"  {root}/train/*.png\n"
            f"  {root}/val/*.png\n"
            f"  {root}/test/*.png\n"
            "Alternatively, set the EWS_DATA_ROOT environment variable or\n"
            "pass --data_root to the evaluation scripts."
        )
    return root


def get_split_path(split: str = "train",
                   data_root: str | os.PathLike | None = None) -> Path:
    """Return the folder path for a given split ('train', 'val', 'test')."""
    assert split in ("train", "val", "test"), f"Invalid split: '{split}'"
    return get_data_root(data_root) / split


# ---------------------------------------------------------------------------
# Image / mask loading 
# ---------------------------------------------------------------------------

def list_image_files(folder_path: str | os.PathLike) -> list[str]:
    """Return a sorted list of image filenames (excluding masks) in folder."""
    return sorted([
        f for f in os.listdir(folder_path)
        if not f.endswith("mask.png") and f.endswith((".jpg", ".png"))
    ])


def load_image_and_mask(folder_path: str | os.PathLike,
                        index: int = 0
                        ) -> Tuple[np.ndarray, np.ndarray, str]:
    """
    Load the image at `index` (sorted order) and its corresponding
    ground-truth mask from a split folder.

    Returns
    -------
    img_rgb   : np.ndarray  (H, W, 3) uint8, RGB
    true_mask : np.ndarray  (H, W)    uint8, binary (1 = wheat, 0 = soil)
    img_name  : str         filename of the loaded image
    """
    image_files = list_image_files(folder_path)

    img_name = image_files[index]
    img_path = os.path.join(folder_path, img_name)

    # corresponding mask
    mask_name = img_name.rsplit(".", 1)[0] + "_mask.png"
    mask_path = os.path.join(folder_path, mask_name)

    # RGB image (OpenCV loads BGR by default)
    img_rgb = cv2.cvtColor(cv2.imread(img_path), cv2.COLOR_BGR2RGB)

    # grayscale mask -> binary -> invert so wheat = 1, soil = 0
    true_mask = cv2.imread(mask_path, cv2.IMREAD_GRAYSCALE)
    _, true_mask = cv2.threshold(true_mask, 127, 1, cv2.THRESH_BINARY)
    true_mask = 1 - true_mask

    return img_rgb, true_mask, img_name