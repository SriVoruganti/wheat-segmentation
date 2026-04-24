import os
import glob
import numpy as np
from PIL import Image


def read_rgb(path: str) -> np.ndarray:
    return np.array(Image.open(path).convert("RGB"))


def read_mask_ews(path: str) -> np.ndarray:
    m = np.array(Image.open(path))
    if m.ndim == 3:
        m = m[:, :, 0]
    return (m == 0).astype(np.uint8)  # wheat=1, soil=0


def iou(pred: np.ndarray, gt: np.ndarray, eps: float = 1e-6) -> float:
    pred = (pred > 0).astype(np.uint8)
    gt = (gt > 0).astype(np.uint8)
    inter = np.logical_and(pred, gt).sum()
    union = np.logical_or(pred, gt).sum()
    return float((inter + eps) / (union + eps))


def f1_score(pred: np.ndarray, gt: np.ndarray, eps: float = 1e-6) -> float:
    pred = (pred > 0).astype(np.uint8)
    gt = (gt > 0).astype(np.uint8)
    tp = np.logical_and(pred == 1, gt == 1).sum()
    fp = np.logical_and(pred == 1, gt == 0).sum()
    fn = np.logical_and(pred == 0, gt == 1).sum()
    return float((2 * tp + eps) / (2 * tp + fp + fn + eps))


def load_pairs_ews(data_root: str, split: str = "test"):
    """Assumes:
    {data_root}/{split}/images
    {data_root}/{split}/masks

    Mask naming: <image_base>_mask.png
    """
    img_dir = os.path.join(data_root, split, "images")
    mask_dir = os.path.join(data_root, split, "masks")

    img_paths = sorted(
        glob.glob(os.path.join(img_dir, "*.png"))
        + glob.glob(os.path.join(img_dir, "*.jpg"))
        + glob.glob(os.path.join(img_dir, "*.jpeg"))
    )

    pairs, missing = [], 0
    for ip in img_paths:
        base = os.path.splitext(os.path.basename(ip))[0]
        mp = os.path.join(mask_dir, base + "_mask.png")
        if os.path.exists(mp):
            pairs.append((ip, mp))
        else:
            missing += 1

    return pairs, missing


def summarise(values: np.ndarray, name: str):
    values = np.asarray(values, dtype=np.float64)
    print(f"{name} mean:   {values.mean():.4f}")
    print(f"{name} median: {np.median(values):.4f}")
    print(f"{name} std:    {values.std():.4f}")
    print(f"{name} min/max:{values.min():.4f} / {values.max():.4f}")