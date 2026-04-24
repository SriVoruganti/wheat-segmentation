import os
import glob
from pathlib import Path

import numpy as np
import cv2
from PIL import Image
import matplotlib.pyplot as plt


# IO (EWS dataset)
def read_rgb(path: str) -> np.ndarray:
    """Read image as RGB uint8 (H,W,3)."""
    return np.array(Image.open(path).convert("RGB"))


def read_mask_ews(path: str) -> np.ndarray:
    """
    EWS mask rule (locked):
    - mask is PNG, can be 2-channel/3D
    - use channel 0
    - wheat/plant = 1 where mask_ch0 == 0
    """
    m = np.array(Image.open(path))
    if m.ndim == 3:
        m = m[:, :, 0]
    return (m == 0).astype(np.uint8)


def load_pairs_ews(data_root: str, split: str = "test"):
    """
    Expected layout:
      {data_root}/{split}/images/*
      {data_root}/{split}/masks/*_mask.png
    Image base name must match mask base name + '_mask.png'
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


# Metrics
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


def summarise(values: np.ndarray, name: str):
    values = np.asarray(values, dtype=np.float64)
    if values.size == 0:
        print(f"{name}: no values (empty).")
        return
    print(f"{name} mean:   {values.mean():.4f}")
    print(f"{name} median: {np.median(values):.4f}")
    print(f"{name} std:    {values.std():.4f}")
    print(f"{name} min/max:{values.min():.4f} / {values.max():.4f}")


# RF features + segmentation
def pixel_features(img_rgb: np.ndarray, mode: str = "rgb") -> np.ndarray:
    """
    Build per-pixel feature vectors.

    Supported mode strings (flexible):
      - "rgb"
      - contains "hsv" -> add HSV (3)
      - contains "exg" -> add ExG (1)
    Example: "rgb_hsv_exg" -> 7 features.
    """
    mode = (mode or "rgb").lower()

    x = img_rgb.astype(np.float32) / 255.0
    R, G, B = x[..., 0], x[..., 1], x[..., 2]
    feats = [R, G, B]

    if "exg" in mode:
        feats.append(2 * G - R - B)

    if "hsv" in mode:
        # OpenCV HSV: H in [0,179], S,V in [0,255]
        hsv = cv2.cvtColor(img_rgb, cv2.COLOR_RGB2HSV).astype(np.float32)
        H = hsv[..., 0] / 179.0
        S = hsv[..., 1] / 255.0
        V = hsv[..., 2] / 255.0
        feats += [H, S, V]

    F = np.stack(feats, axis=-1)              # (H,W,C)
    return F.reshape(-1, F.shape[-1]).astype(np.float32)  # (H*W,C)


def segment_rf(img_rgb: np.ndarray, rf_model, feature_mode: str = "rgb") -> np.ndarray:
    """Predict a full (H,W) binary mask using a trained RF model."""
    H, W = img_rgb.shape[:2]
    X = pixel_features(img_rgb, mode=feature_mode)
    if hasattr(rf_model, "n_features_in_") and X.shape[1] != rf_model.n_features_in_:
        raise ValueError(f"Feature mismatch: X has {X.shape[1]} features, model expects {rf_model.n_features_in_}")
    y = rf_model.predict(X)
    return y.reshape(H, W).astype(np.uint8)

# Visualisation helpers
def save_panel(img_rgb, gt01, pred01, out_path: str, title: str = ""):
    out_path = str(out_path)
    Path(os.path.dirname(out_path)).mkdir(parents=True, exist_ok=True)

    plt.figure(figsize=(9, 3))
    for j, (im, t, cm) in enumerate(
        [(img_rgb, "Original", None), (gt01, "Ground Truth", "gray"), (pred01, "RF Prediction", "gray")], 1
    ):
        plt.subplot(1, 3, j)
        plt.imshow(im, cmap=cm)
        plt.title(t)
        plt.axis("off")

    plt.suptitle(title)
    plt.tight_layout()
    plt.savefig(out_path, dpi=200)
    plt.close()


def combine_panels(panel_paths, out_path: str, layout: str = "vertical"):
    """
    Combine saved panel PNGs into one image.
    Uses OpenCV only (no Pillow).
    """
    imgs = []
    for p in panel_paths:
        im = cv2.imread(str(p), cv2.IMREAD_COLOR)
        if im is None:
            raise FileNotFoundError(f"Could not read: {p}")
        imgs.append(im)

    if layout == "horizontal":
        target_h = max(im.shape[0] for im in imgs)
        resized = []
        for im in imgs:
            h, w = im.shape[:2]
            if h != target_h:
                new_w = int(round(w * (target_h / h)))
                im = cv2.resize(im, (new_w, target_h), interpolation=cv2.INTER_AREA)
            resized.append(im)
        combined = np.hstack(resized)
    else:
        target_w = max(im.shape[1] for im in imgs)
        resized = []
        for im in imgs:
            h, w = im.shape[:2]
            if w != target_w:
                new_h = int(round(h * (target_w / w)))
                im = cv2.resize(im, (target_w, new_h), interpolation=cv2.INTER_AREA)
            resized.append(im)
        combined = np.vstack(resized)

    out_path = str(out_path)
    Path(os.path.dirname(out_path)).mkdir(parents=True, exist_ok=True)
    cv2.imwrite(out_path, combined)
