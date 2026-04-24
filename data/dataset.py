"""
EWS Dataset loader with support for:
  - Deep Learning (PyTorch) — U-Net training and evaluation
  - Classical ML (Random Forest) — pixel-level feature extraction
  - Standard train/val/test loading
  - Automatic folder preprocessing
  - Subset sampling for data scarcity experiments
  - Label noise injection for robustness analysis
  - Robust image/mask filename pairing
  - Optional RF feature modes: RGB, RGB+ExG, RGB+HSV, RGB+HSV+ExG
"""

import inspect
import os
import random
import shutil
from typing import Literal, Optional, Tuple, List

import cv2
import numpy as np
import torch
from PIL import Image
from torch.utils.data import Dataset

import albumentations as A
from albumentations.pytorch import ToTensorV2


MaskPositive = Literal["white", "black"]
FeatureMode = Literal["rgb", "rgb_exg", "rgb_hsv", "rgb_hsv_exg"]


# ===========================================================================
# Shared Utility — Folder Preprocessing
# ===========================================================================

def preprocess_folders(root: str, split: str) -> None:
    """
    Creates images/ and masks/ subdirectories and moves files into them
    if they don't already exist. Safe to re-run.

    Also handles the 'validation' -> 'val' rename automatically.
    """
    source_dir = os.path.join(root, "validation")
    target_dir = os.path.join(root, "val")
    if os.path.exists(source_dir) and not os.path.exists(target_dir):
        os.rename(source_dir, target_dir)

    split_dir = os.path.join(root, split)
    if not os.path.isdir(split_dir):
        raise FileNotFoundError(f"Split directory does not exist: {split_dir}")

    images_dir = os.path.join(split_dir, "images")
    masks_dir = os.path.join(split_dir, "masks")

    os.makedirs(images_dir, exist_ok=True)
    os.makedirs(masks_dir, exist_ok=True)

    for file in sorted(os.listdir(split_dir)):
        file_path = os.path.join(split_dir, file)

        if os.path.isdir(file_path):
            continue
        if not file.lower().endswith(".png"):
            continue

        dest = os.path.join(masks_dir if "_mask.png" in file else images_dir, file)
        if not os.path.exists(dest):
            shutil.move(file_path, dest)


def _expected_mask_name(image_name: str) -> str:
    base, ext = os.path.splitext(image_name)
    return f"{base}_mask{ext}"


def _image_name_from_mask(mask_name: str) -> str:
    base, ext = os.path.splitext(mask_name)
    if base.endswith("_mask"):
        base = base[:-5]
    return f"{base}{ext}"


def _load_file_lists(
    root: str,
    split: str,
    subset_frac: float = 1.0,
    seed: int = 42,
    warn_unpaired: bool = True,
) -> Tuple[str, str, List[str], List[str]]:
    """
    Shared helper to preprocess folders, pair images with masks, and optionally
    take a deterministic subset.
    """
    preprocess_folders(root, split)

    image_dir = os.path.join(root, split, "images")
    mask_dir = os.path.join(root, split, "masks")

    if not os.path.isdir(image_dir):
        raise FileNotFoundError(f"Image directory does not exist: {image_dir}")
    if not os.path.isdir(mask_dir):
        raise FileNotFoundError(f"Mask directory does not exist: {mask_dir}")

    images = sorted(f for f in os.listdir(image_dir) if f.lower().endswith(".png"))
    masks = sorted(f for f in os.listdir(mask_dir) if f.lower().endswith(".png"))

    mask_set = set(masks)
    paired_images, paired_masks = [], []

    for img in images:
        expected_mask = _expected_mask_name(img)
        if expected_mask in mask_set:
            paired_images.append(img)
            paired_masks.append(expected_mask)
        elif warn_unpaired:
            print(f"WARNING: missing mask for image: {img}")

    if warn_unpaired:
        image_set = set(images)
        extra_masks = [m for m in masks if _image_name_from_mask(m) not in image_set]
        if extra_masks:
            print("WARNING: masks with no matching image (first 10):", extra_masks[:10])

    if not paired_images:
        raise RuntimeError(f"No image/mask pairs found in split '{split}'.")

    if subset_frac < 1.0:
        rng = random.Random(seed)
        n = max(1, int(len(paired_images) * subset_frac))
        idx = sorted(rng.sample(range(len(paired_images)), n))
        paired_images = [paired_images[i] for i in idx]
        paired_masks = [paired_masks[i] for i in idx]

    return image_dir, mask_dir, paired_images, paired_masks


def _binary_mask(mask: np.ndarray, positive: MaskPositive = "white") -> np.ndarray:
    """Convert a grayscale or channel mask to binary float32 labels."""
    if mask.ndim == 3:
        mask = mask[:, :, 0]

    if positive == "white":
        return (mask > 127).astype(np.float32)
    if positive == "black":
        return (mask == 0).astype(np.float32)
    raise ValueError("positive must be either 'white' or 'black'")


# ===========================================================================
# Deep Learning Dataset — PyTorch / U-Net
# ===========================================================================

class EWSDatasetTorch(Dataset):
    """
    PyTorch Dataset for U-Net training and evaluation.

    Returns whole images as normalised tensors with albumentations
    augmentations applied. Use this with train.py and evaluate.py.

    Directory structure expected:
        root/
            train/images/*.png   train/masks/*.png
            val/images/*.png     val/masks/*.png
            test/images/*.png    test/masks/*.png

    Args:
        root:          Path to EWS dataset root.
        split:         'train', 'val', or 'test'.
        transform:     Albumentations pipeline.
        subset_frac:   Float in (0, 1] — use only this fraction of the split.
        label_noise:   Float in [0, 1) — randomly flip this fraction of mask pixels.
        seed:          Random seed for reproducibility.
        mask_positive: 'white' for foreground mask pixels >127, or 'black' for
                       foreground mask pixels equal to 0.
    """

    def __init__(
        self,
        root: str,
        split: str = "train",
        transform: Optional[A.Compose] = None,
        subset_frac: float = 1.0,
        label_noise: float = 0.0,
        seed: int = 42,
        mask_positive: MaskPositive = "white",
    ):
        assert split in ("train", "val", "test"), f"Invalid split: '{split}'"
        assert 0 < subset_frac <= 1.0, "subset_frac must be in (0, 1]"
        assert 0 <= label_noise < 1.0, "label_noise must be in [0, 1)"
        assert mask_positive in ("white", "black"), "mask_positive must be 'white' or 'black'"

        self.transform = transform
        self.label_noise = label_noise
        self.mask_positive = mask_positive
        self.rng = np.random.default_rng(seed)

        self.image_dir, self.mask_dir, self.images, self.masks = _load_file_lists(
            root, split, subset_frac, seed
        )

    def __len__(self) -> int:
        return len(self.images)

    def __getitem__(self, idx: int):
        img_path = os.path.join(self.image_dir, self.images[idx])
        mask_path = os.path.join(self.mask_dir, self.masks[idx])

        image = np.array(Image.open(img_path).convert("RGB"), dtype=np.float32)
        mask_raw = np.array(Image.open(mask_path).convert("L"), dtype=np.float32)
        mask = _binary_mask(mask_raw, self.mask_positive)

        if self.label_noise > 0:
            noise_map = self.rng.random(mask.shape) < self.label_noise
            mask = np.where(noise_map, 1.0 - mask, mask).astype(np.float32)

        if self.transform:
            aug = self.transform(image=image, mask=mask)
            image = aug["image"]
            mask = aug["mask"]

        if isinstance(mask, torch.Tensor):
            mask = mask.float().unsqueeze(0)
        else:
            mask = torch.tensor(mask, dtype=torch.float32).unsqueeze(0)

        return image, mask

    def get_filename(self, idx: int) -> str:
        return self.images[idx]


# ===========================================================================
# Classical ML Dataset — Random Forest / Pixel Features
# ===========================================================================

class EWSDatasetRF:
    """
    Classical ML Dataset for Random Forest training and evaluation.

    Flattens images into a pixel-level feature matrix (n_pixels, C) suitable
    for sklearn classifiers. Handles raw EWS folder structure automatically.

    Args:
        root:                 Path to EWS dataset root.
        split:                'train', 'val', or 'test'.
        max_pixels_per_image: Max pixels to sample per image.
        subset_frac:          Float in (0, 1] — use only this fraction of split.
        label_noise:          Float in [0, 1) — randomly flip this fraction of pixels.
        seed:                 Random seed for reproducibility.
        feature_mode:         'rgb', 'rgb_exg', 'rgb_hsv', or 'rgb_hsv_exg'.
        mask_positive:        'black' matches masks where plant/foreground is 0;
                              use 'white' for masks where foreground is >127.
    """

    def __init__(
        self,
        root: str,
        split: str = "train",
        max_pixels_per_image: int = 5000,
        subset_frac: float = 1.0,
        label_noise: float = 0.0,
        seed: int = 42,
        feature_mode: FeatureMode = "rgb",
        mask_positive: MaskPositive = "black",
    ):
        assert split in ("train", "val", "test"), f"Invalid split: '{split}'"
        assert 0 < subset_frac <= 1.0, "subset_frac must be in (0, 1]"
        assert 0 <= label_noise < 1.0, "label_noise must be in [0, 1)"
        assert feature_mode in ("rgb", "rgb_exg", "rgb_hsv", "rgb_hsv_exg"), f"Bad feature_mode: {feature_mode}"
        assert mask_positive in ("white", "black"), "mask_positive must be 'white' or 'black'"

        self.max_pixels_per_image = max_pixels_per_image
        self.subset_frac = subset_frac
        self.label_noise = label_noise
        self.seed = seed
        self.feature_mode = feature_mode
        self.mask_positive = mask_positive

        self.image_dir, self.mask_dir, self.image_files, self.mask_files = _load_file_lists(
            root, split, subset_frac, seed
        )

    def _extract_features(self, image_u8: np.ndarray) -> np.ndarray:
        """Return feature image with shape (H, W, C)."""
        rgb = image_u8.astype(np.float32) / 255.0
        feats = [rgb]

        if self.feature_mode in ("rgb_exg", "rgb_hsv_exg"):
            r, g, b = rgb[:, :, 0], rgb[:, :, 1], rgb[:, :, 2]
            exg = (2.0 * g - r - b)[:, :, None].astype(np.float32)
            feats.append(exg)

        if self.feature_mode in ("rgb_hsv", "rgb_hsv_exg"):
            hsv = cv2.cvtColor(image_u8, cv2.COLOR_RGB2HSV).astype(np.float32)
            hsv[:, :, 0] /= 179.0
            hsv[:, :, 1] /= 255.0
            hsv[:, :, 2] /= 255.0
            feats.append(hsv)

        return np.concatenate(feats, axis=2).astype(np.float32)

    def load(self):
        """
        Loads all images and masks into a flat numpy feature matrix.

        Returns:
            X: (n_pixels, C) — features per pixel
            y: (n_pixels,)   — binary label per pixel (0=background, 1=foreground)
        """
        rng = np.random.default_rng(self.seed)

        X_list, y_list = [], []

        for img_file, mask_file in zip(self.image_files, self.mask_files):
            img_path = os.path.join(self.image_dir, img_file)
            mask_path = os.path.join(self.mask_dir, mask_file)

            image_bgr = cv2.imread(img_path, cv2.IMREAD_COLOR)
            if image_bgr is None:
                raise FileNotFoundError(f"Could not read image: {img_path}")
            image_u8 = cv2.cvtColor(image_bgr, cv2.COLOR_BGR2RGB)

            feat_img = self._extract_features(image_u8)
            h, w, c = feat_img.shape
            pixels = feat_img.reshape(-1, c).astype(np.float32)

            mask_raw = cv2.imread(mask_path, cv2.IMREAD_UNCHANGED)
            if mask_raw is None:
                raise FileNotFoundError(f"Could not read mask: {mask_path}")
            mask = _binary_mask(mask_raw, self.mask_positive)

            if self.label_noise > 0:
                noise_map = rng.random(mask.shape) < self.label_noise
                mask = np.where(noise_map, 1.0 - mask, mask).astype(np.float32)

            labels = mask.reshape(-1).astype(np.float32)

            n = len(pixels)
            k = min(self.max_pixels_per_image, n)
            idx = rng.choice(n, size=k, replace=False)

            X_list.append(pixels[idx])
            y_list.append(labels[idx])

        X = np.vstack(X_list).astype(np.float32)
        y = np.concatenate(y_list).astype(np.float32)

        return X, y

    def get_filename(self, idx: int) -> str:
        return self.image_files[idx]

    def __len__(self) -> int:
        return len(self.image_files)




# ===========================================================================
# Compatibility Dataset — supports all three submitted calling styles
# ===========================================================================

class EWSDataset(EWSDatasetTorch):
    """
    Backward-compatible dataset name used by all three versions.

    Keeps PyTorch/U-Net behaviour and also exposes `.load()` for the Random
    Forest pixel-feature workflow, so both old calling styles work:

        EWSDataset(root, split, transform=get_train_transforms())   # PyTorch
        EWSDataset(root, split).load()                              # RF
    """

    def __init__(
        self,
        root: str,
        split: str = "train",
        transform: Optional[A.Compose] = None,
        subset_frac: float = 1.0,
        label_noise: float = 0.0,
        seed: int = 42,
        max_pixels_per_image: int = 5000,
        feature_mode: FeatureMode = "rgb",
        mask_positive: MaskPositive = "white",
        rf_mask_positive: Optional[MaskPositive] = None,
    ):
        super().__init__(
            root=root,
            split=split,
            transform=transform,
            subset_frac=subset_frac,
            label_noise=label_noise,
            seed=seed,
            mask_positive=mask_positive,
        )

        assert feature_mode in ("rgb", "rgb_exg", "rgb_hsv", "rgb_hsv_exg"), f"Bad feature_mode: {feature_mode}"
        self.root = root
        self.split = split
        self.max_pixels_per_image = max_pixels_per_image
        self.subset_frac = subset_frac
        self.seed = seed
        self.feature_mode = feature_mode
        self.rf_mask_positive = rf_mask_positive or mask_positive

        # RF-only version used these exact names.
        self.image_files = self.images
        self.mask_files = self.masks

    def _extract_features(self, image_u8: np.ndarray) -> np.ndarray:
        rgb = image_u8.astype(np.float32) / 255.0
        feats = [rgb]

        if self.feature_mode in ("rgb_exg", "rgb_hsv_exg"):
            r, g, b = rgb[:, :, 0], rgb[:, :, 1], rgb[:, :, 2]
            exg = (2.0 * g - r - b)[:, :, None].astype(np.float32)
            feats.append(exg)

        if self.feature_mode in ("rgb_hsv", "rgb_hsv_exg"):
            hsv = cv2.cvtColor(image_u8, cv2.COLOR_RGB2HSV).astype(np.float32)
            hsv[:, :, 0] /= 179.0
            hsv[:, :, 1] /= 255.0
            hsv[:, :, 2] /= 255.0
            feats.append(hsv)

        return np.concatenate(feats, axis=2).astype(np.float32)

    def load(self):
        rng = np.random.default_rng(self.seed)
        X_list, y_list = [], []

        for img_file, mask_file in zip(self.image_files, self.mask_files):
            img_path = os.path.join(self.image_dir, img_file)
            mask_path = os.path.join(self.mask_dir, mask_file)

            image_bgr = cv2.imread(img_path, cv2.IMREAD_COLOR)
            if image_bgr is None:
                raise FileNotFoundError(f"Could not read image: {img_path}")
            image_u8 = cv2.cvtColor(image_bgr, cv2.COLOR_BGR2RGB)

            feat_img = self._extract_features(image_u8)
            h, w, c = feat_img.shape
            pixels = feat_img.reshape(-1, c).astype(np.float32)

            mask_raw = cv2.imread(mask_path, cv2.IMREAD_UNCHANGED)
            if mask_raw is None:
                raise FileNotFoundError(f"Could not read mask: {mask_path}")
            mask = _binary_mask(mask_raw, self.rf_mask_positive)

            if self.label_noise > 0:
                noise_map = rng.random(mask.shape) < self.label_noise
                mask = np.where(noise_map, 1.0 - mask, mask).astype(np.float32)

            labels = mask.reshape(-1).astype(np.float32)
            n = len(pixels)
            k = min(self.max_pixels_per_image, n)
            idx = rng.choice(n, size=k, replace=False)

            X_list.append(pixels[idx])
            y_list.append(labels[idx])

        return np.vstack(X_list).astype(np.float32), np.concatenate(y_list).astype(np.float32)


# Explicit aliases for every style/name.
EWSDatasetDL = EWSDatasetTorch
EWSUNetDataset = EWSDatasetTorch
EWSDatasetPyTorch = EWSDatasetTorch
EWSRandomForestDataset = EWSDatasetRF
EWSDatasetClassical = EWSDatasetRF

# ===========================================================================
# Augmentation Pipelines (for EWSDataset / U-Net)
# ===========================================================================

def _albumentations_kwargs(transform_cls, **candidates):
    """Keep only kwargs supported by the installed albumentations version."""
    params = inspect.signature(transform_cls).parameters
    return {k: v for k, v in candidates.items() if k in params}


def get_train_transforms(image_size: int = 350) -> A.Compose:
    """Comprehensive augmentation pipeline for U-Net training."""
    gauss_noise_kwargs = _albumentations_kwargs(
        A.GaussNoise,
        std_range=(0.01, 0.05),   # albumentations newer API
        var_limit=(10, 50),       # albumentations older API
        p=0.3,
    )
    coarse_dropout_kwargs = _albumentations_kwargs(
        A.CoarseDropout,
        num_holes_range=(1, 8),
        hole_height_range=(10, 30),
        hole_width_range=(10, 30),
        max_holes=8,
        max_height=30,
        max_width=30,
        p=0.3,
    )

    return A.Compose([
        A.Resize(image_size, image_size),
        # Geometric
        A.HorizontalFlip(p=0.5),
        A.VerticalFlip(p=0.5),
        A.RandomRotate90(p=0.5),
        A.ShiftScaleRotate(shift_limit=0.1, scale_limit=0.2, rotate_limit=30, p=0.5),
        A.ElasticTransform(alpha=1, sigma=50, p=0.3),
        A.GridDistortion(p=0.3),
        # Colour / photometric
        A.ColorJitter(brightness=0.3, contrast=0.3, saturation=0.3, hue=0.1, p=0.6),
        A.RandomGamma(gamma_limit=(80, 120), p=0.3),
        # Noise & blur
        A.GaussianBlur(blur_limit=(3, 7), p=0.3),
        A.GaussNoise(**gauss_noise_kwargs),
        A.ISONoise(p=0.2),
        # Occlusion
        A.CoarseDropout(**coarse_dropout_kwargs),
        # Normalise to ImageNet stats
        A.Normalize(mean=(0.485, 0.456, 0.406), std=(0.229, 0.224, 0.225)),
        ToTensorV2(),
    ])


def get_val_transforms(image_size: int = 350) -> A.Compose:
    """No augmentation for validation and testing."""
    return A.Compose([
        A.Resize(image_size, image_size),
        A.Normalize(mean=(0.485, 0.456, 0.406), std=(0.229, 0.224, 0.225)),
        ToTensorV2(),
    ])
