"""
EWS Dataset loader for Random Forest with support for:
  - Automatic preprocessing (images/masks folder creation)
  - Standard train/val/test loading
  - Subset sampling for data scarcity experiments
  - Label noise injection for robustness analysis
"""

import os
import random
import shutil
import numpy as np
import cv2


class EWSDataset:
    """
    Eschikon Wheat Segmentation (EWS) Dataset for Random Forest.

    Handles preprocessing automatically — if images/ and masks/
    subdirectories don't exist, they are created and files are
    sorted into them based on the _mask.png naming convention.

    Directory structure after preprocessing:
        root/
            train/images/*.png   train/masks/*.png
            val/images/*.png     val/masks/*.png
            test/images/*.png    test/masks/*.png

    Args:
        root:               Path to EWS dataset root.
        split:              'train', 'val', or 'test'.
        max_pixels_per_image: Max pixels to sample per image.
                              Keeps memory manageable.
        subset_frac:        Float in (0, 1] — use only this fraction
                            of the split. For data scarcity experiments.
        label_noise:        Float in [0, 1) — randomly flip this fraction
                            of mask pixels. For robustness analysis.
        seed:               Random seed for reproducibility.
    """

    def __init__(
        self,
        root:                 str,
        split:                str   = "train",
        max_pixels_per_image: int   = 5000,
        subset_frac:          float = 1.0,
        label_noise:          float = 0.0,
        seed:                 int   = 42,
    ):
        assert split in ("train", "val", "test"), f"Invalid split: '{split}'"
        assert 0 < subset_frac <= 1.0,  "subset_frac must be in (0, 1]"
        assert 0 <= label_noise < 1.0,  "label_noise must be in [0, 1)"

        self.root                 = root
        self.split                = split
        self.max_pixels_per_image = max_pixels_per_image
        self.label_noise          = label_noise
        self.seed                 = seed

        # Run preprocessing if needed
        self._preprocess_folders()

        self.image_dir = os.path.join(root, split, "images")
        self.mask_dir  = os.path.join(root, split, "masks")

        images = sorted(os.listdir(self.image_dir))
        masks  = sorted(os.listdir(self.mask_dir))
        assert len(images) == len(masks), (
            f"Mismatch: {len(images)} images vs {len(masks)} masks"
        )

        # Subset sampling — mirrors original EWSDataset
        if subset_frac < 1.0:
            rng = random.Random(seed)
            n   = max(1, int(len(images) * subset_frac))
            idx = rng.sample(range(len(images)), n)
            images = [images[i] for i in sorted(idx)]
            masks  = [masks[i]  for i in sorted(idx)]

        self.image_files = images
        self.mask_files  = masks

    # ------------------------------------------------------------------
    # Preprocessing
    # ------------------------------------------------------------------

    def _preprocess_folders(self):
        """
        Creates images/ and masks/ subdirectories and moves files
        into them if they don't already exist. Safe to re-run.
        """

        source_dir = os.path.join(self.root, "validation")
        target_dir = os.path.join(self.root, "val")
        if os.path.exists(source_dir) and not os.path.exists(target_dir):
            os.rename(source_dir, target_dir)

        split_dir  = os.path.join(self.root, self.split)
        images_dir = os.path.join(split_dir, "images")
        masks_dir  = os.path.join(split_dir, "masks")

        os.makedirs(images_dir, exist_ok=True)
        os.makedirs(masks_dir,  exist_ok=True)

        for file in sorted(os.listdir(split_dir)):
            file_path = os.path.join(split_dir, file)

            if os.path.isdir(file_path):
                continue
            if not file.lower().endswith(".png"):
                continue

            dest = os.path.join(
                masks_dir  if "_mask.png" in file else images_dir,
                file
            )

            if not os.path.exists(dest):
                shutil.move(file_path, dest)

    # ------------------------------------------------------------------
    # Loading
    # ------------------------------------------------------------------

    def load(self):
        """
        Loads all images and masks into a flat numpy feature matrix.

        Returns:
            X: (n_pixels, 3)  — RGB features per pixel
            y: (n_pixels,)    — binary label per pixel (0=background, 1=wheat)
        """
        np.random.seed(self.seed)

        X_list = []
        y_list = []

        for img_file, mask_file in zip(self.image_files, self.mask_files):
            img_path  = os.path.join(self.image_dir, img_file)
            mask_path = os.path.join(self.mask_dir,  mask_file)

            # Load image (BGR → RGB) and mask (greyscale)
            image = cv2.imread(img_path)
            image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB).astype(np.float32)
            mask  = cv2.imread(mask_path, cv2.IMREAD_GRAYSCALE).astype(np.float32)

            # Binarise mask — same threshold as original EWSDataset
            mask = (mask > 127).astype(np.float32)

            # Inject label noise — mirrors original EWSDataset
            if self.label_noise > 0:
                noise_map = np.random.rand(*mask.shape) < self.label_noise
                mask      = np.where(noise_map, 1.0 - mask, mask)

            # Flatten and subsample
            H, W, C = image.shape
            pixels  = image.reshape(-1, C)
            labels  = mask.reshape(-1)

            idx = np.random.choice(
                len(pixels),
                size=min(self.max_pixels_per_image, len(pixels)),
                replace=False
            )
            X_list.append(pixels[idx])
            y_list.append(labels[idx])

        X = np.vstack(X_list)
        y = np.concatenate(y_list)

        return X, y

    def get_filename(self, idx: int) -> str:
        """Mirrors original EWSDataset.get_filename()."""
        return self.image_files[idx]

    def __len__(self):
        return len(self.image_files)