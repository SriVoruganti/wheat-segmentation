"""
Advanced (classical) segmentation methods for wheat vs. soil segmentation
on the EWS dataset.

Three methods are implemented:
  1. segment_wheat_watershed
        HSV thresholding + distance-transform markers + Watershed.
  2. segment_wheat_superpixel
        SLIC superpixels + Excess-Green (ExG) index + Otsu thresholding.
  3. segment_wheat_superpixel_noise_robust
        Same as (2), preceded by edge-preserving denoising
        (bilateral filter + median filter) for robustness against noise.

All functions take an RGB uint8 image of shape (H, W, 3) and return a
binary uint8 mask of shape (H, W) with 1 = wheat, 0 = soil.
"""

import cv2
import numpy as np
from skimage.segmentation import slic
from skimage.filters import threshold_otsu


# ---------------------------------------------------------------------------
# 1. Watershed
# ---------------------------------------------------------------------------

def segment_wheat_watershed(image_rgb: np.ndarray) -> np.ndarray:
    """
    Watershed-based segmentation.

    Uses HSV colour thresholding to obtain a rough foreground mask, then
    uses morphological opening + distance transform to derive `sure_fg`
    and `sure_bg` markers, and finally runs OpenCV's Watershed algorithm.
    """
    image_bgr = cv2.cvtColor(image_rgb, cv2.COLOR_RGB2BGR)

    # rough mask for start
    hsv = cv2.cvtColor(image_rgb, cv2.COLOR_RGB2HSV)
    rough_mask = cv2.inRange(hsv, np.array([25, 35, 35]), np.array([95, 255, 255]))

    kernel = np.ones((3, 3), np.uint8)
    opening = cv2.morphologyEx(rough_mask, cv2.MORPH_OPEN, kernel, iterations=2)

    # sure background (via dilation)
    sure_bg = cv2.dilate(opening, kernel, iterations=3)

    # sure foreground (via distance transform)
    dist_transform = cv2.distanceTransform(opening, cv2.DIST_L2, 5)
    _, sure_fg = cv2.threshold(dist_transform, 0.15 * dist_transform.max(), 255, 0)
    sure_fg = np.uint8(sure_fg)

    unknown = cv2.subtract(sure_bg, sure_fg)  # unknown region

    # Markers for Watershed
    _, markers = cv2.connectedComponents(sure_fg)
    markers = markers + 1
    markers[unknown == 255] = 0

    markers = cv2.watershed(image_bgr, markers)

    result = np.zeros(image_bgr.shape[:2], dtype=np.uint8)
    result[markers > 1] = 1  # Wheat = 1
    return result


# ---------------------------------------------------------------------------
# 2. Superpixel + ExG + Otsu
# ---------------------------------------------------------------------------

def segment_wheat_superpixel(image_rgb: np.ndarray,
                             n_segments: int = 800,
                             compactness: int = 20) -> np.ndarray:
    """
    SLIC superpixels + Excess-Green index + Otsu thresholding.

    For each SLIC superpixel we compute the mean Excess-Green (ExG) value,
    then apply Otsu's method on the distribution of superpixel-mean ExG
    values to find an automatic foreground/background threshold.
    """
    segments = slic(image_rgb, n_segments=n_segments,
                    compactness=compactness, sigma=1, start_label=1)

    # image values to 0.0 - 1.0 for index calculation
    img_float = image_rgb.astype(float) / 255.0
    R, G, B = img_float[:, :, 0], img_float[:, :, 1], img_float[:, :, 2]

    # Excess Green Index: highlights green vegetation,
    # pushes brown soil into negative values
    ExG = 2 * G - R - B

    result = np.zeros(image_rgb.shape[:2], dtype=np.uint8)
    unique_segments = np.unique(segments)

    # list for the average ExG values of each superpixel
    mean_exg_values = np.zeros(len(unique_segments))

    # calculate average ExG for each superpixel
    for i, seg_id in enumerate(unique_segments):
        mean_exg_values[i] = np.mean(ExG[segments == seg_id])

    # find automatic threshold via Otsu's method
    try:
        optimal_threshold = threshold_otsu(mean_exg_values)
    except ValueError:
        optimal_threshold = 0.05  # fallback if Otsu fails

    # Classify superpixels
    for i, seg_id in enumerate(unique_segments):
        if mean_exg_values[i] > optimal_threshold:
            result[segments == seg_id] = 1

    # remove noise
    kernel = np.ones((3, 3), np.uint8)
    result = cv2.morphologyEx(result, cv2.MORPH_OPEN, kernel)
    result = cv2.morphologyEx(result, cv2.MORPH_CLOSE, kernel)

    return result


# ---------------------------------------------------------------------------
# 3. Superpixel + noise-robust preprocessing
# ---------------------------------------------------------------------------

def segment_wheat_superpixel_noise_robust(image_rgb: np.ndarray,
                                          n_segments: int = 800,
                                          compactness: int = 20) -> np.ndarray:
    """
    Superpixel variant, specifically robust against noise.

    Core idea: Noise primarily disrupts pixel-wise ExG values.
    Solution: Strong edge-preserving denoising BEFORE SLIC and before ExG
    calculation.
      1. Bilateral Filter (edge-preserving, removes high-frequency noise)
      2. Additional Median Filter (robust against salt-and-pepper noise)
      3. Standard ExG + Otsu as in the original
    """
    # double denoising
    # Bilateral: removes Gaussian noise, preserves edges
    denoised = cv2.bilateralFilter(image_rgb, d=9, sigmaColor=75, sigmaSpace=75)
    # Median: additional smoothing, very effective against isolated outliers
    denoised = cv2.medianBlur(denoised, 5)

    # SLIC
    segments = slic(denoised, n_segments=n_segments,
                    compactness=compactness, sigma=1, start_label=1)

    # ExG calculation
    img_float = denoised.astype(float) / 255.0
    R, G, B = img_float[:, :, 0], img_float[:, :, 1], img_float[:, :, 2]
    ExG = 2 * G - R - B

    # per-superpixel mean + Otsu
    unique_segments = np.unique(segments)
    mean_exg_values = np.zeros(len(unique_segments))
    for i, seg_id in enumerate(unique_segments):
        mean_exg_values[i] = np.mean(ExG[segments == seg_id])

    try:
        optimal_threshold = threshold_otsu(mean_exg_values)
    except ValueError:
        optimal_threshold = 0.05

    # classification + morphology
    result = np.zeros(image_rgb.shape[:2], dtype=np.uint8)
    for i, seg_id in enumerate(unique_segments):
        if mean_exg_values[i] > optimal_threshold:
            result[segments == seg_id] = 1

    kernel = np.ones((3, 3), np.uint8)
    result = cv2.morphologyEx(result, cv2.MORPH_OPEN, kernel)
    result = cv2.morphologyEx(result, cv2.MORPH_CLOSE, kernel)

    return result