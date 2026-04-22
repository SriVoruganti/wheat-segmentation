"""
Synthetic image distortions used for robustness testing of the classical
advanced-segmentation methods.

Available distortions:
  - add_blur                     : Gaussian blur
  - add_noise                    : Additive Gaussian noise
  - adjust_brightness_contrast   : Lower brightness + contrast ("dark" image)
  - add_partial_occlusion        : Grey rectangular patches covering parts of
                                   the image
"""

import random
import cv2
import numpy as np


def add_blur(image: np.ndarray, ksize: int = 9) -> np.ndarray:
    """Gaussian blur with an ksize by ksize kernel."""
    return cv2.GaussianBlur(image, (ksize, ksize), 0)


def add_noise(image: np.ndarray, mean: float = 0, std: float = 30) -> np.ndarray:
    """Additive Gaussian noise with given mean and standard deviation."""
    noise = np.random.normal(mean, std, image.shape).astype(np.float32)
    noisy_image = cv2.add(image.astype(np.float32), noise)
    return np.clip(noisy_image, 0, 255).astype(np.uint8)


def adjust_brightness_contrast(image: np.ndarray,
                               alpha: float = 0.6,
                               beta: float = -40) -> np.ndarray:
    """
    simulate bad lighting.
    alpha < 1 lowers contrast, beta < 0 makes the image darker.
    """
    return cv2.convertScaleAbs(image, alpha=alpha, beta=beta)


def add_partial_occlusion(image: np.ndarray,
                          num_blocks: int = 8,
                          block_size: int = 60) -> np.ndarray:
    """Cover random rectangular regions with solid grey patches."""
    occluded = image.copy()
    h, w = occluded.shape[:2]
    for _ in range(num_blocks):
        x = random.randint(0, w - block_size)
        y = random.randint(0, h - block_size)
        cv2.rectangle(
            occluded,
            (x, y),
            (x + block_size, y + block_size),
            (128, 128, 128),
            -1,
        )
    return occluded