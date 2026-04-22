"""
Evaluation metric(s) for the classical advanced-segmentation methods.

Only IoU is used here to reproduce the exact output of the original
notebook. Precision/Recall/F1 are computed separately in the
`utils/metrics.py` module provided by the rest of the project.
"""

import numpy as np


def calculate_iou(mask_true: np.ndarray, mask_pred: np.ndarray) -> float:
    """
    Intersection over Union (IoU) between a ground-truth and a predicted
    binary mask.

    Returns 0.0 if the union is empty (both masks are all zeros).
    """
    intersection = np.logical_and(mask_true, mask_pred)
    union = np.logical_or(mask_true, mask_pred)

    if np.sum(union) == 0:
        return 0.0

    return float(np.sum(intersection) / np.sum(union))