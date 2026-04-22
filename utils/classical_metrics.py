"""
Evaluation metrics for the classical advanced-segmentation methods.

All functions operate on binary numpy masks (0/1 or bool) — no thresholding
or sigmoid is applied, since the classical methods already output binary
masks directly.

This intentionally duplicates functionality found in `utils/metrics.py`
(which is PyTorch-tensor-based and designed for raw logits from deep
learning models) because the two pipelines operate on different data types.
"""

import numpy as np


# ---------------------------------------------------------------------------
# Individual metrics
# ---------------------------------------------------------------------------

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


def calculate_precision(mask_true: np.ndarray, mask_pred: np.ndarray) -> float:
    """
    Precision = TP / (TP + FP).

    "Of all pixels predicted as wheat, how many were actually wheat?"
    Returns 0.0 if nothing was predicted as positive.
    """
    mask_true = mask_true.astype(bool)
    mask_pred = mask_pred.astype(bool)

    tp = np.sum(mask_true & mask_pred)
    fp = np.sum(~mask_true & mask_pred)

    if tp + fp == 0:
        return 0.0
    return float(tp / (tp + fp))


def calculate_recall(mask_true: np.ndarray, mask_pred: np.ndarray) -> float:
    """
    Recall = TP / (TP + FN).

    "Of all actual wheat pixels, how many did we find?"
    Returns 0.0 if there are no positives in the ground truth.
    """
    mask_true = mask_true.astype(bool)
    mask_pred = mask_pred.astype(bool)

    tp = np.sum(mask_true & mask_pred)
    fn = np.sum(mask_true & ~mask_pred)

    if tp + fn == 0:
        return 0.0
    return float(tp / (tp + fn))


def calculate_f1(mask_true: np.ndarray, mask_pred: np.ndarray) -> float:
    """
    F1-score = harmonic mean of precision and recall
             = 2 * P * R / (P + R)
    """
    p = calculate_precision(mask_true, mask_pred)
    r = calculate_recall(mask_true, mask_pred)
    if p + r == 0:
        return 0.0
    return float(2 * p * r / (p + r))


# ---------------------------------------------------------------------------
# Combined — returns all four metrics at once
# ---------------------------------------------------------------------------

def calculate_all_metrics(mask_true: np.ndarray,
                          mask_pred: np.ndarray) -> dict:
    """
    Compute Precision, Recall, F1 and IoU from a single confusion matrix.
    (More efficient than calling the four helpers separately)

    Returns
    -------
    dict with keys {'precision', 'recall', 'f1', 'iou'}
    """
    mask_true = mask_true.astype(bool)
    mask_pred = mask_pred.astype(bool)

    tp = np.sum(mask_true & mask_pred)
    fp = np.sum(~mask_true & mask_pred)
    fn = np.sum(mask_true & ~mask_pred)

    precision = tp / (tp + fp) if (tp + fp) > 0 else 0.0
    recall    = tp / (tp + fn) if (tp + fn) > 0 else 0.0
    f1        = (2 * precision * recall / (precision + recall)
                 if (precision + recall) > 0 else 0.0)
    iou       = tp / (tp + fp + fn) if (tp + fp + fn) > 0 else 0.0

    return {
        "precision": float(precision),
        "recall":    float(recall),
        "f1":        float(f1),
        "iou":       float(iou),
    }