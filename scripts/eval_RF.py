"""
Final test set evaluation for Random Forest wheat pixel classifier.

- Per-image metrics (Precision/Recall/F1/IoU) + mean/std across images
- Optional flip-augmentation (horizontal flip + majority vote)
- Optional visualisation + failure analysis
- Saves JSON metrics for reproducibility

Example:
  python -m scripts.eval_RF ^
    --data_root ./data/EWS-Dataset ^
    --model_path ./results/rf_model.joblib ^
    --feature_mode rgb_hsv_exg ^
    --flip_aug --visualise --failure_analysis
"""

import time
import json
import argparse
from pathlib import Path

import numpy as np
import cv2
import joblib

from scripts.rf_common import (
    load_pairs_ews,
    read_rgb,
    read_mask_ews,
    segment_rf,
)


# Metrics (per-image)
def compute_metrics_from_masks(pred01: np.ndarray, gt01: np.ndarray) -> dict:
    pred01 = pred01.astype(bool)
    gt01 = gt01.astype(bool)

    tp = np.logical_and(pred01, gt01).sum()
    fp = np.logical_and(pred01, ~gt01).sum()
    fn = np.logical_and(~pred01, gt01).sum()

    precision = tp / (tp + fp + 1e-8)
    recall = tp / (tp + fn + 1e-8)
    f1 = 2 * precision * recall / (precision + recall + 1e-8)
    iou = tp / (tp + fp + fn + 1e-8)

    return {
        "precision": float(precision),
        "recall": float(recall),
        "f1": float(f1),
        "iou": float(iou),
    }


def aggregate(metrics_list: list) -> dict:
    out = {}
    for k in ["precision", "recall", "f1", "iou"]:
        vals = np.array([m[k] for m in metrics_list], dtype=np.float64)
        out[k] = float(vals.mean())
        out[k + "_std"] = float(vals.std())
    return out


# Visualisation helpers
def overlay_pred(img_rgb: np.ndarray, pred01: np.ndarray) -> np.ndarray:
    overlay = img_rgb.copy()
    red = np.array([255, 0, 0], dtype=np.uint8)
    overlay[pred01 == 1] = (0.5 * overlay[pred01 == 1] + 0.5 * red).astype(np.uint8)
    return overlay


def save_error_map(gt01: np.ndarray, pred01: np.ndarray, save_path: str):
    """
    Error map:
      TP = white, FP = red, FN = blue
    """
    H, W = gt01.shape
    out = np.zeros((H, W, 3), dtype=np.uint8)

    tp = (pred01 == 1) & (gt01 == 1)
    fp = (pred01 == 1) & (gt01 == 0)
    fn = (pred01 == 0) & (gt01 == 1)

    out[tp] = (255, 255, 255)  # white
    out[fp] = (255, 0, 0)      # red
    out[fn] = (0, 0, 255)      # blue

    cv2.imwrite(save_path, cv2.cvtColor(out, cv2.COLOR_RGB2BGR))


# Prediction (with optional flip aug)
def predict_mask01(rf_model, img_rgb: np.ndarray, feature_mode: str) -> np.ndarray:
    return segment_rf(img_rgb, rf_model, feature_mode=feature_mode)


def predict_mask01_flip_aug(rf_model, img_rgb: np.ndarray, feature_mode: str) -> np.ndarray:
    """
    RF 'TTA-like' option: predict on original + horizontal flip, then majority vote.
    """
    pred0 = predict_mask01(rf_model, img_rgb, feature_mode)

    img_flip = np.ascontiguousarray(img_rgb[:, ::-1, :])
    pred1 = predict_mask01(rf_model, img_flip, feature_mode)[:, ::-1]

    return ((pred0 + pred1) >= 1).astype(np.uint8)


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--data_root", type=str, default="./data/EWS-Dataset")
    p.add_argument("--split", type=str, default="test")
    p.add_argument("--model_path", type=str, default="./results/rf_model_rgb_hsv_exg.pkl")
    p.add_argument("--feature_mode", type=str, default="rgb_hsv_exg")

    p.add_argument("--output_dir", type=str, default="./results/rf_full")

    p.add_argument("--flip_aug", action="store_true", help="Enable flip-augmentation voting")
    p.add_argument("--visualise", action="store_true", help="Save a few prediction examples (first n_vis)")
    p.add_argument("--failure_analysis", action="store_true", help="Save worst predictions by IoU")

    p.add_argument("--n_vis", type=int, default=6)
    p.add_argument("--n_fail", type=int, default=6)
    return p.parse_args()


def main():
    args = parse_args()

    out_dir = Path(args.output_dir)
    fig_dir = out_dir / "figures"
    fig_dir.mkdir(parents=True, exist_ok=True)

    # Load pairs robustly (no assumptions about only PNG)
    pairs, missing = load_pairs_ews(args.data_root, split=args.split)
    print(f"Split={args.split} pairs={len(pairs)} missing_masks={missing}")
    if len(pairs) == 0:
        raise RuntimeError(
            f"No (image, mask) pairs found. Check --data_root and folder layout: {args.data_root}"
        )

    # Load model
    model_path = Path(args.model_path)
    if not model_path.exists():
        raise FileNotFoundError(f"Model not found: {model_path}")

    rf_model = joblib.load(str(model_path))
    print(f"Model: {model_path}")
    print(f"Feature mode: {args.feature_mode}")
    if hasattr(rf_model, "n_features_in_"):
        print(f"Model expects n_features_in_={rf_model.n_features_in_}")

    # Evaluate
    per_image = []
    inference_times_ms = []
    saved_examples = 0

    for img_path, mask_path in pairs:
        img_rgb = read_rgb(img_path)
        gt01 = read_mask_ews(mask_path)

        t0 = time.perf_counter()
        if args.flip_aug:
            pred01 = predict_mask01_flip_aug(rf_model, img_rgb, args.feature_mode)
        else:
            pred01 = predict_mask01(rf_model, img_rgb, args.feature_mode)
        t1 = time.perf_counter()
        inference_times_ms.append((t1 - t0) * 1000.0)

        m = compute_metrics_from_masks(pred01, gt01)
        m["file"] = Path(img_path).name
        per_image.append(m)

        # Save a few examples (first n_vis)
        if args.visualise and saved_examples < args.n_vis:
            stem = Path(img_path).stem

            pred_path = fig_dir / f"{stem}_pred.png"
            gt_path = fig_dir / f"{stem}_gt.png"
            ov_path = fig_dir / f"{stem}_overlay.png"
            err_path = fig_dir / f"{stem}_error.png"

            cv2.imwrite(str(pred_path), pred01 * 255)
            cv2.imwrite(str(gt_path), gt01 * 255)
            cv2.imwrite(str(ov_path), cv2.cvtColor(overlay_pred(img_rgb, pred01), cv2.COLOR_RGB2BGR))
            save_error_map(gt01, pred01, str(err_path))

            saved_examples += 1

    summary = aggregate(per_image)
    summary["avg_inference_time_ms"] = float(np.mean(inference_times_ms))
    summary["std_inference_time_ms"] = float(np.std(inference_times_ms))
    summary["n_images"] = int(len(per_image))
    summary["feature_mode"] = args.feature_mode
    summary["flip_aug"] = bool(args.flip_aug)

    tag = "flip_aug" if args.flip_aug else "no_aug"

    # Print table (similar to DL)
    print("\n" + "=" * 72)
    print(f"{'':30} {'Precision':>9} {'Recall':>8} {'F1':>8} {'IoU':>8} {'ms/img':>8}")
    print("=" * 72)
    print(
        f"  {tag:<28} {summary['precision']:>9.4f} {summary['recall']:>8.4f} "
        f"{summary['f1']:>8.4f} {summary['iou']:>8.4f} {summary['avg_inference_time_ms']:>8.2f}"
    )
    print("=" * 72)

    # Failure analysis (worst IoU)
    if args.failure_analysis:
        worst = sorted(per_image, key=lambda x: x["iou"])[: args.n_fail]
        fail_dir = fig_dir / "failures"
        fail_dir.mkdir(parents=True, exist_ok=True)

        for item in worst:
            img_file = item["file"]
            img_path = next(p[0] for p in pairs if Path(p[0]).name == img_file)
            mask_path = next(p[1] for p in pairs if Path(p[0]).name == img_file)

            img_rgb = read_rgb(img_path)
            gt01 = read_mask_ews(mask_path)

            pred01 = (
                predict_mask01_flip_aug(rf_model, img_rgb, args.feature_mode)
                if args.flip_aug
                else predict_mask01(rf_model, img_rgb, args.feature_mode)
            )

            stem = Path(img_path).stem
            ov = overlay_pred(img_rgb, pred01)

            cv2.imwrite(str(fail_dir / f"{stem}_overlay.png"), cv2.cvtColor(ov, cv2.COLOR_RGB2BGR))
            save_error_map(gt01, pred01, str(fail_dir / f"{stem}_error.png"))
            cv2.imwrite(str(fail_dir / f"{stem}_pred.png"), pred01 * 255)
            cv2.imwrite(str(fail_dir / f"{stem}_gt.png"), gt01 * 255)

        print(f"\nSaved failure analysis → {fail_dir}")

    # Save JSON
    out_path = out_dir / f"test_metrics_rf_{tag}.json"
    out_dir.mkdir(parents=True, exist_ok=True)
    with open(out_path, "w") as f:
        json.dump({"summary": summary, "per_image": per_image}, f, indent=2)
    print(f"\nMetrics saved to {out_path}")

    if args.visualise:
        print(f"Saved prediction examples → {fig_dir}")


if __name__ == "__main__":
    main()
