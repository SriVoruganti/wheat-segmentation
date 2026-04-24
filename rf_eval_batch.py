import os
import numpy as np
import joblib
from skimage.color import rgb2hsv

from rf_eval_common import read_rgb, read_mask_ews, iou, f1_score, load_pairs_ews, summarise


def pixel_features(img_rgb: np.ndarray, mode: str = "rgb") -> np.ndarray:
    x = img_rgb.astype(np.float32) / 255.0
    R, G, B = x[..., 0], x[..., 1], x[..., 2]
    feats = [R, G, B]

    if "exg" in mode:
        feats.append(2 * G - R - B)

    if "hsv" in mode:
        hsv = rgb2hsv(x)
        feats += [hsv[..., 0], hsv[..., 1], hsv[..., 2]]

    F = np.stack(feats, axis=-1)
    return F.reshape(-1, F.shape[-1]).astype(np.float32)


def segment_rf(img_rgb: np.ndarray, rf_model, feature_mode: str = "rgb") -> np.ndarray:
    H, W = img_rgb.shape[:2]
    X = pixel_features(img_rgb, mode=feature_mode)
    y = rf_model.predict(X)
    return y.reshape(H, W).astype(np.uint8)


def main():
    DATA_ROOT = os.environ.get("EWS_ROOT", "/path/to/EWS")
    SPLIT = os.environ.get("EWS_SPLIT", "test")
    MODEL_PATH = os.environ.get("RF_MODEL", "rf_rgb.joblib")
    FEATURE_MODE = os.environ.get("RF_FEATURE_MODE", "rgb")

    rf = joblib.load(MODEL_PATH)

    pairs, missing = load_pairs_ews(DATA_ROOT, SPLIT)
    print(f"Eval split={SPLIT} images={len(pairs)} missing_masks={missing}")
    print(f"model={MODEL_PATH} feature_mode={FEATURE_MODE}")

    ious, f1s = [], []
    for img_path, mask_path in pairs:
        img = read_rgb(img_path)
        gt = read_mask_ews(mask_path)
        pred = segment_rf(img, rf, feature_mode=FEATURE_MODE)
        ious.append(iou(pred, gt))
        f1s.append(f1_score(pred, gt))

    summarise(np.array(ious), "IoU")
    summarise(np.array(f1s), "F1")


if __name__ == "__main__":
    main()