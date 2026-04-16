# experiments/rf_robustness.py
import os, sys, time, json
sys.path.append(os.path.dirname(os.path.dirname(__file__)))

from pathlib import Path
import numpy as np
import cv2
import matplotlib.pyplot as plt

from models.random_forest import load_model


# -----------------------------
# Helpers: load image + mask
# -----------------------------
def load_image_rgb_u8(p: Path) -> np.ndarray:
    bgr = cv2.imread(str(p), cv2.IMREAD_COLOR)
    if bgr is None:
        raise FileNotFoundError(f"Could not read image: {p}")
    return cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB)

def load_gt_mask01(p: Path) -> np.ndarray:
    m = cv2.imread(str(p), cv2.IMREAD_UNCHANGED)
    if m is None:
        raise FileNotFoundError(f"Could not read mask: {p}")
    if m.ndim == 3:
        m = m[:, :, 0]
    return (m == 0).astype(np.uint8)  # plant=1 where ch0==0


# -----------------------------
# Features (match ablation)
# -----------------------------
def build_features(img_rgb_u8: np.ndarray, feature_mode: str):
    """
    Returns:
        X: (H*W, C)
        (H, W)
    """
    rgb = (img_rgb_u8.astype(np.float32) / 255.0)
    feats = [rgb]  # always include RGB

    if feature_mode in ("rgb_exg", "rgb_hsv_exg"):
        R, G, B = rgb[:, :, 0], rgb[:, :, 1], rgb[:, :, 2]
        exg = (2.0 * G - R - B)[:, :, None].astype(np.float32)
        feats.append(exg)

    if feature_mode in ("rgb_hsv", "rgb_hsv_exg"):
        hsv = cv2.cvtColor(img_rgb_u8, cv2.COLOR_RGB2HSV).astype(np.float32)
        hsv[:, :, 0] /= 179.0
        hsv[:, :, 1] /= 255.0
        hsv[:, :, 2] /= 255.0
        feats.append(hsv)

    feat_img = np.concatenate(feats, axis=2)
    H, W, C = feat_img.shape
    X = feat_img.reshape(-1, C).astype(np.float32)
    return X, (H, W)

def predict_mask01(clf, img_rgb_u8: np.ndarray, feature_mode: str) -> np.ndarray:
    X, (H, W) = build_features(img_rgb_u8, feature_mode)
    pred = clf.predict(X).reshape(H, W).astype(np.uint8)
    return pred


# -----------------------------
# Metrics
# -----------------------------
def metrics_iou_f1(gt01: np.ndarray, pred01: np.ndarray):
    gt = gt01.astype(bool)
    pr = pred01.astype(bool)

    tp = np.logical_and(pr, gt).sum()
    fp = np.logical_and(pr, ~gt).sum()
    fn = np.logical_and(~pr, gt).sum()

    iou = tp / (tp + fp + fn + 1e-8)
    precision = tp / (tp + fp + 1e-8)
    recall = tp / (tp + fn + 1e-8)
    f1 = 2 * precision * recall / (precision + recall + 1e-8)
    return float(iou), float(f1)


# -----------------------------
# Distortions
# -----------------------------
def clip_u8(x):
    return np.clip(x, 0, 255).astype(np.uint8)

def low_contrast(img, alpha=0.6):
    # pull values toward mid-gray
    return clip_u8(128 + alpha * (img.astype(np.float32) - 128))

def low_brightness(img, beta=-40):
    return clip_u8(img.astype(np.int16) + int(beta))

def gaussian_noise(img, sigma=10, rng=None):
    if rng is None:
        rng = np.random.default_rng(0)
    noise = rng.normal(0, sigma, img.shape).astype(np.float32)
    return clip_u8(img.astype(np.float32) + noise)

def blur(img, k=5):
    k = int(k)
    if k % 2 == 0:
        k += 1
    return cv2.GaussianBlur(img, (k, k), 0)

def occlusion(img, n=8, size=30, rng=None):
    if rng is None:
        rng = np.random.default_rng(0)
    out = img.copy()
    H, W, _ = out.shape
    size = int(size)
    for _ in range(int(n)):
        y = int(rng.integers(0, max(1, H - size)))
        x = int(rng.integers(0, max(1, W - size)))
        out[y:y+size, x:x+size] = 0
    return out

def jpeg_compress(img, quality=20):
    encode_param = [int(cv2.IMWRITE_JPEG_QUALITY), int(quality)]
    ok, enc = cv2.imencode(".jpg", cv2.cvtColor(img, cv2.COLOR_RGB2BGR), encode_param)
    if not ok:
        return img
    dec = cv2.imdecode(enc, cv2.IMREAD_COLOR)
    return cv2.cvtColor(dec, cv2.COLOR_BGR2RGB)

def make_distortions(seed=0):
    rng = np.random.default_rng(seed)
    return [
        ("clean", lambda x: x),
        ("low_contrast", lambda x: low_contrast(x, alpha=0.6)),
        ("low_brightness", lambda x: low_brightness(x, beta=-40)),
        ("gaussian_noise_mild", lambda x: gaussian_noise(x, sigma=10, rng=rng)),
        ("gaussian_noise_strong", lambda x: gaussian_noise(x, sigma=25, rng=rng)),
        ("occlusion", lambda x: occlusion(x, n=8, size=30, rng=rng)),
        ("blur_mild", lambda x: blur(x, k=5)),
        ("blur_strong", lambda x: blur(x, k=11)),
        ("jpeg_compression", lambda x: jpeg_compress(x, quality=20)),
    ]


# -----------------------------
# Main
# -----------------------------
def main():
    # IMPORTANT: set this to match how your saved model was trained
    feature_mode = "rgb"  # change to "rgb_hsv_exg" if your saved model expects 7 features

    data_root = Path("data/EWS-Dataset")
    img_dir = data_root / "test" / "images"
    msk_dir = data_root / "test" / "masks"

    image_files = sorted([p.name for p in img_dir.glob("*.png")])
    print("Found test images:", len(image_files), "in", img_dir)
    if len(image_files) == 0:
        raise FileNotFoundError(f"No .png images found in {img_dir}. Check dataset path/preprocess.")

    out_dir = Path("results/rf_full")
    fig_dir = out_dir / "figures"
    fig_dir.mkdir(parents=True, exist_ok=True)

    # Load trained RF model
    model_path = Path("results/rf_model.pkl")
    if not model_path.exists():
        raise FileNotFoundError(f"Model not found: {model_path} (update path in rf_robustness.py)")

    clf = load_model(str(model_path))

    # Optional: sanity check expected feature dimension
    if hasattr(clf, "n_features_in_"):
        print("Model expects n_features_in_ =", clf.n_features_in_)
        # quick warning if mismatch
        expected = clf.n_features_in_
        mode_dims = {"rgb": 3, "rgb_exg": 4, "rgb_hsv": 6, "rgb_hsv_exg": 7}
        if feature_mode in mode_dims and mode_dims[feature_mode] != expected:
            print(f"WARNING: feature_mode='{feature_mode}' gives {mode_dims[feature_mode]} dims, "
                  f"but model expects {expected}. Update feature_mode or retrain/resave model.")

    distortions = make_distortions(seed=0)

    results = []
    for name, fn in distortions:
        ious, f1s = [], []
        t0 = time.perf_counter()

        for img_file in image_files:
            stem = Path(img_file).stem
            img = load_image_rgb_u8(img_dir / img_file)
            gt  = load_gt_mask01(msk_dir / f"{stem}_mask.png")

            img_d = fn(img)
            pred = predict_mask01(clf, img_d, feature_mode=feature_mode)

            iou_v, f1_v = metrics_iou_f1(gt, pred)
            ious.append(iou_v)
            f1s.append(f1_v)

        t1 = time.perf_counter()

        row = {
            "distortion": name,
            "iou": float(np.mean(ious)),
            "f1": float(np.mean(f1s)),
            "std_iou": float(np.std(ious)),
            "ms_img": float((t1 - t0) * 1000 / len(image_files)),
            "feature_mode": feature_mode,
        }
        results.append(row)
        print(f"{name:>18s} | IoU {row['iou']:.4f} | F1 {row['f1']:.4f} | ms/img {row['ms_img']:.1f}")

    # Save JSON
    out_path = out_dir / "rf_robustness.json"
    with open(out_path, "w") as f:
        json.dump(results, f, indent=2)
    print("Saved:", out_path)

    # Plot IoU bars + clean dashed line
    clean_iou = [r["iou"] for r in results if r["distortion"] == "clean"][0]
    labels = [r["distortion"] for r in results]
    vals   = [r["iou"] for r in results]

    plt.figure(figsize=(10, 4.5))
    plt.bar(range(len(vals)), vals, color="#4C78A8")
    plt.axhline(clean_iou, linestyle="--", color="black", linewidth=1, label=f"Clean IoU ({clean_iou:.3f})")
    plt.ylim(0, 1)
    plt.ylabel("IoU")
    plt.title("Robustness Under Image Distortions — Random Forest")
    plt.xticks(range(len(labels)), labels, rotation=25, ha="right")
    plt.legend()
    plt.tight_layout()

    fig_path = fig_dir / "rf_robustness.png"
    plt.savefig(fig_path, dpi=200)
    print("Saved:", fig_path)


if __name__ == "__main__":
    main()
