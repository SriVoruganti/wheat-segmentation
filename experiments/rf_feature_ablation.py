import os, sys, time, json
sys.path.append(os.path.dirname(os.path.dirname(__file__)))

from pathlib import Path
import numpy as np
import cv2
import matplotlib.pyplot as plt

from sklearn.ensemble import RandomForestClassifier

from data.dataset import EWSDatasetRF

# ---------- EWS helpers ----------
def load_image_rgb_u8(img_path: str) -> np.ndarray:
    bgr = cv2.imread(img_path, cv2.IMREAD_COLOR)
    return cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB)

def load_gt_mask01(mask_path: str) -> np.ndarray:
    m = cv2.imread(mask_path, cv2.IMREAD_UNCHANGED)
    if m.ndim == 3:
        m = m[:, :, 0]
    return (m == 0).astype(np.uint8)

def build_features(img_rgb_u8: np.ndarray, feature_mode: str):
    img = img_rgb_u8.astype(np.float32)
    rgb = (img / 255.0).astype(np.float32)
    feats = [rgb]

    if feature_mode in ("rgb_exg", "rgb_hsv_exg"):
        R, G, B = rgb[:, :, 0], rgb[:, :, 1], rgb[:, :, 2]
        exg = (2 * G - R - B)[:, :, None].astype(np.float32)
        feats.append(exg)

    if feature_mode in ("rgb_hsv", "rgb_hsv_exg"):
        hsv = cv2.cvtColor(img_rgb_u8, cv2.COLOR_RGB2HSV).astype(np.float32)
        hsv[:, :, 0] /= 179.0
        hsv[:, :, 1] /= 255.0
        hsv[:, :, 2] /= 255.0
        feats.append(hsv)

    feat_img = np.concatenate(feats, axis=2)
    H, W, C = feat_img.shape
    return feat_img.reshape(-1, C).astype(np.float32), (H, W)

def metrics_from_masks(pred01, gt01):
    pred = pred01.astype(bool)
    gt   = gt01.astype(bool)
    tp = np.logical_and(pred, gt).sum()
    fp = np.logical_and(pred, ~gt).sum()
    fn = np.logical_and(~pred, gt).sum()
    precision = tp / (tp + fp + 1e-8)
    recall    = tp / (tp + fn + 1e-8)
    f1        = 2 * precision * recall / (precision + recall + 1e-8)
    iou       = tp / (tp + fp + fn + 1e-8)
    return precision, recall, f1, iou

def eval_full_image(clf, feature_mode: str, data_root="data/EWS-Dataset"):
    test_root = Path(data_root) / "test"
    img_dir = test_root / "images"
    msk_dir = test_root / "masks"
    files = sorted([p.name for p in img_dir.glob("*.png")])

    Ps, Rs, F1s, IoUs, times = [], [], [], [], []

    for img_file in files:
        stem = Path(img_file).stem
        img = load_image_rgb_u8(str(img_dir / img_file))
        gt  = load_gt_mask01(str(msk_dir / f"{stem}_mask.png"))

        X, (H, W) = build_features(img, feature_mode)

        t0 = time.perf_counter()
        pred = clf.predict(X).reshape(H, W).astype(np.uint8)
        t1 = time.perf_counter()
        times.append((t1 - t0) * 1000)

        p, r, f1, iou = metrics_from_masks(pred, gt)
        Ps.append(p); Rs.append(r); F1s.append(f1); IoUs.append(iou)

    return {
        "precision": float(np.mean(Ps)),
        "recall": float(np.mean(Rs)),
        "f1": float(np.mean(F1s)),
        "iou": float(np.mean(IoUs)),
        "ms_img": float(np.mean(times)),
    }

def main():
    out_dir = Path("results/rf_full")
    fig_dir = out_dir / "figures"
    fig_dir.mkdir(parents=True, exist_ok=True)

    configs = [
        ("RF (RGB)", "rgb"),
        ("RF (RGB+ExG)", "rgb_exg"),
        ("RF (RGB+HSV)", "rgb_hsv"),
        ("RF (RGB+HSV+ExG)", "rgb_hsv_exg"),
    ]

    rows = []
    for label, mode in configs:
        print("\n===", label, "===")

        ds = EWSDatasetRF(
            root="data/EWS-Dataset",
            split="train",
            max_pixels_per_image=5000,
            seed=42,
            feature_mode=mode,
        )
        X_train, y_train = ds.load()

        clf = RandomForestClassifier(
            n_estimators=200,
            max_depth=None,
            n_jobs=-1,
            random_state=42,
        )

        t0 = time.perf_counter()
        clf.fit(X_train, y_train)
        t1 = time.perf_counter()

        m = eval_full_image(clf, mode)
        m["model"] = label
        m["feature_mode"] = mode
        m["train_time_s"] = float(t1 - t0)
        rows.append(m)

        print(f"IoU {m['iou']:.4f} | F1 {m['f1']:.4f} | ms/img {m['ms_img']:.1f}")

    # Save JSON
    out_path = out_dir / "rf_feature_ablation.json"
    with open(out_path, "w") as f:
        json.dump(rows, f, indent=2)
    print("Saved:", out_path)

    # Print markdown table
    print("\n| Model | Precision | Recall | F1 | IoU | ms/img |")
    print("|---|---:|---:|---:|---:|---:|")
    for r in rows:
        print(f"| {r['model']} | {r['precision']:.3f} | {r['recall']:.3f} | {r['f1']:.3f} | {r['iou']:.3f} | {r['ms_img']:.1f} |")

    # Plot IoU + F1
    labels = [r["model"] for r in rows]
    ious   = [r["iou"] for r in rows]
    f1s    = [r["f1"] for r in rows]
    x = np.arange(len(labels))

    plt.figure(figsize=(9,4))
    plt.plot(x, ious, marker="o", label="IoU")
    plt.plot(x, f1s, marker="s", label="F1-Score")
    plt.xticks(x, labels, rotation=15, ha="right")
    plt.ylim(0, 1)
    plt.ylabel("Score")
    plt.title("Handcrafted Feature Ablation — Random Forest")
    plt.grid(True, alpha=0.3)
    plt.legend()
    plt.tight_layout()
    plt.savefig(fig_dir / "rf_feature_ablation.png", dpi=200)
    print("Saved:", fig_dir / "rf_feature_ablation.png")

if __name__ == "__main__":
    main()
