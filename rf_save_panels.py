import os
import numpy as np
import joblib
import matplotlib.pyplot as plt
from skimage.color import rgb2hsv

from rf_eval_common import read_rgb, read_mask_ews, iou, load_pairs_ews


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


def segment_rf(img_rgb, rf_model, feature_mode="rgb"):
    H, W = img_rgb.shape[:2]
    X = pixel_features(img_rgb, mode=feature_mode)
    y = rf_model.predict(X)
    return y.reshape(H, W).astype(np.uint8)


def save_panel(img, gt, pred, out_path, title=""):
    plt.figure(figsize=(9, 3))
    plt.subplot(1, 3, 1)
    plt.imshow(img)
    plt.title("Original")
    plt.axis("off")

    plt.subplot(1, 3, 2)
    plt.imshow(gt, cmap="gray")
    plt.title("Ground Truth")
    plt.axis("off")

    plt.subplot(1, 3, 3)
    plt.imshow(pred, cmap="gray")
    plt.title("RF Prediction")
    plt.axis("off")

    plt.suptitle(title)
    plt.tight_layout()
    plt.savefig(out_path, dpi=200)
    plt.close()

def combine_panels(panel_paths, out_path, layout="vertical"):
    """
    Combine already-saved panel PNGs into one image.
    layout: "vertical" (3 rows) or "horizontal" (3 columns)
    """
    import matplotlib.image as mpimg

    imgs = [mpimg.imread(p) for p in panel_paths]  # RGB(A), float [0,1] or uint8

    # Ensure all same width (for vertical) or same height (for horizontal)
    if layout == "vertical":
        target_w = max(im.shape[1] for im in imgs)
        resized = []
        for im in imgs:
            h, w = im.shape[:2]
            if w != target_w:
                # resize with matplotlib-friendly path via PIL (no extra deps if pillow exists)
                from PIL import Image
                im_pil = Image.fromarray((im * 255).astype(np.uint8)) if im.dtype != np.uint8 else Image.fromarray(im)
                new_h = int(round(h * (target_w / w)))
                im_pil = im_pil.resize((target_w, new_h), resample=Image.BILINEAR)
                im = np.array(im_pil)
            resized.append(im)
        combined = np.vstack(resized)
    else:
        target_h = max(im.shape[0] for im in imgs)
        resized = []
        for im in imgs:
            h, w = im.shape[:2]
            if h != target_h:
                from PIL import Image
                im_pil = Image.fromarray((im * 255).astype(np.uint8)) if im.dtype != np.uint8 else Image.fromarray(im)
                new_w = int(round(w * (target_h / h)))
                im_pil = im_pil.resize((new_w, target_h), resample=Image.BILINEAR)
                im = np.array(im_pil)
            resized.append(im)
        combined = np.hstack(resized)

    plt.figure(figsize=(12, 9) if layout == "vertical" else (18, 4))
    plt.imshow(combined)
    plt.axis("off")
    plt.tight_layout(pad=0)
    plt.savefig(out_path, dpi=200, bbox_inches="tight", pad_inches=0)
    plt.close()

def main():
    DATA_ROOT = os.environ.get("EWS_ROOT", "/path/to/EWS")
    SPLIT = os.environ.get("EWS_SPLIT", "test")
    MODEL_PATH = os.environ.get("RF_MODEL", "rf_rgb.joblib")
    FEATURE_MODE = os.environ.get("RF_FEATURE_MODE", "rgb")
    OUT_DIR = os.environ.get("PANEL_DIR", "rf_panels")

    os.makedirs(OUT_DIR, exist_ok=True)
    rf = joblib.load(MODEL_PATH)

    pairs, _ = load_pairs_ews(DATA_ROOT, SPLIT)

    ious = []
    cache = []
    for img_path, mask_path in pairs:
        img = read_rgb(img_path)
        gt = read_mask_ews(mask_path)
        pred = segment_rf(img, rf, feature_mode=FEATURE_MODE)
        score = iou(pred, gt)
        ious.append(score)
        cache.append((img_path, mask_path, pred, score))

    ious = np.array(ious)
    order = np.argsort(ious)
    picks = [order[0], order[len(order)//2], order[-1]]
    tags = ["worst", "median", "best"]

    panel_paths = []
    for idx, tag in zip(picks, tags):
        img_path, mask_path, pred, score = cache[idx]
        img = read_rgb(img_path)
        gt = read_mask_ews(mask_path)
        out = os.path.join(OUT_DIR, f"{tag}_{idx:02d}_iou_{score:.3f}.png")
        save_panel(img, gt, pred, out, title=f"{tag} example — IoU={score:.3f}")
        panel_paths.append(out)

    # Combine into one image (worst, median, best stacked)
    combined_out = os.path.join(OUT_DIR, "rf_examples_combined.png")
    combine_panels(panel_paths, combined_out, layout="vertical")

    print("Saved panels to:", OUT_DIR)
    print("Saved combined panel to:", combined_out)

if __name__ == "__main__":
    main()