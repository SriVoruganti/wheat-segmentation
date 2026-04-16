import os
from pathlib import Path
import numpy as np
import cv2

def check_split(root="data/EWS-Dataset", split="train"):
    img_dir = Path(root) / split / "images"
    msk_dir = Path(root) / split / "masks"

    imgs = sorted([p for p in img_dir.glob("*.png")])
    msks = sorted([p for p in msk_dir.glob("*.png")])

    print(f"\n== {split} ==")
    print("images:", len(imgs), "masks:", len(msks))
    assert img_dir.exists() and msk_dir.exists(), "Missing images/ or masks/ folder"

    # map stems
    img_stems = {p.stem for p in imgs}
    # masks are usually xxx_mask.png
    msk_stems = {p.stem.replace("_mask", "") for p in msks}

    orphan_imgs = sorted(list(img_stems - msk_stems))
    orphan_msks = sorted(list(msk_stems - img_stems))

    print("orphan images:", len(orphan_imgs))
    print("orphan masks :", len(orphan_msks))
    if orphan_imgs[:5]: print(" sample orphan img:", orphan_imgs[:5])
    if orphan_msks[:5]: print(" sample orphan msk:", orphan_msks[:5])

    # inspect a few masks for channel/values + plant ratio
    ratios = []
    for p in msks[:20]:
        m = cv2.imread(str(p), cv2.IMREAD_UNCHANGED)
        if m.ndim == 3:
            m0 = m[:, :, 0]
        else:
            m0 = m
        plant = (m0 == 0).astype(np.uint8)
        ratios.append(plant.mean())

        uniq = np.unique(m0)
        print(f"{p.name}: shape={m.shape}, ch0 uniq(min..max)={uniq.min()}..{uniq.max()}, plant%={plant.mean()*100:.1f}")

    if ratios:
        ratios = np.array(ratios)
        print(f"plant% (first 20 masks): mean={ratios.mean()*100:.1f} std={ratios.std()*100:.1f}")

if __name__ == "__main__":
    for split in ["train", "val", "test"]:
        check_split(split=split)
