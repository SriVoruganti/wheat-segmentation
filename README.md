# comp9517-wheat-segmentation (Deep Learning — HD)

Deep learning-based wheat crop segmentation using the EWS dataset — UNSW COMP9517 Group Project 2026 T1

---

## Project Structure

```
wheat-seg/
├── data/
│   ├── dataset.py          # EWSDataset with subset & label-noise support
│   └── distortions.py      # Synthetic distortion functions for robustness testing
├── models/
│   ├── unet.py             # Vanilla U-Net (training from scratch)
│   ├── unet_pretrained.py  # U-Net with pretrained ResNet-34 encoder (recommended)
│   └── losses.py           # BCE, Dice, Focal, Tversky, Combo, FocalDice losses
├── utils/
│   ├── metrics.py          # Precision, Recall, F1, IoU (shared across all methods)
│   ├── tta.py              # Test-Time Augmentation
│   └── visualise.py        # Prediction grids, failure analysis, training curves
├── experiments/
│   ├── robustness_eval.py  # HD: evaluate under noise/blur/occlusion/compression
│   └── data_scarcity.py    # HD: train with 25/50/75/100% data + label noise
├── results/
│   └── figures/
├── train.py                # Main training script (supports two-phase training)
├── evaluate.py             # Test evaluation with TTA + failure analysis
├── requirements.txt
└── README.md
```

---

## Dataset Setup

Download the EWS dataset and structure it as:

```
EWS-Dataset/
├── train/images/   train/masks/
├── val/images/     val/masks/
└── test/images/    test/masks/
```

> ⚠️ Never mix splits. The test set is used **only** for final evaluation.

---

## Installation

```bash
pip install -r requirements.txt
```

---

## Training

### Recommended: Pretrained Encoder with Two-Phase Training

```bash
python train.py \
    --model pretrained \
    --data_root ./EWS-Dataset \
    --loss focal_dice \
    --epochs 40 \
    --two_phase \
    --phase1_epochs 10
```

Phase 1 (epochs 1–10): encoder frozen, decoder trains quickly.  
Phase 2 (epochs 11–40): full network fine-tuned end-to-end at 10× lower LR.

### Baseline: Vanilla U-Net (from scratch)

```bash
python train.py \
    --model unet \
    --data_root ./EWS-Dataset \
    --loss combo \
    --epochs 60
```

### Loss Options

| Loss | Best for |
|---|---|
| `focal_dice` | Class imbalance + boundary precision (default) |
| `tversky` | Maximising recall (thin wheat structures) |
| `combo` | Simple, reliable baseline |

---

## Evaluation

```bash
python evaluate.py \
    --data_root ./EWS-Dataset \
    --checkpoint ./results/pretrained_focal_dice/best.pth \
    --model pretrained \
    --tta \
    --visualise \
    --failure_analysis \
    --history_path ./results/pretrained_focal_dice/history.json
```

---

## Experiments

### Robustness Testing

```bash
python experiments/robustness_eval.py \
    --data_root ./EWS-Dataset \
    --checkpoint ./results/pretrained_focal_dice/best.pth \
    --model pretrained
```

Tests performance under: Gaussian noise (mild/strong), blur (mild/strong),
low brightness, low contrast, partial occlusion, JPEG compression.

### Data Scarcity Analysis

```bash
# Effect of training set size
python experiments/data_scarcity.py \
    --data_root ./EWS-Dataset \
    --model pretrained \
    --epochs 30

# Effect of label noise
python experiments/data_scarcity.py \
    --data_root ./EWS-Dataset \
    --model pretrained \
    --label_noise 0.1 \
    --epochs 30
```

---

## Methods (Deep Learning)

### Vanilla U-Net (`models/unet.py`)
Encoder-decoder with skip connections, BatchNorm, spatial Dropout. Trained from scratch.

### Pretrained U-Net (`models/unet_pretrained.py`)
ResNet-34 ImageNet encoder + lightweight decoder with skip connections.
Two-phase training: freeze encoder → unfreeze and fine-tune.
Recommended for small datasets like EWS.

### Advanced Loss Functions (`models/losses.py`)
- **Focal Loss** — penalises easy background pixels, focuses on hard wheat boundaries
- **Tversky Loss** — β > α to prioritise recall over precision
- **FocalDice** — combined; best empirical performance on EWS

### Test-Time Augmentation (`utils/tta.py`)
6-fold TTA (original + hflip + vflip + 3 rotations). Averages probability maps.
Consistently improves IoU by 1–3% at inference time.

---

## External Libraries

| Library | Purpose |
|---|---|
| PyTorch / torchvision | Training framework + ResNet-34 pretrained weights |
| albumentations | Augmentation (geometric + photometric + noise) |
| OpenCV | Distortion simulation (JPEG compression, blur) |
| matplotlib | Visualisation and result figures |

All code in this repository is original group work. Pretrained ResNet-34 weights are from torchvision (ImageNet).
