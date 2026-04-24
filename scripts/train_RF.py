"""
Training script for Random Forest wheat pixel classifier.
"""

import os
import sys

sys.path.append(os.path.dirname(__file__))

from data.dataset import EWSDatasetRF
from models.random_forest import build_model, save_model

from pathlib import Path
REPO_ROOT = Path(__file__).resolve().parents[1]
DATA_ROOT = REPO_ROOT / "data" / "EWS-Dataset"

FEATURE_MODE = "rgb_hsv_exg"  # 7 features: RGB + HSV + ExG
RF_FULL_DIR = REPO_ROOT / "results" / "rf_full"
RF_FULL_DIR.mkdir(parents=True, exist_ok=True)

MODEL_OUT = RF_FULL_DIR / f"rf_model_{FEATURE_MODE}.pkl"

# Load data
print("Loading training data...")
train_ds = EWSDatasetRF(
    root=DATA_ROOT,
    split="train",
    max_pixels_per_image=5000,
    feature_mode=FEATURE_MODE,
)
X_train, y_train = train_ds.load()

print("Loading validation data...")
val_ds = EWSDatasetRF(
    root=DATA_ROOT,
    split="val",
    max_pixels_per_image=5000,
    feature_mode=FEATURE_MODE,
)
X_val, y_val = val_ds.load()

print(f"  X_train : {X_train.shape}")
print(f"  X_val   : {X_val.shape}")
print(f"  Class balance (train): {y_train.mean():.2%} wheat pixels")

# Train
print("\nTraining Random Forest...")
clf = build_model()
clf.fit(X_train, y_train)

# Sanity check: should be 7
if hasattr(clf, "n_features_in_"):
    print("Model expects n_features_in_ =", clf.n_features_in_)

# Save
save_model(clf, MODEL_OUT)
print("Saved model ->", MODEL_OUT)
