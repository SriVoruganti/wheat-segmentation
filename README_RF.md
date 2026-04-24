# 🌾 Wheat Crop Segmentation — Machine Learning. 

> UNSW COMP9517 Group Project 2026 T1 — Machine Learning Component  
> **EWS (Eschikon Wheat Segmentation) Dataset**

---

## 📋 Overview

This file describes the implementation of the machine learning focused binary segmentation of wheat crops from field images. Given an RGB image, the model produces a binary mask classifying every pixel as either **wheat** or **soil**.

A Random Forest classifier is utilised.


---

## 📁 Project Structure (related to Machine Learning component)

```
wheat-segmentation/
├── data/
│   ├── dataset.py                  # EWSDataset loader with subset & label-noise support
├── models/
│   ├── random_forest.py            # Random Forest Model that learns on different feature sets.
├── experiments/
│   ├── rf_robustness.py            # Evaluate under image distortions
│   └── data_scarcity_RF.py         # Train with 25/50/75/100% data
│   └── rf_feature_abalation.py     # Examines different hand-crafted colour features
├── results/
│   ├── rf_full/                    # evaluation results 
│   ├── robustness/                 # Robustness experiment results
│   ├── scarcity/                   # Data scarcity experiment results
├── scripts/
│   ├── eval_RF/                    # Test set evaluation 
│   ├── train_RF/                   # Main training script for the RF classifier
│   ├── rf_save_panels.py/          # Save best/median/worst qualitative panels
```

---

## 🏆 Results Summary

### Test Set Performance

| Model | Precision | Recall | F1-Score | IoU | Inference (ms/img) |
|---|---|---|---|---|---|
| Random Forest | 0.818 | 0.817 | 0.817 | 0.691 | 1,750 |


---v

## ⚙️ Setup & Usage

To run the machine learning component of this analysis, the folling steps are required. 

### Dependencies

Ensure the below Python packages are installed and functional. 

```
os
path
argparse
json
random
time
numpy
cv2
matplotlib
sklearn
```

### Dataset Structure

The EWS dataset should be stored in a data directory. The train, test and validation data is separated in subdirectories, each of which contain images and masks in separate locations. 

```
data/EWS-Dataset/
├── train/images/   train/masks/
├── val/images/     val/masks/
└── test/images/    test/masks/
```


### Training

The random forest classifier is trained on the dataset and saved to a *model_out* location. File is run based on desired handcrafted colour feature set. 

```bash
    #Train RGB only:
    python -m scripts.train_RF --feature_mode rgb --out_dir .\results\rf_full

    #Train RGB+ExG:
    python -m scripts.train_RF --feature_mode rgb_exg --out_dir .\results\rf_full

    #Train RGB+HSV:
    python -m scripts.train_RF --feature_mode rgb_hsv --out_dir .\results\rf_full

    #Train RGB+HSV+ExG:
    python -m scripts.train_RF --feature_mode rgb_hsv_exg --out_dir .\results\rf_full
```


### Evaluation

To run the evaluation file: 

```bash
#Evaluate your final model (RGB+HSV+ExG)
python -m scripts.eval_RF --data_root .\data\EWS-Dataset --split test --model_path .\results\rf_model_rgb_hsv_exg.pkl --feature_mode rgb_hsv_exg

#With flip-augmentation voting + visual outputs
python -m scripts.eval_RF --data_root .\data\EWS-Dataset --split test --model_path .\results\rf_model_rgb_hsv_exg.pkl --feature_mode rgb_hsv_exg --flip_aug --visualise --failure_analysis
```

Evaluation outputs are stored in the below structure

```
Outputs:
    Metrics JSON:  results/rf_full/test_metrics_rf_<tag>.json 
    Example figures:  results/rf_full/figures/ 
    Failure analysis:  results/rf_full/figures/failures/ 
    Where  <tag>  is  no_aug  or  flip_aug .

Evaluate other feature modes
    --model_path  (model trained with that feature mode)   
    --feature_mode  (must match)
```

Example usage (RGB-only model):

```
python -m scripts.eval_RF --data_root .\data\EWS-Dataset --split test --model_path .\results\rf_model_rgb_hsv_exg.pkl --feature_mode rgb_hsv_exg

```

Save best/median/worst qualitattive panels:
```bash
python -m scripts.rf_save_panels --data_root .\data\EWS-Dataset --split test --model_path .\results\rf_model_rgb_hsv_exg.pkl --feature_mode rgb_hsv_exg --out_dir .\results\rf_full\panels
```

### Experiments

#### Feature Ablation

Feature ablation script (experiments/rf_feature_ablation.py)

``` bash
# Default (train on train, evaluate on val)
python .\experiments\rf_feature_ablation.p

# Evaluate on test (only when you’re done tuning)
python .\experiments\rf_feature_ablation.py --split test

# Faster run (fewer trees + fewer pixels per image)
python .\experiments\rf_feature_ablation.py --split val --n_estimators 100 --max_pixels_per_image 2000

```

#### Robustness

```bash
# Default (val split, uses rgb_hsv_exg model path default)
python .\experiments\rf_robustness.py

#Explicit model + test split
python .\experiments\rf_robustness.py --split test --feature_mode rgb_hsv_exg --model_path .\results\rf_full\rf_model_rgb_hsv_exg.pkl

#Change randomness of noise/occlusion (different distortion seed)
python .\experiments\rf_robustness.py --split val --seed 1
```

#### Data Scarcity

```bash
# Default run (RGB, fraction splits: 25/50/75/100)
python .\experiments\data_scarcity_RF.py

# Run a single feature mode (recommended for report figures)
python .\experiments\data_scarcity_RF.py --feature_mode rgb_hsv_exg

# Run all feature modes (produces per-mode plots + a comparison plot)
python .\experiments\data_scarcity_RF.py --feature_mode all
```


