# Advanced Segmentation (classical methods)

This part of the project implements and evaluates three **classical**
(non-deep-learning) wheat segmentation methods:

1. **Watershed** — HSV thresholding + distance-transform markers + OpenCV Watershed.
2. **Superpixel + ExG + Otsu** — SLIC superpixels, classified by their mean
   Excess-Green (ExG) index using Otsu's automatic threshold.
3. **Superpixel (noise-robust)** — same as (2), preceded by a bilateral
   filter and a median filter for edge-preserving denoising.

## File overview

| File                                         | Purpose                                                      |
| -------------------------------------------- | ------------------------------------------------------------ |
| `models/advanced_segmentation.py`            | The 3 segmentation methods                                   |
| `data/classical_loader.py`                   | Image + ground-truth mask loader (original flat EWS layout)  |
| `utils/classical_distortions.py`             | Blur / Noise / Darkening / Occlusion distortions             |
| `utils/classical_metrics.py`                 | `calculate_iou`                                              |
| `experiments/evaluate_advanced.py`           | Compare Watershed vs. Superpixel (single image or batch)     |
| `experiments/distortion_eval_advanced.py`    | IoU of Superpixel under 4 distortions                        |
| `experiments/noise_robustness_eval.py`       | Noise-robust variant vs. baseline Superpixel                 |

## Dataset setup

The classical methods load images directly from the **original** EWS folder
layout (image and mask side by side in the same split folder). No
restructuring is required.

1. Download `EWS-Dataset.zip` from
   <https://www.research-collection.ethz.ch/entities/researchdata/165d22fc-6b0f-4fc3-a441-20d8bdc50a70>
2. Extract it so the folder structure is:

   ```
   wheat-segmentation/
       data/
           EWS-Dataset/
               train/
                   FPWW0220032_RGB1_20180411_113950_6.png
                   FPWW0220032_RGB1_20180411_113950_6_mask.png
                   ...
               val/
                   ...
               test/
                   ...
   ```

   (This is exactly the layout you get from the ZIP — no renaming needed.)

If you cannot place the dataset at the default location, override it via
either of:

- the `--data_root` CLI flag on any of the evaluation scripts, **or**
- the `EWS_DATA_ROOT` environment variable (e.g.
  `export EWS_DATA_ROOT=/path/to/EWS-Dataset`).

## Dependencies

Only standard scientific-Python packages are required:

```
numpy
opencv-python
scikit-image
matplotlib
```

Install with:

```bash
pip install numpy opencv-python scikit-image matplotlib
```

(These are subsets of the project's top-level `requirements.txt`.)

## How to run

All scripts are run from the **project root** (the directory that contains
`data/`, `models/`, `utils/`, `experiments/`) using Python's `-m` flag.

### 1. Compare Watershed vs. Superpixel

Single-image visualisation:

```bash
python -m experiments.evaluate_advanced --mode single --index 1
```

Batch table (default indices 0–12 from the train split):

```bash
python -m experiments.evaluate_advanced --mode batch
```

### 2. Evaluate robustness to distortions

Single-image 2x5 grid (original + 4 distortions, with IoU below each):

```bash
python -m experiments.distortion_eval_advanced --mode single --index 10
```

Batch table with mean IoU per distortion:

```bash
python -m experiments.distortion_eval_advanced --mode batch
```

### 3. Noise-robust variant vs. baseline

```bash
python -m experiments.noise_robustness_eval
```

### Common options

Every script supports:

- `--split {train, val, test}` — which split to evaluate on (default `train`)
- `--data_root PATH` — override the dataset root
- `--indices I1 I2 ...` — which image indices to evaluate
