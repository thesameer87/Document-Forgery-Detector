# Document Forgery Detector

> Image forgery detection using **Error Level Analysis (ELA)** and **EfficientNet** with **LoRA** fine-tuning on the CASIA v2 dataset.

---

## Overview

This project detects tampered (spliced / copy-moved) images by:

1. Computing **Error Level Analysis (ELA)** maps that highlight compression inconsistencies.
2. Classifying the ELA maps with an **EfficientNet-B0** backbone fine-tuned via **LoRA** (Low-Rank Adaptation).
3. Evaluating robustness against common post-processing (JPEG compression, noise, blur).

## Repository Structure

```text
doc-forgery-detector/
│
├── configs/
│   └── config.yaml            # Single source of truth for all hyperparameters
├── data/
│   ├── raw/                   # Original CASIA v2 images (not committed)
│   ├── ela/                   # Generated ELA maps (not committed)
│   └── corrupted/             # Robustness-test images (not committed)
├── notebooks/
│   └── 01_eda.ipynb           # Exploratory data analysis
├── src/
│   ├── ela.py                 # ELA computation
│   ├── dataset.py             # PyTorch Dataset & DataLoader utilities
│   ├── model.py               # Model architecture (EfficientNet + LoRA)
│   ├── train.py               # Training loop
│   ├── evaluate.py            # Evaluation & metrics
│   ├── robustness.py          # Robustness testing pipeline
│   ├── failure_analysis.py    # Failure-case analysis & Grad-CAM
│   └── utils.py               # Shared helpers (config loading, logging, seeding)
├── tests/                     # Unit & integration tests
├── results/                   # Evaluation outputs (not committed)
├── checkpoints/               # Model weights (not committed)
├── README.md
├── requirements.txt
├── .gitignore
└── LICENSE
```

## Setup

### 1. Clone the repository

```bash
git clone https://github.com/<your-username>/doc-forgery-detector.git
cd doc-forgery-detector
```

### 2. Create a virtual environment

```bash
python -m venv venv
source venv/bin/activate        # Linux / macOS
venv\Scripts\activate           # Windows
```

### 3. Install dependencies

```bash
pip install -r requirements.txt
```

### 4. Prepare the dataset

Download [CASIA v2](https://github.com/namtpham/casia2groundtruth) and place the images under:

```text
data/raw/
├── Au/    # Authentic images
└── Tp/    # Tampered images
```

## Configuration

All tuneable parameters live in [`configs/config.yaml`](configs/config.yaml).  
Edit that file — **do not hardcode values** in source modules.

## 7-Day Development Plan

| Day | Milestone                              | Status |
|-----|----------------------------------------|--------|
| 1   | Repository setup, environment, EDA     | ✅      |
| 2   | ELA preprocessing pipeline             | ⬜      |
| 3   | Dataset class & data loaders           | ⬜      |
| 4   | Model architecture & LoRA integration  | ⬜      |
| 5   | Training loop & validation             | ⬜      |
| 6   | Evaluation, robustness, failure analysis | ⬜    |
| 7   | Documentation, testing, final polish   | ⬜      |

## Results

### Metrics

| Model | Accuracy | Precision | Recall | F1 Score | AUC-ROC |
|-------|----------|-----------|--------|----------|---------|
| ResNet-18 | — | — | — | — | — |
| EfficientNet-B0 | — | — | — | — | — |
| EfficientNet-B0 + LoRA | — | — | — | — | — |

### Robustness

| Perturbation | Level | Accuracy |
|--------------|-------|----------|
| JPEG Compression | Q70 / Q50 / Q30 | — |
| Gaussian Noise | σ 0.01 / 0.05 / 0.1 | — |
| Gaussian Blur | k3 / k5 / k7 | — |

### Failure Analysis

*(To be populated after Day 6 — Grad-CAM visualizations, misclassification patterns, and edge-case discussion.)*

## License

This project is licensed under the [MIT License](LICENSE).
