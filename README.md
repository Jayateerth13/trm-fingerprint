# fingerprint-verification-trm

Fingerprint verification baselines and Tiny Recursive Model (TRM) for parameter-efficient biometric matching on FVC2004.

---

## Overview

This project investigates whether a compact recursive reasoning module (TRM) can match the verification performance of a full deep baseline (Siamese ResNet-18) while using significantly fewer task-specific trainable parameters.

The evaluation task is fingerprint verification: given two fingerprint images, determine if they belong to the same finger (genuine) or different fingers (impostor). Performance is measured using EER (Equal Error Rate), FAR, and FRR.

**Dataset:** FVC2004 — 3,520 images, 440 finger identities, 4 sensors, 8 impressions per finger.

---

## Project Structure

```
.
├── notebooks/
│   ├── 01_data_engineering.ipynb   # Dataset manifest, splits, pair generation
│   ├── 02_eda.ipynb                # Exploratory data analysis and plots
│   └── 03_baseline.ipynb           # NCC and Siamese ResNet-18 baselines
│
├── artifacts/
│   ├── manifest.csv                # Full image manifest
│   ├── manifest_with_split.csv     # Manifest with train/val/test labels
│   ├── pairs_train/val/test.csv    # Genuine and impostor pairs per split
│   ├── dataset_summary.json        # Dataset statistics
│   ├── baseline_results.json       # NCC and ResNet-18 evaluation results
│   └── figures/                    # All EDA and baseline plots
│
├── dataset/FVC2004/                        # Raw dataset (not tracked in git)
├── requirements.txt
└── README.md
```

---

## Setup

**1. Create and activate a virtual environment**

```bash
python3 -m venv .venv
source .venv/bin/activate
```

**2. Install dependencies**

```bash
pip install -r requirements.txt
```

**3. Place the FVC2004 dataset**

The dataset is not included in this repository. Place it at:

```
dataset/FVC2004/DB1_A/
dataset/FVC2004/DB1_B/
dataset/FVC2004/DB2_A/
...
dataset/FVC2004/DB4_B/
```

Each folder should contain `.tif` files named `{finger}_{impression}.tif` (e.g. `1_1.tif`).

> **Need the dataset?** If you have trouble obtaining FVC2004, feel free to reach out at **jayateerth.kamatgi@sjsu.edu**.

---

## Running the Notebooks

Run the notebooks in order using Jupyter:

```bash
jupyter notebook
```

| Step | Notebook | What it does |
|------|----------|--------------|
| 1 | `01_data_engineering.ipynb` | Builds manifest, splits by finger identity, generates pairs |
| 2 | `02_eda.ipynb` | EDA — intensity distributions, NCC analysis, class balance plots |
| 3 | `03_baseline.ipynb` | NCC baseline (EER 30.8%) and Siamese ResNet-18 (EER 8.7%) |

All outputs (CSVs, plots, model weights, result JSON) are saved to `artifacts/`.

> **GPU:** The baseline notebook runs on CPU by default. To use a GPU, set `FORCE_CPU = False` in Cell 1 of `03_baseline.ipynb`. For free GPU access, use [Google Colab](https://colab.research.google.com) or [Kaggle Notebooks](https://www.kaggle.com/code).

---

## Baseline Results (FVC2004 Test Set)

| Method | EER | Accuracy@EER | FAR | FRR |
|--------|-----|--------------|-----|-----|
| NCC (no learning) | 0.308 | 0.692 | 0.307 | 0.309 |
| Siamese ResNet-18 | **0.087** | **0.913** | 0.086 | 0.088 |

Siamese ResNet-18: 11.2M parameters, trained for 6 epochs on 8,000 pairs (CPU).

---

## Work in Progress — Tiny Recursive Model (TRM)

The next phase implements **TRM**: a small, parameter-efficient decision module that operates on top of a frozen, general-purpose visual encoder (ImageNet-pretrained ResNet-18, weights not updated).

**Key idea:** Instead of fine-tuning 11M parameters end-to-end on fingerprint data, TRM learns a compact recursive comparison function (~50K parameters) that iteratively refines a match decision from generic visual embeddings.

**Goals:**
- Match Siamese ResNet-18 verification performance (EER ≤ 10%)
- Use significantly fewer fingerprint-specific trainable parameters
- Faster training time due to frozen encoder (no backprop through ResNet-18)

TRM implementation will be added to `notebooks/04_trm.ipynb`.
