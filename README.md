# fingerprint-verification-trm

Fingerprint verification baselines and Tiny Recursive Model (TRM) for parameter-efficient biometric matching on FVC2004.

---

## Overview

This project evaluates fingerprint verification baselines and a compact recursive Transformer model (SiamViTRM / TRM-style) for parameter-efficient biometric matching, compared against a Siamese ResNet-18 baseline.

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
│   ├── 04_trm_model.ipynb          # SiamViTRM (TRM/ViTRM-style) training + evaluation (local)
│   └── SiamViTRM_Colab_Final.ipynb # Colab notebook used to run training (GPU)
│
├── artifacts/
│   ├── manifest.csv                # Full image manifest
│   ├── manifest_with_split.csv     # Manifest with train/val/test labels
│   ├── pairs_train/val/test.csv    # Genuine and impostor pairs per split
│   ├── dataset_summary.json        # Dataset statistics
│   ├── baseline_results.json       # NCC and ResNet-18 evaluation results
│   └── figures/                    # All EDA and baseline plots
│   └── trm/                        # SiamViTRM artifacts (weights, figures, results)
│       ├── trm_results.json
│       ├── siamvitrm_main.pt
│       ├── siamvitrm_ema.pt
│       ├── siamvitrm_ema_best.pt
│       └── figures/
│           ├── siamvitrm_architecture.png
│           ├── siamvitrm_training_curve.png
│           ├── far_frr_siamvitrm.png
│           ├── siamvitrm_roc_curve.png
│           └── siamvitrm_score_distribution.png
│
├── dataset/FVC2004/                        # Raw dataset (not tracked in git)
├── scripts/
│   └── trm_model.py                        # SiamViTRM model + EMA implementation
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
| 4 | `04_trm_model.ipynb` | SiamViTRM (TRM/ViTRM-style) training + evaluation; saves outputs to `artifacts/trm/` |

All outputs (CSVs, plots, model weights, result JSON) are saved to `artifacts/`.

> **GPU:** The baseline notebook runs on CPU by default. To use a GPU, set `FORCE_CPU = False` in Cell 1 of `03_baseline.ipynb`. For free GPU access, use [Google Colab](https://colab.research.google.com) or [Kaggle Notebooks](https://www.kaggle.com/code).

> **Colab training:** The notebook `notebooks/SiamViTRM_Colab_Final.ipynb` was used to run the SiamViTRM training on GPU and export the resulting artifacts under `artifacts/trm/`.

---

## Baseline Results (FVC2004 Test Set)

| Method | Test pairs | Params | EER | Accuracy@EER | FAR | FRR |
|--------|-----------:|------:|----:|-------------:|----:|----:|
| NCC (no learning) | 2,500 | 0 | 0.308 | 0.692 | 0.307 | 0.309 |
| Siamese ResNet-18 | 2,500 | 11.24M | **0.087** | **0.913** | 0.086 | 0.088 |
| SiamViTRM (TRM/ViTRM-style) | 3,696 | 1.837M | 0.159 | 0.841 | 0.160 | 0.158 |

Siamese ResNet-18: 11.2M parameters, trained for 6 epochs on 8,000 pairs (CPU).

---

## SiamViTRM — Tiny Recursive Model (TRM/ViTRM-style)

This repository includes **SiamViTRM**, a parameter-efficient Siamese verifier that adapts:
- **ViTRM**: a *weight-tied* recursive Transformer encoder with a prediction token \(y\) and latent memory tokens \(z\).
- **TRM-style training**: EMA evaluation (and optional deep supervision).

**Architecture (this run):**
- **Input**: grayscale fingerprints resized to 128×128
- **Patch embedding**: 16×16 patches → 64 tokens/image
- **Transformer dimension**: \(d_{model}=192\), \(n_{heads}=6\)
- **Shared block depth**: \(n_{blocks}=4\) (weight-tied across recursion)
- **Latent memory**: \(K=24\) tokens
- **Latent steps**: 3 z-updates + 1 y-update per recursion cycle
- **Recursion**: \(T_{recursion}=1\) (full-gradient)
- **Trainable params**: 1,837,440

Files:
- **Local notebook**: `notebooks/04_trm_model.ipynb`
- **Model implementation**: `scripts/trm_model.py`
- **Colab training notebook**: `notebooks/SiamViTRM_Colab_Final.ipynb`
- **Saved outputs**: `artifacts/trm/` (weights, plots, `trm_results.json`)

