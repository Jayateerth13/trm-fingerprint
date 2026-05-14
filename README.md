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

**3. FVC2004**

Not included in the repo. For local notebooks, put the `DB*_A` / `DB*_B` folders under `dataset/FVC2004/` (see `01_data_engineering.ipynb`).

> **Need the dataset?** If you have trouble obtaining FVC2004, email **jayateerth.kamatgi@sjsu.edu**.

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
| 4 | `04_trm_model.ipynb` | SiamViTRM (TRM/ViTRM-style) training + evaluation; saves outputs to `artifacts/trm/` (best Colab run: **15.07%** test EER — see `trm_results.json`) |

All outputs (CSVs, plots, model weights, result JSON) are saved to `artifacts/`.

> **GPU:** The baseline notebook runs on CPU by default. To use a GPU, set `FORCE_CPU = False` in Cell 1 of `03_baseline.ipynb`. For free GPU access, use [Google Colab](https://colab.research.google.com) or [Kaggle Notebooks](https://www.kaggle.com/code).

> **Colab:** `notebooks/SiamViTRM_Colab_Final.ipynb` (GPU training; shipped metrics in `artifacts/trm/trm_results.json` — **15.07%** test EER for the saved run).

---

## Baseline Results (FVC2004 Test Set)

| Method | Test pairs | Params | EER | Accuracy@EER | FAR | FRR |
|--------|-----------:|------:|----:|-------------:|----:|----:|
| NCC (no learning) | 2,500 | 0 | 0.308 | 0.692 | 0.307 | 0.309 |
| Siamese ResNet-18 | 2,500 | 11.24M | **0.087** | **0.913** | 0.086 | 0.088 |
| SiamViTRM (TRM/ViTRM-style) | 3,696 | 1.837M | **0.151** (15.07%) | 0.849 | 0.152 | 0.150 |

SiamViTRM numbers are from `artifacts/trm/trm_results.json`: **test EER 15.07%** (peak LR $10^{-4}$, $N_{\mathrm{sup}}{=}1$, 20 epochs, EMA checkpoint chosen by best validation EER). The EER column is the rate as a fraction (~0.1507, shown rounded). NCC and ResNet-18 use 2,500 random test pairs; SiamViTRM uses all 3,696 test pairs.

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
- **Peak learning rate**: \(10^{-4}\), \(N_{\mathrm{sup}}=1\) (matches `trm_results.json`; **15.07%** test EER for the saved Colab run)

Files:
- **Local notebook**: `notebooks/04_trm_model.ipynb`
- **Model implementation**: `scripts/trm_model.py`
- **Colab training notebook**: `notebooks/SiamViTRM_Colab_Final.ipynb`
- **Saved outputs**: `artifacts/trm/` (weights, plots, `trm_results.json`)

**Code Repository:** [github.com/Jayateerth13/trm-fingerprint](https://github.com/Jayateerth13/trm-fingerprint)

**FVC2004 dataset:** [https://huggingface.co/datasets/tourmii/FVC2004](https://huggingface.co/datasets/tourmii/FVC2004) (Hugging Face).
