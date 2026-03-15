# Advanced Methods for Enhancing Interpretability and Performance in Explainable Boosting Machines

**Master's Thesis** — Ata Berk Çakır  
Department of Statistics, Ludwig-Maximilians-Universität München  
Supervised by Dr. Giuseppe Casalicchio  
Submitted: March 15, 2026

---

## Overview

This repository contains all code to reproduce the experiments and results reported in the thesis. The implementation extends the Explainable Boosting Machine (EBM) framework with:

- **FAST-3D screening**: residual-weighted histogram statistic for triple interaction candidate ranking
- **Scaled-BIC backward pruning**: single-parameter sparsity control over the full term set
- **Cyclic joint refit**: jointly refits all terms (mains + interactions) from scratch to resolve signal entanglement

Experiments cover synthetic benchmarks with planted 3-way signals (H1–H3) and seven real-world OpenML regression datasets (H4).

---

## Repository Structure

```
.
├── README.md                          # This file
├── requirements.txt                   # Python package versions
├── Thesis_Final_Code_EarlyStop.ipynb  # Main experiment notebook (all blocks)
└── outputs/                           # Auto-generated on run (not tracked by git)
    ├── tables/                        # CSV tables
    ├── figs/                          # PDF + PNG figures
    ├── splits/                        # Train/test split indices (.npy)
    ├── hparam/                        # Hyperparameter cache (PerDataset_BestParams.json)
    └── logs/                          # Run configuration log
```

---

## Notebook Block Map — What Produces What

Run blocks **in order** (top to bottom). Each block is self-contained after the preceding ones have run.

| Block | Title | Thesis Output |
|-------|-------|---------------|
| BLOCK 0 | Global Config | — (set `DEBUG_MODE=False` for full run) |
| BLOCK 1 | Installs & Imports | — |
| BLOCK 2 | Core Algorithm | — (`SparseScratchEBM`, `ScratchEBMWithBagging`) |
| BLOCK 3 | Model Wrappers | — |
| BLOCK 4 | Dataset Suite | Table 1, Table 2 |
| BLOCK 5 | Splits | Deterministic train/test splits (saved to `outputs/splits/`) |
| BLOCK 6 | Hyperparameter Tuning | `PerDataset_BestParams.json` (cached; skip if cache exists) |
| BLOCK 7 | Build Models | — |
| BLOCK 8 | Benchmark Engine | — (`run_benchmark()`, `summarize()`) |
| BLOCK 9 | Standard Synthetic Benchmark | **Table 5** (H1 competitiveness leg) |
| BLOCK 10 | Modified Synthetic Benchmark | **Table 6** (H1 gain leg + H2 found_rate) |
| BLOCK 11 | BIC Sweep (H3) | **Table 10** (Appendix A.1), **Figure 3** |
| BLOCK 12 | FAST-3D Recovery (H2) | **Table 7** + **Figure 2** (Friedman1_Mod score distribution) |
| BLOCK 13 | Real Dataset Benchmark | **Table 8** (H4) |
| BLOCK 13b | Term Overlap Analysis | **Table 11** (Appendix A.2) |
| BLOCK 14 | Ablation: Cyclic Refit | **Table 9** |
| BLOCK 15 | max_bins Sensitivity | **Table 12** (Appendix A.3) |

---

## How to Reproduce

### 1. Install dependencies

```bash
pip install -r requirements.txt
```

### 2. Open the notebook

```bash
jupyter notebook Thesis_Final_Code.ipynb
```

### 3. Run all blocks in order

- Set `DEBUG_MODE = False` in BLOCK 0 for the full experimental run.
- Set `DEBUG_MODE = True` for a fast smoke test (`n=300`, 1 seed, ~5 minutes).
- BLOCK 5 must be run **once** to generate split files before any benchmark block.
- BLOCK 6 (hyperparameter tuning) is cached: if `outputs/hparam/PerDataset_BestParams.json` exists, tuning is skipped and cached parameters are loaded. Delete this file to re-run tuning from scratch.
- Real datasets (BLOCK 13) are fetched automatically from OpenML on first access and cached in memory.

### 4. Expected runtime (full run, `DEBUG_MODE=False`)

| Block | Approx. runtime |
|-------|----------------|
| BLOCK 6 (tuning) | ~300 min (first run only; cached thereafter) |
| BLOCK 9 (standard synthetic) | ~5 min |
| BLOCK 10 (modified synthetic) | ~20 min |
| BLOCK 11 (BIC sweep) | ~36 min |
| BLOCK 12 (Recall@K) | ~10 min |
| BLOCK 13 (real datasets) | ~150 min |
| BLOCK 14 (ablation) | ~140 min |
| **Total (excl. tuning)** | **~4–5 hours** |

Runtimes measured on a MacBook Air M1 (8 GB RAM), `N_JOBS=4`.  
On a standard laptop with fewer cores, expect 2–3× longer.

---

## Data

### Synthetic Datasets

All synthetic datasets are generated programmatically — no external data files required. The benchmark includes three standard and three modified variants (with planted 3-way signal), plus noise-augmented versions for H3 testing:

| Dataset | DGP Type | Embedded Triplet |
|---------|----------|-----------------|
| Friedman1 | Mixed (pairwise + main) | — |
| Ishigami | Trigonometric | — |
| Hartmann6D | Exponential mixture | — |
| Friedman1_Mod | Friedman1 + 25·sin(πx₀x₁x₂) | (0, 1, 2) |
| Ishigami_Mod | Ishigami + 15·sin(x₀)sin(x₁)sin(x₂) | (0, 1, 2) |
| Hartmann6D_Mod | Hartmann6D + 20·(x₀x₃x₅) | (0, 3, 5) |

`*_ModNoise10` variants add K=10 uniform noise features to each modified dataset (used for H3 BIC sweep).

### Real Datasets

Real datasets are fetched automatically from [OpenML](https://www.openml.org) via `sklearn.datasets.fetch_openml`. Internet access is required for the first run. Only numeric features are used (`select_dtypes(include=[np.number])`).

| Dataset | OpenML ID | n | p | Target | Source |
|---------|-----------|---|---|--------|--------|
| abalone | 183 | 4,177 | 7† | rings (age) | OpenML-CTR23 |
| cpu_act | 197 | 8,192 | 21 | usr (CPU usage) | OpenML-CTR23 |
| diamonds | 42225 | 53,940 | 6† | price (USD) | OpenML-CTR23 |
| elevators | 216 | 16,599 | 18 | Goal (elevator action) | OpenML |
| naval_propulsion_plant | 44969 | 11,934 | 14 | GT compressor decay coeff. | OpenML-CTR23 |
| superconduct | 43174 | 21,263 | 81 | critical_temp | OpenML-CTR23 |
| red_wine | 40691 | 1,599 | 11 | quality (wine score) | OpenML-CTR23 |

† Categorical variables excluded: abalone (Sex; p: 8→7), diamonds (cut, color, clarity; p: 10→6).

Reference: Fischer, S. F., Harutyunyan, L., Feurer, M., and Bischl, B. (2023). OpenML-CTR23 — A curated tabular regression benchmarking suite. *AutoML Conference 2023 (Workshop)*. https://openreview.net/forum?id=HebAOoMm94

---

## Models

Four models are compared throughout the benchmark:

| Model | Description |
|-------|-------------|
| 3-way EBM | Proposed model with FAST-3D + cyclic joint refit, BIC disabled (αBIC=0) |
| 3-way EBM + BIC | Proposed model with Scaled-BIC backward pruning enabled |
| Vanilla EBM | Standard `ExplainableBoostingRegressor` from interpretML (main effects + pairwise only) |
| XGBoost | Regularized gradient boosting baseline (Chen & Guestrin, 2016) |
| Random Forest | Bagging ensemble baseline (Breiman, 2001) |

All EBM variants share the same core hyperparameters (fairness contract): `n_estimators`, `learning_rate`, `max_bins=32`, `interactions`, `outer_bags=5`, `early_stopping_rounds=200`.

---

## Software Environment

```
Python 3.10+
scikit-learn >= 1.3
numpy >= 1.24
pandas >= 2.0
matplotlib >= 3.7
xgboost >= 2.0
interpret >= 0.4   # for VanillaEBM (ExplainableBoostingRegressor)
joblib >= 1.3
```

Full pinned versions: see `requirements.txt`.

The notebook was developed and tested on:
- macOS 14 (Apple M1)
- Python 3.10.12

To check your environment after running BLOCK 1:
```python
import sklearn, numpy, pandas, xgboost
print(sklearn.__version__, numpy.__version__, pandas.__version__, xgboost.__version__)
```

---

## Reproducibility Notes

- All random seeds are fixed via `np.random.RandomState(seed)` with seeds `[0, 1, 2]`.
- Train/test splits (80/20) are pre-generated and saved as `.npy` files in `outputs/splits/`. Noise-augmented datasets reuse the split indices of their base `_Mod` variant.
- Noise features in `*_ModNoise10` variants are generated with a separate RNG (`seed + 99999`) to ensure independence from the signal features.
- Hyperparameter tuning results are cached in `outputs/hparam/PerDataset_BestParams.json`. The same cache file stores the per-dataset optimal BIC scale (`BIC_SCALE` key).

---

## Citation

If you use this code, please cite:

```
Çakır, A. B. (2026). Advanced Methods for Enhancing Interpretability and Performance
in Explainable Boosting Machines. Master's Thesis, Department of Statistics,
Ludwig-Maximilians-Universität München.
```
