# Bayesian Prediction-Powered Inference with MCMC

**Bayesian PPI (Chain-Rule Estimator) for Medical Imaging + Reproducible Code**

> Companion repository for the manuscript:
>
> **“Bayesian Prediction-Powered Inference with MCMC: Methods and a Medical Imaging Case Study”**
> PDF: `paper/tmlr_bayesianppi_draft.pdf`

---

## Table of Contents

* [1. Abstract (Paper Summary)](#1-abstract-paper-summary)
* [2. What is Bayesian PPI? (One-Minute Intuition)](#2-what-is-bayesian-ppi-one-minute-intuition)
* [3. Repository Layout](#3-repository-layout)
* [4. Installation](#4-installation)

  * [4.1. Python version](#41-python-version)
  * [4.2. Create a virtual environment](#42-create-a-virtual-environment)
  * [4.3. Install the package (editable)](#43-install-the-package-editable)
  * [4.4. Extra dependencies for CNN training (optional)](#44-extra-dependencies-for-cnn-training-optional)
  * [4.5. Windows + PyMC note (PyTensor)](#45-windows--pymc-note-pytensor)
* [5. Quickstart](#5-quickstart)

  * [5.1. Prepare data files](#51-prepare-data-files)
  * [5.2. Sanity import](#52-sanity-import)
  * [5.3. Run key scripts](#53-run-key-scripts)
* [6. Configuration](#6-configuration)

  * [6.1. Minimal config examples](#61-minimal-config-examples)
  * [6.2. Where configs are consumed](#62-where-configs-are-consumed)
* [7. Methods (Paper-level Summary)](#7-methods-paper-level-summary)

  * [7.1. Chain-Rule Estimand and Generative Model](#71-chain-rule-estimand-and-generative-model)
  * [7.2. Bayesian Computation & Diagnostics](#72-bayesian-computation--diagnostics)
  * [7.3. Thresholds & Calibration](#73-thresholds--calibration)
  * [7.4. Extensions: Binning and Hierarchies](#74-extensions-binning-and-hierarchies)
* [8. Experiments & Reproduction](#8-experiments--reproduction)

  * [8.1. Simulations](#81-simulations)
  * [8.2. Coverage & Width Benchmarks](#82-coverage--width-benchmarks)
  * [8.3. MRI Case Study (ADNI)](#83-mri-case-study-adni)
  * [8.4. Seeds, SBC, PPCs](#84-seeds-sbc-ppcs)
* [9. Data Expectations](#9-data-expectations)
* [10. Troubleshooting](#10-troubleshooting)
* [11. Citing](#11-citing)
* [12. License](#12-license)

---

## 1. Abstract (Paper Summary)

Modern predictors trained on vast unlabeled corpora can be accurate yet systematically biased, complicating valid uncertainty quantification when only a small labeled set is available. **Prediction–Powered Inference (PPI)** debiases an imputed estimator using a *rectifier* computed on labeled data, yielding frequentist guarantees.
We develop a **fully Bayesian** analogue of PPI for non-conjugate models using **MCMC (NUTS/HMC)**. Treating the PPI chain-rule as a generative model over **$$$1$$**, we couple abundant autorater outputs with scarce labels and propagate posterior uncertainty to the functional
$$$1$$
Across synthetic and medical-imaging experiments (Alzheimer’s disease MRI), our **Bayesian chain-rule estimator (CRE)** delivers calibrated intervals with competitive or shorter widths than (i) labeled-only Bayesian and (ii) classical difference estimators, while following best-practice Bayesian workflow (SBC, PPCs, $$$1$$, ESS, divergence checks).

---

## 2. What is Bayesian PPI? (One-Minute Intuition)

* We observe a large pool of **autorater decisions** $$$1$$ and a much smaller set of **human labels** $$$1$$ on a subset.
* The target prevalence $$$1$$ decomposes by the chain rule:
  $$$1$$
* Put **weak Beta priors** on $$$1$$, use Bernoulli likelihoods, run **NUTS** to sample the posterior, and transform draws to $$$1$$.
* Result: **label-efficient**, calibrated credible intervals for $$$1$$, leveraging abundant predictions while correcting with few labels.

---

## 3. Repository Layout

```text
bayesian-ppi/
├─ paper/
│  └─ tmlr_bayesianppi_draft.pdf        # manuscript
├─ src/
│  └─ bayesppi/
│     ├─ __init__.py
│     ├─ data/
│     │  ├─ __init__.py
│     │  └─ process.py
│     ├─ models/
│     │  ├─ __init__.py
│     │  ├─ model_adcn.py               # lightweight 3D-CNN (autorater)
│     │  └─ autorater.py
│     ├─ inference/
│     │  ├─ __init__.py
│     │  ├─ simulation.py
│     │  ├─ simulation_all.py
│     │  ├─ simulation_6570.py
│     │  ├─ coverage_interval_all.py
│     │  ├─ coverage_interval_6570.py
│     │  └─ prior_predictive_checks.py
│     ├─ analysis/
│     │  ├─ __init__.py
│     │  ├─ age_analysis.py
│     │  └─ histogram_posterior.py
│     └─ viz/
│        ├─ __init__.py
│        └─ fig1_pipeline.py
├─ scripts/
│  ├─ train_cnn_all.py
│  ├─ train_cnn_6570.py
│  ├─ run_simulation.py                 # (skeleton; extend as needed)
│  └─ run_*                             # add runners for analyses/figures
├─ configs/
│  ├─ default.yaml
│  ├─ sim_all.yaml
│  ├─ sim_6570.yaml
│  └─ coverage.yaml
├─ data/
│  ├─ raw/.gitkeep
│  ├─ interim/.gitkeep
│  └─ processed/.gitkeep                # place CSVs here (see §9)
├─ notebooks/
│  └─ 00_quickstart.ipynb               # optional
├─ tests/
│  └─ test_imports.py                   # minimal import smoke test
├─ pyproject.toml
├─ .gitignore
└─ README.md
```

---

## 4. Installation

### 4.1. Python version

* **Python ≥ 3.11** is recommended (tested).
* PyMC v5+ requires modern Python; older versions (e.g., 3.7) will fail to resolve dependencies.

### 4.2. Create a virtual environment

**Windows (PowerShell or Git Bash):**

```bash
# in repo root
py -3.11 -m venv .venv

# Git Bash
source .venv/Scripts/activate

# PowerShell
# .venv\Scripts\Activate.ps1

python -V
```

### 4.3. Install the package (editable)

```bash
python -m pip install --upgrade pip setuptools wheel
pip install -e .
```

### 4.4. Extra dependencies for CNN training (optional)

```bash
# CPU-only
pip install torch torchvision --index-url https://download.pytorch.org/whl/cpu

# or CUDA 12.1 (if GPU/driver is set up)
# pip install torch torchvision --index-url https://download.pytorch.org/whl/cu121
```

### 4.5. Windows + PyMC note (PyTensor)

PyTensor warns if `g++` is missing. You can ignore during development or set:

```bash
# suppress C-compile attempts (uses Python fallback; slower but fine for this repo)
export PYTENSOR_FLAGS='cxx='
```

For speed via C-compilation on Windows (conda):

```bash
# conda
# conda install gxx
```

---

## 5. Quickstart

### 5.1. Prepare data files

Place input CSVs in `data/processed/` (see §9 Data Expectations):

```text
data/processed/
├─ autorater_predictions_all4.csv    # autorater decisions & labels (subset)
└─ all_people_7_20_2025.csv          # meta: Subject / Age / Sex / Acq_Date
```

### 5.2. Sanity import

```bash
python - <<'PY'
import importlib
mods = [
    "bayesppi",
    "bayesppi.data.process",
    "bayesppi.models.model_adcn",
    "bayesppi.inference.simulation",
    "bayesppi.analysis.age_analysis",
    "bayesppi.viz.fig1_pipeline",
]
for m in mods:
    importlib.import_module(m)
    print("[OK]", m)
PY
```

### 5.3. Run key scripts

```bash
# (extend these runners as needed; see scripts/ and configs/)
python scripts/run_simulation.py --config configs/sim_all.yaml
python scripts/train_cnn_all.py --help
python scripts/train_cnn_6570.py --help
```

---

## 6. Configuration

### 6.1. Minimal config examples

Create/edit files under `configs/`:

**`configs/default.yaml`**

```yaml
data:
  meta_csv: "data/processed/all_people_7_20_2025.csv"
  pred_csv: "data/processed/autorater_predictions_all4.csv"
  # if you store outputs
  out_dir: "runs/default"

mcmc:
  draws: 2000
  tune: 1000
  chains: 2
  target_accept: 0.90

priors:
  type: "uniform"  # or "jeffreys"

thresholding:
  policy: "fixed"  # "fixed" | "youden" | "cost"
  fixed_t: 0.5
  cost:
    C01: 1.0
    C10: 1.0

strata:
  by_age: false
  bins: [[50, 73], [74, 79], [80, 100]]
```

**`configs/sim_all.yaml`**

```yaml
simulation:
  N_A: 1000
  N_H: 100
  theta_A: 0.6
  theta_H_given_1: 0.8
  theta_H_given_0: 0.3

mcmc:
  draws: 2000
  tune: 1000
  chains: 2
  target_accept: 0.90
```

**`configs/coverage.yaml`**

```yaml
coverage:
  n_rep: 50
  label_budgets: [10, 20, 40, 80]
  priors: ["uniform", "jeffreys"]
```

### 6.2. Where configs are consumed

* `src/bayesppi/inference/simulation.py` — generative experiments & SBC/PPC helpers.
* `src/bayesppi/analysis/age_analysis.py` — age-stratified analysis; expects `meta_csv` & `pred_csv`.
* `scripts/run_simulation.py` — thin CLI wrapper to call into `bayesppi.inference`.

*Tip:* keep paths out of source files; route them via `configs/*.yaml` and small CLI wrappers under `scripts/`.

---

## 7. Methods (Paper-level Summary)

### 7.1. Chain-Rule Estimand and Generative Model

Abundant autorater decisions **A** and scarce human labels **H**.

**Parameters:**

$$$1$$

**Estimand:**

$$$1$$

Likelihood for unlabeled **A** and labeled pairs **(A, H)**; weak Beta priors (Uniform/Jeffreys).

*Identifiability:* $$$1$$ from unlabeled pool; $$$1$$ from labeled subset; prior regularization handles small cells.

### 7.2. Bayesian Computation & Diagnostics

* NUTS/HMC via PyMC; sample logits for stability.
* Defaults: `draws=2000`, `tune=1000`, `chains=2`, `target_accept≈0.90`.
* Diagnostics: rank-normalized $$$1$$; bulk/tail ESS > 400; no divergences; E-BFMI > 0.3.
* PPCs on $$$1$$ margins and induced $$$1$$.
* SBC over synthetic draws to validate calibration.

### 7.3. Thresholds & Calibration

Map probabilities $$$1$$ to decisions $$$1$$ using:

* fixed $$$1$$;
* Youden’s $$$1$$ (maximize TPR+TNR−1);
* cost-sensitive Bayes threshold
  $$$1$$

Reliability (calibration) via temperature scaling or isotonic regression before thresholding; AUC unchanged, decisions improved.

Refit CRE per $$$1$$ (and per stratum) so uncertainty reflects the induced $$$1$$.

### 7.4. Extensions: Binning and Hierarchies

Replace $$$1$$ with $$$1$$ bins of score $$$1$$:
$$$1$$

Hierarchical partial pooling across strata $$$1$$:
$$$1$$
with weak hyperpriors; stabilizes group-wise estimates.

---

## 8. Experiments & Reproduction

### 8.1. Simulations

Match the generative model; typical setting: $$$1$$.

Vary label budgets $$$1$$; compare CRE vs Naïve Bayes (labeled-only) vs Difference Estimator (bootstrap CIs).

### 8.2. Coverage & Width Benchmarks

Report absolute bias, RMSE, average 95% width, empirical coverage over $$$1$$ replications.

CRE typically yields near-nominal coverage with narrower intervals than baselines at fixed label budgets.

### 8.3. MRI Case Study (ADNI)

DICOM→NIfTI (`dcm2niix`), NiBabel I/O, 3D-CNN autorater (lightweight).

Overall and age-stratified operating points (fixed vs Youden), plus calibration.

Refit CRE per stratum; monitor prevalence $$$1$$ with calibrated uncertainty.

### 8.4. Seeds, SBC, PPCs

Seeds are fixed in scripts; SBC and PPCs are available in `inference/` helpers.

Summaries and figures are produced under `figures/` and `runs/` (configurable).

---

## 9. Data Expectations

Place CSVs here: `data/processed/`

**Meta CSV: `all_people_7_20_2025.csv` (example header)**

```csv
Subject,Age,Sex,Acq_Date
S_0001,73,M,2017-05-17
...
```

**Predictions CSV: `autorater_predictions_all4.csv` (example header)**

```csv
subject_id,Acq_Date,autorater_prediction,H,label
S_0001,2017-05-17,0.87,1,AD
...
```

`label` may be categorical; scripts normalize to `{0,1}` internally or during preprocessing.

If your columns differ (e.g., `AcqDate` vs `Acq_Date`), either rename in CSV or adapt the small normalization helper.

**Privacy & licensing:** raw imaging is not included. Use your own ADNI access where applicable and comply with ADNI data use terms.

---

## 10. Troubleshooting

* **PyMC warns: g++ not available (Windows)**
  Set `export PYTENSOR_FLAGS='cxx='` to use Python-mode (slower, but fine). For compiled mode, install a C++ toolchain (e.g., `conda install gxx`).

* **Import succeeds but scripts fail on file not found**
  Ensure `configs/default.yaml` points to existing CSVs, or place files under `data/processed/`.

* **Mismatched headers (AcqDate vs Acq_Date)**
  Rename headers in CSV or enable the normalization snippet; harmonize date parsing to `YYYY-MM-DD`.

* **Stratum too small / extreme imbalance**
  Switch priors to Jeffreys in config to stabilize tail behavior; consider hierarchical pooling.

* **Long runtimes**
  Reduce draws/tune for dev cycles; increase later for final figures.

---

## 11. Citing

If you build on this code or ideas, please cite the paper (update year, openreview when available):

```bibtex
@article{kim202Y_bayesppi,
  title   = {Bayesian Prediction-Powered Inference with MCMC: Methods and a Medical Imaging Case Study},
  author  = {Dowoo Kim},
  journal = {Transactions on Machine Learning Research},
  year    = {YYYY},
  url     = {https://openreview.net/forum?id=XXXX}
}
```

**PPI & related references (selection):**

* Angelopoulos et al., Prediction-Powered Inference (Science/ArXiv).
* Särndal et al., Model Assisted Survey Sampling.
* Vehtari et al., Rank-normalized $$$1$$, ESS; Bayesian workflow.
* Salvatier et al., PyMC; Carpenter et al., Stan; Hoffman & Gelman, NUTS; Betancourt, HMC.

---

## 12. License

This repository is released under the `LICENSE` file at the project root.
Data used in the case study (e.g., ADNI) are subject to their own licenses/terms and are not redistributed here.

---

## Appendix: Minimal Reproduction Commands

```bash
# 0) Create venv (Windows; Git Bash shown)
py -3.11 -m venv .venv
source .venv/Scripts/activate
python -m pip install --upgrade pip setuptools wheel

# 1) Install package
pip install -e .

# 2) Put CSVs
# data/processed/all_people_7_20_2025.csv
# data/processed/autorater_predictions_all4.csv

# 3) Smoke test
python - <<'PY'
import importlib
for m in [
    "bayesppi",
    "bayesppi.inference.simulation",
    "bayesppi.analysis.age_analysis",
]:
    importlib.import_module(m)
    print("[OK]", m)
PY

# 4) Run (fill in the runner bodies as needed)
python scripts/run_simulation.py --config configs/sim_all.yaml
```

---

## Acknowledgments

We thank collaborators and maintainers of PyMC/ArviZ/Stan ecosystems and ADNI for enabling this research workflow.
