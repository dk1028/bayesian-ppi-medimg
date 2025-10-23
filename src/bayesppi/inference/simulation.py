#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
simulation.py

Prior-sensitivity simulation for prevalence estimation (g) using:
  - Bayesian chain-rule estimator (CRE; "prediction-powered" via chain rule)
  - Naïve labeled-only Beta-Bernoulli estimator
  - Difference estimator with bootstrap CIs

TMLR/GitHub–friendly changes:
- No hard-coded paths; all I/O via CLI
- Reproducible RNG (single seed for NumPy + per-replicate PyMC seeds)
- Optional progress bar without hard dependency (uses tqdm if available)
- Logging instead of print; results saved to CSV
- Parameterized MCMC controls and bootstrap size

The objective, data flow, and default values mirror the original snippet.

Example
-------
python simulation.py \
  --csv data/processed/autorater_predictions_all4.csv \
  --out runs/sims/ \
  --label-col label \
  --pred-col autorater_prediction \
  --h-threshold 0.5 \
  --nsim 50 \
  --label-sizes 10 20 40 80 \
  --priors uniform:1:1 jeffreys:0.5:0.5 \
  --draws 500 --tune 500 --chains 2 --cores 1 --target-accept 0.9 \
  --bootstrap 1000 \
  --seed 2025 \
  --progress

Data requirements
-----------------
CSV must contain prediction probabilities and either:
  (a) a numeric binary column H in {0,1}, or
  (b) a categorical column (default "label") where AD=1 and others=0.

The script constructs:
  H       = 1 if AD else 0
  A_class = 1{ prediction >= h_threshold }

Outputs
-------
- CSV results at: <out>/simulation_results_prior_sensitivity.csv
- Log summary to stdout
"""
from __future__ import annotations

import argparse
import logging
from pathlib import Path
from typing import Dict, Iterable, List, Tuple

import numpy as np
import pandas as pd
import pymc as pm


# ---------------------------------------------------------------------
# Logging
# ---------------------------------------------------------------------
def setup_logging(level: str = "INFO") -> None:
    logging.basicConfig(level=getattr(logging, level.upper(), logging.INFO),
                        format="[%(levelname)s] %(message)s")


# ---------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------
def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Prior-sensitivity simulation for CRE / Naïve / Difference estimators.")
    p.add_argument("--csv", type=Path, required=True, help="Input CSV with predictions and labels.")
    p.add_argument("--out", type=Path, required=True, help="Output directory for results CSV.")
    p.add_argument("--pred-col", type=str, default="autorater_prediction", help="Column with prediction probabilities in [0,1].")
    p.add_argument("--h-col", type=str, default=None, help="Optional numeric binary column for H (0/1). If not provided, derived from --label-col.")
    p.add_argument("--label-col", type=str, default="label", help="Categorical label column (AD=1, else 0) used if --h-col not provided.")
    p.add_argument("--h-threshold", type=float, default=0.5, help="Threshold to binarize predictions into A_class (default: 0.5).")

    # Simulation controls
    p.add_argument("--nsim", type=int, default=50, help="Number of simulation replicates.")
    p.add_argument("--label-sizes", type=int, nargs="+", default=[10, 20, 40, 80], help="Label budgets to evaluate.")
    p.add_argument(
        "--priors", type=str, nargs="+", default=["uniform:1:1", "jeffreys:0.5:0.5"],
        help="List like name:alpha:beta (e.g., uniform:1:1 jeffreys:0.5:0.5)."
    )

    # MCMC controls (PyMC)
    p.add_argument("--draws", type=int, default=500, help="Posterior draws per chain.")
    p.add_argument("--tune", type=int, default=500, help="Tuning steps per chain.")
    p.add_argument("--chains", type=int, default=2, help="Number of chains.")
    p.add_argument("--cores", type=int, default=1, help="Cores for sampling (1 is safest cross-platform).")
    p.add_argument("--target-accept", type=float, default=0.9, help="NUTS target_accept.")
    p.add_argument("--progressbar", action="store_true", help="Show PyMC progress bar during sampling.")

    # Bootstrap controls
    p.add_argument("--bootstrap", type=int, default=1000, help="Bootstrap repeats for the difference estimator.")

    # Misc
    p.add_argument("--seed", type=int, default=2025, help="RNG seed.")
    p.add_argument("--log", type=str, default="INFO", help="Logging level (DEBUG, INFO, WARNING, ERROR).")
    p.add_argument("--progress", action="store_true", help="Use tqdm progress bar if available.")
    return p.parse_args()


# ---------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------
def maybe_tqdm(it: Iterable, enable: bool, desc: str) -> Iterable:
    if not enable:
        return it
    try:
        from tqdm import tqdm  # type: ignore
        return tqdm(it, desc=desc)
    except Exception:
        logging.warning("tqdm not available; continuing without a progress bar.")
        return it


def parse_priors(specs: List[str]) -> List[Dict[str, float]]:
    out = []
    for s in specs:
        try:
            name, a, b = s.split(":")
            out.append({"name": name, "alpha": float(a), "beta": float(b)})
        except Exception as e:
            raise ValueError(f"Bad --priors item '{s}'. Expected 'name:alpha:beta'.") from e
    return out


def prepare_dataframe(
    csv_path: Path,
    pred_col: str,
    h_col: str | None,
    label_col: str,
    h_threshold: float,
) -> pd.DataFrame:
    df = pd.read_csv(csv_path)

    if pred_col not in df.columns:
        raise ValueError(f"Missing prediction column '{pred_col}'. Found: {list(df.columns)}")
    if h_col is None and label_col not in df.columns:
        raise ValueError(f"Need either numeric H column via --h-col or categorical --label-col (default '{label_col}').")

    # H as numeric 0/1
    if h_col and h_col in df.columns:
        H = pd.to_numeric(df[h_col], errors="coerce")
        if not set(H.dropna().unique()).issubset({0, 1}):
            raise ValueError(f"Column '{h_col}' must be binary in {{0,1}}.")
        df["H"] = H.astype("Int64")
    else:
        # derive from label column: AD -> 1, else 0
        s = df[label_col].astype(str).str.strip().str.upper()
        df["H"] = (s == "AD").astype(int)

    # Predictions
    p = pd.to_numeric(df[pred_col], errors="coerce")
    if p.isna().any():
        n_bad = int(p.isna().sum())
        logging.warning("Found %d rows with non-numeric predictions in '%s'; dropping them.", n_bad, pred_col)
    df = df.loc[p.notna()].copy()
    df["autorater_prediction"] = p.loc[p.notna()].astype(float)

    # Threshold to define A_class
    df["A_class"] = (df["autorater_prediction"] >= float(h_threshold)).astype(int)

    return df


# ---------------------------------------------------------------------
# Estimators
# ---------------------------------------------------------------------
def chain_rule_estimator(
    df: pd.DataFrame,
    labeled_idx: np.ndarray,
    alpha: float,
    beta: float,
    draws: int,
    tune: int,
    chains: int,
    cores: int,
    target_accept: float,
    random_seed: int,
    progressbar: bool,
) -> Tuple[float, np.ndarray]:
    """
    Bayesian chain-rule estimator (CRE) with Beta(alpha, beta) priors:

      θ_A  ~ Beta(alpha, beta)
      θ_H1 ~ Beta(alpha, beta)
      θ_H0 ~ Beta(alpha, beta)

      NA1  ~ Binomial(N, θ_A)
      H1   ~ Binomial(n1, θ_H1)
      H0   ~ Binomial(n0, θ_H0)

      g = θ_A*θ_H1 + (1-θ_A)*θ_H0
    """
    # Counts based on labeled subset
    sub = df.iloc[labeled_idx]
    n1 = int((sub["A_class"] == 1).sum())
    H1 = int(sub.loc[sub["A_class"] == 1, "H"].sum())
    n0 = int((sub["A_class"] == 0).sum())
    H0 = int(sub.loc[sub["A_class"] == 0, "H"].sum())

    # Unlabeled pool stats (from full df)
    N = int(len(df))
    NA1 = int(df["A_class"].sum())

    with pm.Model() as model:
        theta_A = pm.Beta("theta_A", alpha, beta)
        theta_H1 = pm.Beta("theta_H1", alpha, beta)
        theta_H0 = pm.Beta("theta_H0", alpha, beta)

        pm.Binomial("obs_A", n=N, p=theta_A, observed=NA1)
        pm.Binomial("obs_H1", n=n1, p=theta_H1, observed=H1)
        pm.Binomial("obs_H0", n=n0, p=theta_H0, observed=H0)

        g = pm.Deterministic("g", theta_H1 * theta_A + theta_H0 * (1.0 - theta_A))

        idata = pm.sample(
            draws=draws,
            tune=tune,
            chains=chains,
            cores=cores,
            target_accept=target_accept,
            random_seed=random_seed,
            progressbar=progressbar,
            return_inferencedata=True,
        )

    g_samples = idata.posterior["g"].values.reshape(-1)
    ci = np.quantile(g_samples, [0.025, 0.975])
    return float(g_samples.mean()), ci


def naive_estimator(
    df: pd.DataFrame,
    labeled_idx: np.ndarray,
    alpha: float,
    beta: float,
    rng: np.random.Generator,
    n_draws: int = 2000,
) -> Tuple[float, np.ndarray]:
    """
    Labeled-only Beta-Bernoulli estimator for g = P(H=1).
    """
    sub = df.iloc[labeled_idx]
    n = int(len(sub))
    H_sum = int(sub["H"].sum())
    a_post = alpha + H_sum
    b_post = beta + n - H_sum
    samples = rng.beta(a_post, b_post, size=n_draws)
    ci = np.quantile(samples, [0.025, 0.975])
    return float(samples.mean()), ci


def difference_estimator(
    df: pd.DataFrame,
    labeled_idx: np.ndarray,
    rng: np.random.Generator,
    B: int,
) -> Tuple[float, np.ndarray]:
    """
    Difference estimator:
      g_hat = E[A] + mean(H - A) over labeled subset
    with bootstrap CI via resampling the residuals H-A.
    """
    A_bar = float(df["A_class"].mean())
    resid = (df.iloc[labeled_idx]["H"] - df.iloc[labeled_idx]["A_class"]).to_numpy(dtype=float)
    g_hat = A_bar + float(resid.mean())

    boots = np.empty(B, dtype=float)
    n = resid.size
    for b in range(B):
        idx = rng.integers(0, n, size=n)
        boots[b] = A_bar + float(resid[idx].mean())

    ci = np.quantile(boots, [0.025, 0.975])
    return g_hat, ci


# ---------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------
def main() -> None:
    args = parse_args()
    setup_logging(args.log)

    out_dir: Path = args.out
    out_dir.mkdir(parents=True, exist_ok=True)

    # Data
    df = prepare_dataframe(
        csv_path=args.csv,
        pred_col=args.pred_col,
        h_col=args.h_col,
        label_col=args.label_col,
        h_threshold=args.h_threshold,
    )

    # Ground truth and pool size
    N = int(len(df))
    g_true = float(df["H"].mean())
    logging.info("Dataset size N=%d, empirical g_true=%.6f", N, g_true)

    # Priors
    priors = parse_priors(args.priors)
    logging.info("Priors: %s", ", ".join([f"{p['name']}({p['alpha']},{p['beta']})" for p in priors]))
    logging.info("Label sizes: %s", args.label_sizes)

    # RNG
    rng = np.random.default_rng(args.seed)

    # Results container
    all_results: List[Dict[str, float]] = []

    for prior in priors:
        pname, alpha, beta = prior["name"], float(prior["alpha"]), float(prior["beta"])
        for nh in args.label_sizes:
            cov_counts = {"chain": 0, "naive": 0, "diff": 0}
            widths = {"chain": [], "naive": [], "diff": []}

            it = maybe_tqdm(range(args.nsim), enable=args.progress, desc=f"{pname} | labels={nh}")
            for i in it:
                # sample labeled indices without replacement
                if nh > N:
                    raise ValueError(f"Label size {nh} exceeds dataset size {N}.")
                labeled_idx = rng.choice(N, size=nh, replace=False)

                # Chain-rule estimator (PyMC); per-replicate seed for stability
                m_c, ci_c = chain_rule_estimator(
                    df=df,
                    labeled_idx=labeled_idx,
                    alpha=alpha,
                    beta=beta,
                    draws=args.draws,
                    tune=args.tune,
                    chains=args.chains,
                    cores=args.cores,
                    target_accept=args.target_accept,
                    random_seed=int(args.seed + i),
                    progressbar=args.progressbar,
                )
                cov_counts["chain"] += int(ci_c[0] <= g_true <= ci_c[1])
                widths["chain"].append(float(ci_c[1] - ci_c[0]))

                # Naïve estimator
                m_n, ci_n = naive_estimator(df=df, labeled_idx=labeled_idx, alpha=alpha, beta=beta, rng=rng)
                cov_counts["naive"] += int(ci_n[0] <= g_true <= ci_n[1])
                widths["naive"].append(float(ci_n[1] - ci_n[0]))

                # Difference estimator
                m_d, ci_d = difference_estimator(df=df, labeled_idx=labeled_idx, rng=rng, B=args.bootstrap)
                cov_counts["diff"] += int(ci_d[0] <= g_true <= ci_d[1])
                widths["diff"].append(float(ci_d[1] - ci_d[0]))

            all_results.append(
                {
                    "prior": pname,
                    "n_labels": int(nh),
                    "chain_cov": cov_counts["chain"] / args.nsim,
                    "chain_w": float(np.mean(widths["chain"])) if widths["chain"] else float("nan"),
                    "naive_cov": cov_counts["naive"] / args.nsim,
                    "naive_w": float(np.mean(widths["naive"])) if widths["naive"] else float("nan"),
                    "diff_cov": cov_counts["diff"] / args.nsim,
                    "diff_w": float(np.mean(widths["diff"])) if widths["diff"] else float("nan"),
                }
            )

    # Save & report
    res_df = pd.DataFrame(all_results)
    out_csv = out_dir / "simulation_results_prior_sensitivity.csv"
    res_df.to_csv(out_csv, index=False)
    logging.info("Saved results CSV: %s", out_csv.resolve())
    logging.info("\n%s", res_df.to_string(index=False))


if __name__ == "__main__":
    main()
