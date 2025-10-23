#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
histogram_posterior.py

Bayesian chain-rule (PPI-style) posterior summaries over repeated simulations,
rewritten for GitHub/TMLR-friendly standards:

- No hard-coded paths: all I/O and hyperparameters via CLI
- Reproducibility: single RNG seed governs NumPy + PyMC
- Headless plotting with matplotlib (no seaborn)
- Saves figures/CSVs instead of interactive .show()
- Clear modular functions, logging, and type hints

Example
-------
python histogram_posterior.py \
  --out runs/hist/ \
  --M 50 --N-A 1000 --N-H 100 \
  --theta-A 0.6 --theta-H1 0.8 --theta-H0 0.3 \
  --draws 500 --tune 500 --chains 2 --target-accept 0.9 \
  --seed 2025 --log INFO
"""
from __future__ import annotations

import argparse
import logging
from pathlib import Path
from typing import Tuple, Dict

import matplotlib

# headless backend (CI/docs/servers)
matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
import pymc as pm  # noqa: E402
import pandas as pd  # noqa: E402


# ---------------------------------------------------------------------
# Logging
# ---------------------------------------------------------------------
def setup_logging(level: str = "INFO") -> None:
    log_level = getattr(logging, level.upper(), logging.INFO)
    logging.basicConfig(level=log_level, format="[%(levelname)s] %(message)s")


# ---------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------
def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description="Posterior histograms across repeated simulations for the Bayesian chain-rule estimator."
    )
    p.add_argument("--out", type=Path, required=True, help="Output directory for figures and CSVs.")
    p.add_argument("--M", type=int, default=50, help="Number of simulated datasets.")
    p.add_argument("--N-A", dest="N_A", type=int, default=1000, help="Number of unlabeled A observations.")
    p.add_argument("--N-H", dest="N_H", type=int, default=100, help="Number of labeled (A,H) pairs.")
    p.add_argument("--theta-A", dest="theta_A", type=float, default=0.6, help="True P(A=1).")
    p.add_argument("--theta-H1", dest="theta_H1", type=float, default=0.8, help="True P(H=1|A=1).")
    p.add_argument("--theta-H0", dest="theta_H0", type=float, default=0.3, help="True P(H=1|A=0).")

    # MCMC controls
    p.add_argument("--draws", type=int, default=500, help="Posterior draws per chain.")
    p.add_argument("--tune", type=int, default=500, help="Tuning steps per chain.")
    p.add_argument("--chains", type=int, default=2, help="Number of chains.")
    p.add_argument("--cores", type=int, default=1, help="Cores for sampling (1 is safest cross-platform).")
    p.add_argument("--target-accept", dest="target_accept", type=float, default=0.9, help="NUTS target_accept.")
    p.add_argument("--progressbar", action="store_true", help="Show PyMC progress bar.")

    # misc
    p.add_argument("--seed", type=int, default=2025, help="RNG seed (NumPy+PyMC).")
    p.add_argument("--log", type=str, default="INFO", help="Logging level (DEBUG, INFO, WARNING, ERROR).")
    return p.parse_args()


# ---------------------------------------------------------------------
# Simulation + Inference
# ---------------------------------------------------------------------
def simulate_dataset(
    rng: np.random.Generator, N_A: int, N_H: int, theta_A: float, theta_H1: float, theta_H0: float
) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """
    Generate one synthetic dataset:
      - A_data: size N_A Bernoulli(theta_A)
      - A_H: first N_H of A_data (paired with H)
      - H_data: Bernoulli(theta_H1) if A_H=1 else Bernoulli(theta_H0)
    """
    A_data = rng.binomial(1, theta_A, size=N_A).astype(int)
    A_H = A_data[:N_H]
    H_data = np.array([rng.binomial(1, theta_H1 if a == 1 else theta_H0) for a in A_H], dtype=int)
    return A_data, A_H, H_data


def fit_chain_rule_posterior(
    A_data: np.ndarray,
    A_H: np.ndarray,
    H_data: np.ndarray,
    draws: int,
    tune: int,
    chains: int,
    cores: int,
    target_accept: float,
    random_seed: int,
    progressbar: bool,
) -> Dict[str, np.ndarray]:
    """
    Fit the simple chain-rule model with weak Beta(1,1) priors and return posterior draws of g.
    """
    with pm.Model() as model:
        theta_A = pm.Beta("theta_A", 1, 1)
        theta_H1 = pm.Beta("theta_H1", 1, 1)
        theta_H0 = pm.Beta("theta_H0", 1, 1)

        pm.Bernoulli("A_obs", p=theta_A, observed=A_data)
        pm.Bernoulli("H_obs", p=theta_H1 * A_H + theta_H0 * (1 - A_H), observed=H_data)

        g = pm.Deterministic("g", theta_H1 * theta_A + theta_H0 * (1 - theta_A))

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
    return {"g": g_samples}


# ---------------------------------------------------------------------
# Plotting
# ---------------------------------------------------------------------
def plot_hist(
    samples: np.ndarray,
    true_val: float,
    title: str,
    xlabel: str,
    out_path: Path,
    bins: int = 15,
) -> None:
    plt.figure(figsize=(6.2, 4.2))
    plt.hist(samples, bins=bins, density=False)
    plt.axvline(true_val, linestyle="--")
    plt.title(title)
    plt.xlabel(xlabel)
    plt.ylabel("Frequency")
    plt.tight_layout()
    plt.savefig(out_path, dpi=200)
    plt.close()


# ---------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------
def main() -> None:
    args = parse_args()
    setup_logging(args.log)

    out_dir = args.out
    out_dir.mkdir(parents=True, exist_ok=True)

    # True quantity g
    true_g = args.theta_A * args.theta_H1 + (1 - args.theta_A) * args.theta_H0
    logging.info("True parameters: θ_A=%.3f, θ_H|1=%.3f, θ_H|0=%.3f -> g=%.3f", args.theta_A, args.theta_H1, args.theta_H0, true_g)

    # Storage
    means = np.zeros(args.M)
    medians = np.zeros(args.M)
    ci_lower = np.zeros(args.M)
    ci_upper = np.zeros(args.M)

    # RNG (NumPy) seeded once; pass deterministic seeds to PyMC per replicate
    rng = np.random.default_rng(args.seed)

    for i in range(args.M):
        # Simulate data
        A_data, A_H, H_data = simulate_dataset(
            rng=rng, N_A=args.N_A, N_H=args.N_H, theta_A=args.theta_A, theta_H1=args.theta_H1, theta_H0=args.theta_H0
        )

        # Fit posterior (deterministic per i)
        posterior = fit_chain_rule_posterior(
            A_data=A_data,
            A_H=A_H,
            H_data=H_data,
            draws=args.draws,
            tune=args.tune,
            chains=args.chains,
            cores=args.cores,
            target_accept=args.target_accept,
            random_seed=int(args.seed + i),
            progressbar=args.progressbar,
        )
        g_samples = posterior["g"]

        # Summaries
        means[i] = float(np.mean(g_samples))
        medians[i] = float(np.median(g_samples))
        ci_lower[i], ci_upper[i] = np.percentile(g_samples, [2.5, 97.5])

    # Empirical coverage
    coverage = float(np.mean((ci_lower <= true_g) & (true_g <= ci_upper)))
    logging.info("Empirical coverage of 95%% CI: %.1f%%", 100 * coverage)

    # Save summaries CSV
    df = pd.DataFrame(
        {
            "replicate": np.arange(args.M, dtype=int),
            "post_mean_g": means,
            "post_median_g": medians,
            "ci2p5": ci_lower,
            "ci97p5": ci_upper,
        }
    )
    csv_path = out_dir / "posterior_summary_over_reps.csv"
    df.to_csv(csv_path, index=False)
    logging.info("Saved CSV: %s", csv_path.resolve())

    # Plots (saved)
    plot_hist(
        means,
        true_val=true_g,
        title="Histogram of Posterior Means of g",
        xlabel="Posterior mean of g",
        out_path=out_dir / "hist_posterior_means_g.png",
    )
    plot_hist(
        medians,
        true_val=true_g,
        title="Histogram of Posterior Medians of g",
        xlabel="Posterior median of g",
        out_path=out_dir / "hist_posterior_medians_g.png",
    )

    # Coverage file
    with open(out_dir / "coverage.txt", "w", encoding="utf-8") as f:
        f.write(f"Empirical coverage of 95% CI: {coverage*100:.1f}%\n")
        f.write(f"True g: {true_g:.6f}\n")

    logging.info("Saved figures: %s, %s",
                 (out_dir / "hist_posterior_means_g.png").resolve(),
                 (out_dir / "hist_posterior_medians_g.png").resolve())
    logging.info("Done.")


if __name__ == "__main__":
    main()
