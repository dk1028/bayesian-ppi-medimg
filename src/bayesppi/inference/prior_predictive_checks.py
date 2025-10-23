#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
prior_predictive_checks.py

Purpose
-------
Prior predictive simulation for the chain-rule target:
    g = theta_A * theta_H1 + (1 - theta_A) * theta_H0

This script keeps the original values and intent:
- RNG seed = 42
- n = 20000 draws
- theta_A, theta_H1, theta_H0 ~ Beta(1, 1)
- Histogram with 50 bins, density=True
- Vertical line at 0.5 (mean of g under the prior)

Repo/TMLR-friendly changes:
- Headless matplotlib backend (Agg)
- CLI flag for output path (default: figures/prior_predictive_checks.png)
- No interactive windows; saves figure to file
- Lightweight logging
"""

from __future__ import annotations

import argparse
import logging
from pathlib import Path

import matplotlib

# Headless for CI/docs
matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402


def setup_logging(level: str = "INFO") -> None:
    logging.basicConfig(
        level=getattr(logging, level.upper(), logging.INFO),
        format="[%(levelname)s] %(message)s",
    )


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Prior predictive simulation for g.")
    # Keep original default file intent, but write to repo-friendly path by default.
    p.add_argument(
        "--out",
        type=Path,
        default=Path("figures/prior_predictive_checks.png"),
        help="Output image path (.png, .svg, ...). Default: figures/prior_predictive_checks.png",
    )
    # Expose bins only to mirror original (default=50). Other values are fixed per requirement.
    p.add_argument(
        "--bins",
        type=int,
        default=50,
        help="Number of histogram bins (default: 50).",
    )
    p.add_argument(
        "--log",
        type=str,
        default="INFO",
        help="Logging level (DEBUG, INFO, WARNING, ERROR).",
    )
    return p.parse_args()


def main() -> None:
    args = parse_args()
    setup_logging(args.log)

    # --- Fixed values by requirement ---
    seed = 42
    n = 20000
    a, b = 1.0, 1.0  # Beta(1,1)
    # -----------------------------------

    rng = np.random.default_rng(seed)
    theta_A = rng.beta(a, b, n)
    theta_H1 = rng.beta(a, b, n)
    theta_H0 = rng.beta(a, b, n)
    g = theta_A * theta_H1 + (1.0 - theta_A) * theta_H0

    # Plot (headless)
    args.out.parent.mkdir(parents=True, exist_ok=True)
    plt.figure(figsize=(6, 4))
    plt.hist(g, bins=args.bins, density=True)
    plt.axvline(0.5, linestyle="--")
    plt.xlabel("g = theta_A*theta_H1 + (1-theta_A)*theta_H0")
    plt.ylabel("Density")
    plt.title("Prior predictive distribution of g")
    plt.tight_layout()
    plt.savefig(args.out, dpi=200)
    plt.close()

    logging.info("Saved prior predictive figure: %s", args.out.resolve())


if __name__ == "__main__":
    main()
