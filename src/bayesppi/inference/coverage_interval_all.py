#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
coverage_interval_all.py

Reproducible plotting script for coverage and interval width vs. number of labels
for the "ALL" simulation setting. Designed to be GitHub/TMLR friendly:

- Headless matplotlib backend (Agg) for CI/docs
- No interactive windows; saves figures to files
- CLI-driven; can read an external CSV or use baked-in defaults
- Input validation + logging
- Minimal dependencies (pandas, matplotlib)

Expected data schema (columns)
------------------------------
prior, n_labels, chain_cov, chain_w, naive_cov, naive_w, diff_cov, diff_w

Examples
--------
# 1) Use built-in default dataset and write PNGs to ./figures
python coverage_interval_all.py --out figures/

# 2) Use a CSV file with the required columns
python coverage_interval_all.py --csv data/coverage_all.csv --out figures/ --dpi 200 --style default
"""
from __future__ import annotations

import argparse
import logging
from pathlib import Path
from typing import List

import matplotlib

# Headless backend for CI/docs/servers
matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import pandas as pd  # noqa: E402


REQUIRED_COLS: List[str] = [
    "prior",
    "n_labels",
    "chain_cov",
    "chain_w",
    "naive_cov",
    "naive_w",
    "diff_cov",
    "diff_w",
]


def setup_logging(level: str = "INFO") -> None:
    logging.basicConfig(level=getattr(logging, level.upper(), logging.INFO),
                        format="[%(levelname)s] %(message)s")


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Plot coverage and interval width vs. number of labels (ALL setting).")
    p.add_argument("--csv", type=Path, default=None, help="Optional CSV with required columns.")
    p.add_argument("--out", type=Path, required=True, help="Output directory for figures.")
    p.add_argument("--dpi", type=int, default=200, help="DPI for PNGs.")
    p.add_argument("--style", type=str, default="default", help="matplotlib style (e.g., 'default', 'ggplot').")
    p.add_argument("--coverage-ylim", type=float, nargs=2, default=(0.4, 1.05),
                   help="Y-limits for coverage plot.")
    p.add_argument("--width-ylim", type=float, nargs=2, default=None,
                   help="Y-limits for interval width plot (e.g., 0 0.5).")
    p.add_argument("--log", type=str, default="INFO", help="Logging level (DEBUG, INFO, WARNING, ERROR).")
    return p.parse_args()


def load_data(csv_path: Path | None) -> pd.DataFrame:
    if csv_path is not None:
        df = pd.read_csv(csv_path)
        logging.info("Loaded CSV: %s (n=%d)", csv_path, len(df))
        return df

    # Built-in default dataset (from original snippet)
    data = {
        "prior": ["uniform"] * 4 + ["jeffreys"] * 4,
        "n_labels": [10, 20, 40, 80] * 2,
        "chain_cov": [0.96, 1.00, 0.94, 0.94, 1.00, 1.00, 0.90, 0.92],
        "chain_w": [0.396875, 0.271335, 0.189356, 0.131579, 0.369410, 0.262156, 0.178783, 0.122471],
        "naive_cov": [0.92, 0.96, 0.96, 0.94, 0.98, 0.90, 0.92, 0.94],
        "naive_w": [0.481963, 0.362300, 0.272014, 0.196974, 0.484546, 0.357574, 0.269901, 0.199326],
        "diff_cov": [0.56, 0.68, 0.88, 0.88, 0.48, 0.72, 0.86, 0.90],
        "diff_w": [0.288050, 0.202000, 0.174525, 0.129263, 0.238000, 0.208075, 0.164525, 0.123537],
    }
    df = pd.DataFrame(data)
    logging.info("Using built-in default dataset (n=%d).", len(df))
    return df


def validate_columns(df: pd.DataFrame) -> None:
    missing = [c for c in REQUIRED_COLS if c not in df.columns]
    if missing:
        raise ValueError(f"Missing required columns: {missing}. Found: {list(df.columns)}")


def plot_coverage(df: pd.DataFrame, out_path: Path, dpi: int, ylim: tuple[float, float] | None) -> None:
    plt.figure(figsize=(10, 5))
    for prior in ["uniform", "jeffreys"]:
        subset = df[df["prior"] == prior].sort_values("n_labels")
        plt.plot(subset["n_labels"], subset["chain_cov"], "o-", label=f"Chain ({prior})")
        plt.plot(subset["n_labels"], subset["naive_cov"], "s--", label=f"Naive ({prior})")
        plt.plot(subset["n_labels"], subset["diff_cov"], "d-.", label=f"Diff ({prior})")
    if ylim:
        plt.ylim(*ylim)
    plt.xlabel("Number of Labels")
    plt.ylabel("Coverage")
    plt.title("Coverage vs. Number of Labels (ALL)")
    plt.grid(True, alpha=0.3, linestyle="--")
    plt.legend()
    plt.tight_layout()
    plt.savefig(out_path, dpi=dpi)
    plt.close()
    logging.info("Saved: %s", out_path.resolve())


def plot_width(df: pd.DataFrame, out_path: Path, dpi: int, ylim: tuple[float, float] | None) -> None:
    plt.figure(figsize=(10, 5))
    for prior in ["uniform", "jeffreys"]:
        subset = df[df["prior"] == prior].sort_values("n_labels")
        plt.plot(subset["n_labels"], subset["chain_w"], "o-", label=f"Chain ({prior})")
        plt.plot(subset["n_labels"], subset["naive_w"], "s--", label=f"Naive ({prior})")
        plt.plot(subset["n_labels"], subset["diff_w"], "d-.", label=f"Diff ({prior})")
    if ylim:
        plt.ylim(*ylim)
    plt.xlabel("Number of Labels")
    plt.ylabel("Interval Width")
    plt.title("Interval Width vs. Number of Labels (ALL)")
    plt.grid(True, alpha=0.3, linestyle="--")
    plt.legend()
    plt.tight_layout()
    plt.savefig(out_path, dpi=dpi)
    plt.close()
    logging.info("Saved: %s", out_path.resolve())


def main() -> None:
    args = parse_args()
    setup_logging(args.log)

    # style
    try:
        plt.style.use(args.style)
    except Exception as e:
        logging.warning("Failed to apply style '%s': %s. Falling back to default.", args.style, e)

    out_dir: Path = args.out
    out_dir.mkdir(parents=True, exist_ok=True)

    df = load_data(args.csv)
    validate_columns(df)

    # Ensure numeric types
    num_cols = [c for c in df.columns if c != "prior"]
    df[num_cols] = df[num_cols].apply(pd.to_numeric, errors="coerce")

    # Prepare for plotting
    df = df.dropna(subset=["n_labels"]).sort_values(["prior", "n_labels"])

    # Plots
    plot_coverage(df, out_dir / "coverage_vs_labels_all.png", dpi=args.dpi, ylim=tuple(args.coverage_ylim))
    plot_width(df, out_dir / "interval_width_vs_labels_all.png",
               dpi=args.dpi,
               ylim=None if args.width_ylim is None else tuple(args.width_ylim))

    logging.info("All done. Figures written to: %s", out_dir.resolve())


if __name__ == "__main__":
    main()
