#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
coverage_interval_6570.py

Coverage & interval width vs. number of labels (65–70 subset).
- Headless matplotlib backend (Agg) for CI/docs
- CLI-driven; reads CSV or falls back to baked-in defaults (your latest numbers)
- Optional overlay of PPI(analytic) line (independent of prior)
- Input validation + logging

Expected CSV columns (required)
--------------------------------
prior, n_labels, chain_cov, chain_w, naive_cov, naive_w, diff_cov, diff_w

Optional CSV columns (for PPI overlay)
--------------------------------------
ppi_n, ppi_cov, ppi_w     # may repeat per-row; uniqued/sorted by n

Examples
--------
# 1) Built-in defaults (65–70 subset numbers) → save PNGs to ./figures
python coverage_interval_6570.py --out figures/

# 2) From CSV (no PPI overlay)
python coverage_interval_6570.py --csv data/coverage_6570.csv --out figures/

# 3) From CSV + PPI arrays via CLI (comma-separated)
python coverage_interval_6570.py --csv data/coverage_6570.csv --out figures/ \
  --ppi-n 10,20,40,80 --ppi-cov 0.96,0.98,0.98,0.98 \
  --ppi-w 0.527969,0.345964,0.233935,0.165516
"""
from __future__ import annotations

import argparse
import logging
from pathlib import Path
from typing import List, Tuple, Optional

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
    p = argparse.ArgumentParser(description="Plot coverage and interval width vs. number of labels (65–70 subset).")
    p.add_argument("--csv", type=Path, default=None, help="Optional CSV with required columns.")
    p.add_argument("--out", type=Path, required=True, help="Output directory for figures.")
    p.add_argument("--dpi", type=int, default=200, help="DPI for PNGs.")
    p.add_argument("--style", type=str, default="default", help="matplotlib style (e.g., 'default', 'ggplot').")
    p.add_argument("--coverage-ylim", type=float, nargs=2, default=(0.3, 1.05),
                   help="Y-limits for coverage plot.")
    p.add_argument("--width-ylim", type=float, nargs=2, default=None,
                   help="Y-limits for interval width plot (e.g., 0 0.5).")

    # Optional PPI overlay via CLI (comma-separated)
    p.add_argument("--ppi-n", type=str, default=None, help="Comma-separated n_labels for PPI (e.g., 10,20,40,80).")
    p.add_argument("--ppi-cov", type=str, default=None, help="Comma-separated PPI coverage values.")
    p.add_argument("--ppi-w", type=str, default=None, help="Comma-separated PPI interval width values.")

    p.add_argument("--log", type=str, default="INFO", help="Logging level (DEBUG, INFO, WARNING, ERROR).")
    return p.parse_args()


def parse_csv_series(s: Optional[str], cast=float) -> Optional[list]:
    if s is None:
        return None
    vals = [x.strip() for x in s.split(",") if x.strip() != ""]
    return [cast(x) for x in vals]


def load_data(csv_path: Path | None) -> tuple[pd.DataFrame, Optional[pd.DataFrame], bool]:
    """
    Returns:
      df_main: required columns
      df_ppi:  optional PPI columns (ppi_n, ppi_cov, ppi_w) if present or None
      used_default: whether baked-in defaults were used
    """
    if csv_path is not None:
        df = pd.read_csv(csv_path)
        logging.info("Loaded CSV: %s (n=%d)", csv_path, len(df))

        # Detect optional PPI columns (case-insensitive)
        cols_lower = {c.lower(): c for c in df.columns}
        has_ppi = all(k in cols_lower for k in ("ppi_n", "ppi_cov", "ppi_w"))

        df_ppi = None
        if has_ppi:
            df_ppi = df[[cols_lower["ppi_n"], cols_lower["ppi_cov"], cols_lower["ppi_w"]]].copy()
            df_ppi.columns = ["ppi_n", "ppi_cov", "ppi_w"]
            logging.info("Detected PPI columns in CSV; will overlay PPI line.")

        return df, df_ppi, False

    # Built-in default dataset (your updated 65–70 numbers)
    data = {
        "prior": ["uniform"] * 4 + ["jeffreys"] * 4,
        "n_labels": [10, 20, 40, 80] * 2,

        # CRE (Chain-rule)
        "chain_cov": [1.00, 1.00, 0.98, 1.00, 1.00, 1.00, 0.98, 1.00],
        "chain_w":   [0.379, 0.259, 0.194, 0.144, 0.344, 0.254, 0.182, 0.139],

        # Naïve
        "naive_cov": [0.98, 1.00, 0.98, 1.00, 0.90, 0.98, 0.94, 0.98],
        "naive_w":   [0.461, 0.336, 0.249, 0.179, 0.434, 0.343, 0.247, 0.179],

        # Difference
        "diff_cov":  [0.58, 0.62, 0.94, 1.00, 0.44, 0.76, 0.96, 0.96],
        "diff_w":    [0.202, 0.139, 0.144, 0.110, 0.160, 0.172, 0.138, 0.106],
    }
    df_main = pd.DataFrame(data)

    # Built-in PPI line (estimator-only; independent of prior)
    df_ppi = pd.DataFrame({
        "ppi_n":   [10, 20, 40, 80],
        "ppi_cov": [0.96, 0.98, 0.98, 0.98],
        "ppi_w":   [0.527969, 0.345964, 0.233935, 0.165516],
    })

    logging.info("Using built-in default dataset (n=%d) with PPI overlay.", len(df_main))
    return df_main, df_ppi, True


def validate_columns(df: pd.DataFrame) -> None:
    missing = [c for c in REQUIRED_COLS if c not in df.columns]
    if missing:
        raise ValueError(f"Missing required columns: {missing}. Found: {list(df.columns)}")


def coerce_numeric(df: pd.DataFrame) -> pd.DataFrame:
    for c in df.columns:
        if c != "prior":
            df[c] = pd.to_numeric(df[c], errors="coerce")
    return df


def prepare_ppi_from_cli(args: argparse.Namespace) -> Optional[pd.DataFrame]:
    ppi_n = parse_csv_series(args.ppi_n, cast=int)
    ppi_cov = parse_csv_series(args.ppi_cov, cast=float)
    ppi_w = parse_csv_series(args.ppi_w, cast=float)
    if ppi_n is None and ppi_cov is None and ppi_w is None:
        return None
    if not (ppi_n and ppi_cov and ppi_w) or not (len(ppi_n) == len(ppi_cov) == len(ppi_w)):
        raise ValueError("--ppi-n, --ppi-cov, --ppi-w must be provided together with equal lengths.")
    return pd.DataFrame({"ppi_n": ppi_n, "ppi_cov": ppi_cov, "ppi_w": ppi_w})


def plot_coverage(
    df: pd.DataFrame,
    out_path: Path,
    dpi: int,
    ylim: Tuple[float, float] | None,
    df_ppi: Optional[pd.DataFrame],
) -> None:
    plt.figure(figsize=(10, 5))
    for prior in ["uniform", "jeffreys"]:
        subset = df[df["prior"] == prior].sort_values("n_labels")
        if subset.empty:
            continue
        plt.plot(subset["n_labels"], subset["chain_cov"], "o-", label=f"Chain ({prior})")
        plt.plot(subset["n_labels"], subset["naive_cov"], "s--", label=f"Naive ({prior})")
        plt.plot(subset["n_labels"], subset["diff_cov"], "d-.", label=f"Diff ({prior})")

    # PPI overlay (estimator-only)
    if df_ppi is not None and not df_ppi.empty:
        ppi_line = df_ppi.dropna().drop_duplicates(subset=["ppi_n"]).sort_values("ppi_n")
        plt.plot(ppi_line["ppi_n"], ppi_line["ppi_cov"], "^:", label="PPI (analytic)")

    if ylim:
        plt.ylim(*ylim)
    plt.xlabel("Number of Labels")
    plt.ylabel("Coverage")
    plt.title("Coverage vs. Number of Labels (65–70 subset)")
    plt.grid(True, alpha=0.3, linestyle="--")
    plt.legend()
    plt.tight_layout()
    plt.savefig(out_path, dpi=dpi)
    plt.close()
    logging.info("Saved: %s", out_path.resolve())


def plot_width(
    df: pd.DataFrame,
    out_path: Path,
    dpi: int,
    ylim: Tuple[float, float] | None,
    df_ppi: Optional[pd.DataFrame],
) -> None:
    plt.figure(figsize=(10, 5))
    for prior in ["uniform", "jeffreys"]:
        subset = df[df["prior"] == prior].sort_values("n_labels")
        if subset.empty:
            continue
        plt.plot(subset["n_labels"], subset["chain_w"], "o-", label=f"Chain ({prior})")
        plt.plot(subset["n_labels"], subset["naive_w"], "s--", label=f"Naive ({prior})")
        plt.plot(subset["n_labels"], subset["diff_w"], "d-.", label=f"Diff ({prior})")

    # PPI overlay (estimator-only)
    if df_ppi is not None and not df_ppi.empty:
        ppi_line = df_ppi.dropna().drop_duplicates(subset=["ppi_n"]).sort_values("ppi_n")
        plt.plot(ppi_line["ppi_n"], ppi_line["ppi_w"], "^:", label="PPI (analytic)")

    if ylim:
        plt.ylim(*ylim)
    plt.xlabel("Number of Labels")
    plt.ylabel("Interval Width")
    plt.title("Interval Width vs. Number of Labels (65–70 subset)")
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

    df, df_ppi_csv, used_default = load_data(args.csv)

    # Validate & coerce main data
    validate_columns(df)
    df = coerce_numeric(df)
    df = df.dropna(subset=["n_labels"]).sort_values(["prior", "n_labels"])

    # PPI from CLI overrides CSV; if neither present and default dataset used, df_ppi is already set.
    df_ppi_cli = prepare_ppi_from_cli(args)
    if df_ppi_cli is not None:
        df_ppi = df_ppi_cli
        logging.info("Using PPI series from CLI.")
    else:
        df_ppi = df_ppi_csv
        if df_ppi is not None:
            logging.info("Using PPI series from CSV.")
        elif used_default:
            logging.info("Using PPI series from built-in defaults.")
        else:
            logging.info("No PPI overlay (CSV provided without PPI columns, and no CLI PPI).")

    # Plots
    plot_coverage(
        df=df,
        out_path=out_dir / "coverage_vs_labels_6570.png",
        dpi=args.dpi,
        ylim=tuple(args.coverage_ylim) if args.coverage_ylim else None,
        df_ppi=df_ppi,
    )
    plot_width(
        df=df,
        out_path=out_dir / "interval_width_vs_labels_6570.png",
        dpi=args.dpi,
        ylim=None if args.width_ylim is None else tuple(args.width_ylim),
        df_ppi=df_ppi,
    )

    logging.info("All done. Figures written to: %s", out_dir.resolve())


if __name__ == "__main__":
    main()
