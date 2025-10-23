#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
age_effect_analysis.py

Autorater vs Age analysis with robust CLI, reproducibility, and repository-friendly structure.
Designed to meet common GitHub / TMLR expectations:
- No hard-coded paths (all I/O is CLI-driven)
- Deterministic outputs via fixed RNG seed
- Clear modular functions with type hints and docstrings
- Headless plotting (matplotlib Agg)
- Logging instead of ad-hoc prints
- CSV + PNG artifacts written under an output directory

Features
--------
- Merge predictions with ADNI metadata (exact Subject+Date, then ±k-day nearest fallback)
- Scatter: predicted probability vs Age (colored by H)
- ROC: overall + per age bin + optional single-band ROC
- Metrics per bin: n, prevalence, AUC, ACC/TPR/TNR @0.5
- AUC 95% bootstrap CI per bin + overall
- Youden’s J optimal threshold per bin (ACC/TPR/TNR @thr)
- Calibration curves + Brier score per bin (if sufficient data)

Usage
-----
python age_effect_analysis.py \
  --meta data/processed/all_people_7_20_2025.csv \
  --pred data/processed/autorater_predictions_all4.csv \
  --out  figures/age_analysis \
  --age-bins 50,74,80,101 \
  --age-labels 50-73,74-79,80-100 \
  --band 74:80 \
  --date-tol-days 14 \
  --bootstrap 2000 \
  --calib-min-n 50 \
  --seed 2025
"""
from __future__ import annotations

import argparse
import logging
import os
from pathlib import Path
from typing import Iterable, List, Optional, Sequence, Tuple

import matplotlib

# Headless backend for CI/docs/servers
matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402
from matplotlib.lines import Line2D  # noqa: E402
from sklearn.calibration import calibration_curve  # noqa: E402
from sklearn.metrics import (  # noqa: E402
    accuracy_score,
    auc,
    confusion_matrix,
    roc_auc_score,
    roc_curve,
    brier_score_loss,
)


# ---------------------------------------------------------------------
# Logging
# ---------------------------------------------------------------------
def setup_logging(level: str = "INFO") -> None:
    """Configure root logger."""
    log_level = getattr(logging, level.upper(), logging.INFO)
    logging.basicConfig(
        level=log_level,
        format="[%(levelname)s] %(message)s",
    )


# ---------------------------------------------------------------------
# IO & Parsing
# ---------------------------------------------------------------------
REQUIRED_META = ["Subject", "Age", "Sex", "Acq Date"]
REQUIRED_PRED = ["subject_id", "Acq_Date", "autorater_prediction", "H"]


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description="Autorater vs Age analysis with ROC/metrics/calibration by age bins."
    )
    p.add_argument("--meta", type=Path, required=True, help="Metadata CSV (Subject, Age, Sex, Acq Date).")
    p.add_argument("--pred", type=Path, required=True, help="Predictions CSV (subject_id, Acq_Date, autorater_prediction, H[, label]).")
    p.add_argument("--out", type=Path, required=True, help="Output directory for figures/CSVs.")
    p.add_argument("--age-bins", type=str, default="50,74,80,101", help="Inclusive-left, exclusive-right bin edges, comma-separated (e.g., 50,74,80,101).")
    p.add_argument("--age-labels", type=str, default="50-73,74-79,80-100", help="Comma-separated labels matching bins-1 (e.g., 50-73,74-79,80-100).")
    p.add_argument("--band", type=str, default="74:80", help="Optional single-band ROC as 'low:high' (e.g., 74:80).")
    p.add_argument("--date-tol-days", type=int, default=14, help="Tolerance for nearest-date matching.")
    p.add_argument("--bootstrap", type=int, default=2000, help="Bootstrap repetitions for AUC CI.")
    p.add_argument("--calib-min-n", type=int, default=50, help="Min per-bin n to draw calibration curves.")
    p.add_argument("--seed", type=int, default=2025, help="RNG seed for reproducibility.")
    p.add_argument("--log", type=str, default="INFO", help="Logging level (DEBUG, INFO, WARNING, ERROR).")
    return p.parse_args()


def ensure_required_columns(df: pd.DataFrame, req: Sequence[str], tag: str) -> None:
    missing = [c for c in req if c not in df.columns]
    if missing:
        raise ValueError(f"[{tag}] Missing columns {missing}. Current columns: {list(df.columns)}")


def parse_date_series(s: pd.Series) -> pd.Series:
    """
    Robust date parsing with a few common formats.
    Returns a series of dtype 'datetime64[ns]' (date at midnight) or NaT.
    """
    out = pd.to_datetime(s, errors="coerce", infer_datetime_format=True)
    mask = out.isna()
    if mask.any():
        try_formats = ["%m/%d/%Y", "%m/%d/%y", "%Y-%m-%d", "%Y/%m/%d", "%m-%d-%Y", "%d/%m/%Y"]
        for fmt in try_formats:
            # only attempt for still-NaT entries
            tmp = pd.to_datetime(s[mask], format=fmt, errors="coerce")
            out.loc[mask] = tmp
            mask = out.isna()
            if not mask.any():
                break
    return out.dt.normalize()  # keep as datetime (not pure date) for timedelta ops


def coerce_binary(series: pd.Series) -> pd.Series:
    """
    Coerce to binary 0/1 if possible; accepts strings like '0','1', numeric, or {CN,AD} via 'label' pre-processing.
    """
    x = pd.to_numeric(series, errors="coerce")
    # If still NaN-heavy, try mapping common strings
    if x.isna().mean() > 0.5:
        s = series.astype(str).str.upper().str.strip()
        mapping = {"CN": 0, "CONTROL": 0, "NEG": 0, "NO": 0, "AD": 1, "POS": 1, "YES": 1}
        x = s.map(mapping)
    return x.astype("Int64")


# ---------------------------------------------------------------------
# Matching
# ---------------------------------------------------------------------
def nearest_date_match(
    pred: pd.DataFrame,
    meta: pd.DataFrame,
    date_tol_days: int,
) -> pd.DataFrame:
    """
    Exact join on (subject_id, AcqDate_std); fallback to nearest date within ±date_tol_days.
    """
    meta_key = meta[["Subject", "AcqDate_std", "Age", "Sex"]].drop_duplicates()
    merged = pred.merge(
        meta_key,
        left_on=["subject_id", "AcqDate_std"],
        right_on=["Subject", "AcqDate_std"],
        how="left",
        suffixes=("", "_meta"),
    )

    need_fill = merged["Age"].isna()
    if need_fill.any():
        logging.info(
            "Exact match failed rows: %d → trying nearest-date matching (±%d days)",
            need_fill.sum(),
            date_tol_days,
        )
        meta_grp = {
            sid: df[["AcqDate_std", "Age", "Sex"]]
            .dropna(subset=["AcqDate_std"])
            .sort_values("AcqDate_std")
            for sid, df in meta.groupby("Subject", sort=False)
        }

        ages, sexes = [], []
        for _, row in merged.loc[need_fill].iterrows():
            sid = row["subject_id"]
            d0 = row["AcqDate_std"]
            age_val, sex_val = np.nan, np.nan
            if pd.notna(d0) and sid in meta_grp:
                cand = meta_grp[sid]
                if len(cand):
                    diffs = (cand["AcqDate_std"] - d0).abs().dt.days
                    j = diffs.idxmin()
                    if pd.notna(j) and diffs.loc[j] <= date_tol_days:
                        age_val = cand.loc[j, "Age"]
                        sex_val = cand.loc[j, "Sex"]
            ages.append(age_val)
            sexes.append(sex_val)
        merged.loc[need_fill, "Age"] = ages
        merged.loc[need_fill, "Sex"] = sexes

    return merged


# ---------------------------------------------------------------------
# Plotting helpers
# ---------------------------------------------------------------------
def save_scatter_pred_vs_age(
    df: pd.DataFrame,
    out_path: Path,
) -> None:
    plt.figure(figsize=(8.5, 5.2))
    colors = np.where(df["H"] == 1, "crimson", "royalblue")
    plt.scatter(
        df["Age"].values,
        df["autorater_prediction"].values,
        c=colors,
        s=16,
        alpha=0.55,
        edgecolors="none",
    )
    plt.xlabel("Age (years)")
    plt.ylabel("Autorater predicted probability")
    plt.title("Predicted Probability vs Age (colored by H)")
    legend_elems = [
        Line2D([0], [0], marker="o", color="w", label="H=0", markerfacecolor="royalblue", markersize=7),
        Line2D([0], [0], marker="o", color="w", label="H=1", markerfacecolor="crimson", markersize=7),
    ]
    plt.legend(handles=legend_elems, loc="lower right")
    plt.grid(alpha=0.25, linestyle="--")
    plt.tight_layout()
    plt.savefig(out_path, dpi=200)
    plt.close()


def plot_and_save_roc(
    y_true: np.ndarray,
    y_score: np.ndarray,
    title: str,
    savepath: Path,
) -> float:
    fpr, tpr, _ = roc_curve(y_true, y_score)
    roc_auc = auc(fpr, tpr)
    plt.figure(figsize=(5.6, 5.6))
    plt.plot(fpr, tpr, lw=2, label=f"AUC = {roc_auc:.3f}")
    plt.plot([0, 1], [0, 1], "k--", lw=1)
    plt.xlim(0, 1)
    plt.ylim(0, 1.03)
    plt.xlabel("False Positive Rate (1 - Specificity)")
    plt.ylabel("True Positive Rate (Sensitivity)")
    plt.title(title)
    plt.legend(loc="lower right")
    plt.grid(alpha=0.25, linestyle="--")
    plt.tight_layout()
    plt.savefig(savepath, dpi=200)
    plt.close()
    return roc_auc


# ---------------------------------------------------------------------
# Metrics
# ---------------------------------------------------------------------
def bootstrap_auc_ci(
    y: np.ndarray, p: np.ndarray, B: int, rng: np.random.Generator
) -> Tuple[float, float, float]:
    """Return (auc, lo, hi) with 95% bootstrap percentile CI."""
    auc0 = roc_auc_score(y, p)
    boots: List[float] = []
    n = len(y)
    for _ in range(B):
        idx = rng.integers(0, n, size=n)
        if len(np.unique(y[idx])) < 2:
            continue
        boots.append(roc_auc_score(y[idx], p[idx]))
    if len(boots) == 0:
        return auc0, np.nan, np.nan
    lo, hi = np.percentile(boots, [2.5, 97.5])
    return auc0, float(lo), float(hi)


def youden_threshold(y: np.ndarray, p: np.ndarray) -> Tuple[float, float, float]:
    """Return (thr*, TPR*, TNR*) at Youden's J optimum."""
    fpr, tpr, thr = roc_curve(y, p)
    j = tpr - fpr
    k = np.argmax(j)
    return float(thr[k]), float(tpr[k]), float(1 - fpr[k])


# ---------------------------------------------------------------------
# Main analysis
# ---------------------------------------------------------------------
def run_analysis(
    meta_csv: Path,
    pred_csv: Path,
    out_dir: Path,
    age_bins: Sequence[float],
    age_labels: Sequence[str],
    band: Optional[Tuple[float, float]],
    date_tol_days: int,
    bootstrap_B: int,
    calib_min_n: int,
    seed: int,
) -> None:
    out_dir.mkdir(parents=True, exist_ok=True)
    cal_dir = out_dir / "calibration_curves"
    cal_dir.mkdir(exist_ok=True)

    rng = np.random.default_rng(seed)

    # Load
    meta = pd.read_csv(meta_csv, dtype=str)
    pred = pd.read_csv(pred_csv, dtype=str)

    # Clean headers & check
    meta.columns = [c.strip() for c in meta.columns]
    pred.columns = [c.strip() for c in pred.columns]
    ensure_required_columns(meta, REQUIRED_META, "META")
    ensure_required_columns(pred, REQUIRED_PRED, "PRED")

    # Standardize key fields
    meta["Subject"] = meta["Subject"].astype(str).str.strip()
    pred["subject_id"] = pred["subject_id"].astype(str).str.strip()
    meta["AcqDate_std"] = parse_date_series(meta["Acq Date"])
    pred["AcqDate_std"] = parse_date_series(pred["Acq_Date"])

    meta["Age"] = pd.to_numeric(meta["Age"], errors="coerce")
    pred["autorater_prediction"] = pd.to_numeric(pred["autorater_prediction"], errors="coerce")

    # H: prefer provided numeric; fallback via 'label' if needed
    H_series = coerce_binary(pred["H"])
    if H_series.isna().any() and "label" in pred.columns:
        logging.info("Detected NaNs in 'H'; attempting to derive from 'label' column.")
        derived = coerce_binary(pred["label"])
        H_series = H_series.fillna(derived)
    pred["H"] = H_series

    # Optional AD flag (not required)
    if "label" in pred.columns and "is_AD" not in pred.columns:
        pred["is_AD"] = (pred["label"].astype(str).str.upper() == "AD").astype(int)

    # Match meta to predictions
    merged = nearest_date_match(pred, meta, date_tol_days)

    # Usable rows
    use = merged.dropna(subset=["Age", "autorater_prediction", "H"]).copy()
    use["Age"] = use["Age"].astype(float)
    use["H"] = use["H"].astype(int)

    logging.info("Final matches: %d / %d", len(use), len(pred))

    # Scatter
    save_scatter_pred_vs_age(use, out_dir / "fig_pred_vs_age.png")

    # Overall ROC
    auc_overall = plot_and_save_roc(
        use["H"].values,
        use["autorater_prediction"].values,
        title="ROC (All ages)",
        savepath=out_dir / "fig_roc_overall.png",
    )

    # Single band ROC (optional)
    auc_band = np.nan
    if band is not None:
        low, high = band
        band_df = use[(use["Age"] >= low) & (use["Age"] < high)].copy()
        if len(band_df) >= 10 and band_df["H"].nunique() == 2:
            auc_band = plot_and_save_roc(
                band_df["H"].values,
                band_df["autorater_prediction"].values,
                title=f"ROC (Age {int(low)}–{int(high) - 1})",
                savepath=out_dir / f"fig_roc_{int(low)}_{int(high) - 1}.png",
            )

    # Age bins & per-bin metrics
    use["age_bin"] = pd.cut(
        use["Age"],
        bins=age_bins,
        labels=age_labels,
        right=False,
        include_lowest=True,
    )

    rows = []
    for b in age_labels:
        sub = use[use["age_bin"] == b]
        if len(sub) < 20 or sub["H"].nunique() < 2:
            continue
        y = sub["H"].values
        p = sub["autorater_prediction"].values

        # ROC/AUC per bin
        fpr, tpr, _ = roc_curve(y, p)
        auc_b = auc(fpr, tpr)
        plt.figure(figsize=(5.6, 5.6))
        plt.plot(fpr, tpr, lw=1.8, label=f"AUC = {auc_b:.3f}")
        plt.plot([0, 1], [0, 1], "k--", lw=1)
        plt.xlabel("False Positive Rate")
        plt.ylabel("True Positive Rate")
        plt.title(f"ROC (Age {b})")
        plt.legend(loc="lower right")
        plt.grid(alpha=0.25, linestyle="--")
        plt.tight_layout()
        plt.savefig(out_dir / f"fig_roc_{str(b).replace('-', '_')}.png", dpi=200)
        plt.close()

        # Threshold @ 0.5
        yhat = (p >= 0.5).astype(int)
        acc = accuracy_score(y, yhat)
        tn, fp, fn, tp = confusion_matrix(y, yhat, labels=[0, 1]).ravel()
        tpr_ = tp / (tp + fn) if (tp + fn) else np.nan
        tnr_ = tn / (tn + fp) if (tn + fp) else np.nan
        prev = y.mean()

        rows.append(
            {
                "age_bin": b,
                "n": len(sub),
                "prevalence_AD": prev,
                "AUC": auc_b,
                "ACC@0.5": acc,
                "TPR@0.5": tpr_,
                "TNR@0.5": tnr_,
            }
        )

    perf = pd.DataFrame(rows).sort_values("age_bin")
    perf.to_csv(out_dir / "metrics_by_age.csv", index=False, encoding="utf-8-sig")

    # Overlay ROC by age bins
    plt.figure(figsize=(7.6, 6))
    for b in age_labels:
        sub = use[use["age_bin"] == b]
        if len(sub) < 20 or sub["H"].nunique() < 2:
            continue
        fpr, tpr, _ = roc_curve(sub["H"].values, sub["autorater_prediction"].values)
        roc_auc = auc(fpr, tpr)
        plt.plot(fpr, tpr, lw=1.8, label=f"{b} (AUC {roc_auc:.2f})")
    plt.plot([0, 1], [0, 1], "k--", lw=1)
    plt.xlabel("False Positive Rate")
    plt.ylabel("True Positive Rate")
    plt.title("ROC by Age Bins")
    plt.legend(loc="lower right", fontsize=9)
    plt.grid(alpha=0.25)
    plt.tight_layout()
    plt.savefig(out_dir / "fig_roc_by_age_bins.png", dpi=200)
    plt.close()

    # AUC 95% CI (overall & per-bin)
    a0, lo, hi = bootstrap_auc_ci(
        use["H"].values, use["autorater_prediction"].values, B=bootstrap_B, rng=rng
    )
    logging.info("Overall AUC = %.3f  (95%% CI %.3f–%.3f)", a0, lo, hi)

    ci_rows = []
    for b in age_labels:
        sub = use[use["age_bin"] == b]
        if len(sub) >= 20 and sub["H"].nunique() == 2:
            a_bin, l, h = bootstrap_auc_ci(
                sub["H"].values, sub["autorater_prediction"].values, B=bootstrap_B, rng=rng
            )
        else:
            a_bin, l, h = (np.nan, np.nan, np.nan)
        ci_rows.append({"age_bin": b, "AUC": a_bin, "AUC_CI_low": l, "AUC_CI_high": h})

    perf_ci = perf.merge(pd.DataFrame(ci_rows), on="age_bin", how="left")
    perf_ci.to_csv(out_dir / "metrics_by_age_with_auc_ci.csv", index=False, encoding="utf-8-sig")

    # Youden thresholds per bin
    rows_thr = []
    for b in age_labels:
        sub = use[use["age_bin"] == b]
        if len(sub) < 20 or sub["H"].nunique() < 2:
            continue
        y = sub["H"].values
        p = sub["autorater_prediction"].values
        thr, tpr_star, tnr_star = youden_threshold(y, p)
        yhat_star = (p >= thr).astype(int)
        acc_star = accuracy_score(y, yhat_star)
        rows_thr.append(
            {
                "age_bin": b,
                "thr_youden": float(thr),
                "ACC@thr": acc_star,
                "TPR@thr": float(tpr_star),
                "TNR@thr": float(tnr_star),
            }
        )
    thr_df = pd.DataFrame(rows_thr).sort_values("age_bin")
    thr_df.to_csv(out_dir / "metrics_by_age_youden.csv", index=False, encoding="utf-8-sig")

    # Calibration curves & Brier
    cal_rows = []
    for b in age_labels:
        sub = use[use["age_bin"] == b]
        if len(sub) < calib_min_n or sub["H"].nunique() < 2:
            continue
        y = sub["H"].values
        p = sub["autorater_prediction"].values
        frac_pos, mean_pred = calibration_curve(y, p, n_bins=10, strategy="uniform")
        bs = brier_score_loss(y, p)

        plt.figure(figsize=(5.2, 5))
        plt.plot(mean_pred, frac_pos, "o-", label="Observed")
        plt.plot([0, 1], [0, 1], "k--", label="Ideal")
        plt.xlabel("Predicted probability")
        plt.ylabel("Observed fraction (H=1)")
        plt.title(f"Calibration (Age {b})  Brier={bs:.3f}")
        plt.legend()
        plt.grid(alpha=0.25)
        plt.tight_layout()
        plt.savefig(cal_dir / f"calibration_{str(b).replace('-', '_')}.png", dpi=200)
        plt.close()

        cal_rows.append({"age_bin": b, "Brier": bs})
    pd.DataFrame(cal_rows).to_csv(out_dir / "calibration_by_age.csv", index=False, encoding="utf-8-sig")

    # Summary log
    logging.info("=== SUMMARY ===")
    logging.info("Overall AUC: %.3f", auc_overall)
    if not np.isnan(auc_band):
        logging.info("AUC (Band %s–%s): %.3f", str(band[0]), str(band[1] - 1), auc_band)
    else:
        logging.info("AUC (Band %s–%s): NA (insufficient data)", str(band[0]), str(band[1] - 1))

    logging.info("Saved artifacts under: %s", out_dir.resolve())


# ---------------------------------------------------------------------
# Utilities
# ---------------------------------------------------------------------
def parse_bins_labels(bins_str: str, labels_str: str) -> Tuple[List[float], List[str]]:
    bins = [float(x) for x in bins_str.split(",")]
    labels = [x.strip() for x in labels_str.split(",")]
    if len(labels) != (len(bins) - 1):
        raise ValueError(
            f"age-labels must have len(bins)-1 entries. Got {len(labels)} labels for {len(bins)} bin edges."
        )
    return bins, labels


def parse_band(band_str: str) -> Tuple[float, float]:
    if ":" not in band_str:
        raise ValueError("--band must be formatted as 'low:high' (e.g., 74:80)")
    low, high = band_str.split(":")
    return float(low), float(high)


# ---------------------------------------------------------------------
# Entry
# ---------------------------------------------------------------------
def main() -> None:
    args = parse_args()
    setup_logging(args.log)

    age_bins, age_labels = parse_bins_labels(args.age_bins, args.age_labels)
    band = parse_band(args.band) if args.band else None

    run_analysis(
        meta_csv=args.meta,
        pred_csv=args.pred,
        out_dir=args.out,
        age_bins=age_bins,
        age_labels=age_labels,
        band=band,
        date_tol_days=args.date_tol_days,
        bootstrap_B=args.bootstrap,
        calib_min_n=args.calib_min_n,
        seed=args.seed,
    )


if __name__ == "__main__":
    main()
