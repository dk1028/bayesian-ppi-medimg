#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
age_analysis.py

Highlights
----------
- Paths are provided via CLI (no hard‑coded local/Colab paths)
- Deterministic RNG for reproducibility (seeded)
- Exact + nearest‑date (±k days) metadata matching
- Scatter plot (P(AD) vs Age), overall ROC, per‑bin ROC overlay
- Per‑bin metrics (@0.5 threshold): ACC, TPR, TNR, prevalence
- AUC 95% bootstrap CIs (overall & per bin)
- Youden J* optimal thresholds per bin (with metrics at J*)
- Optional pairwise AUC difference tests across age bins via permutation + Holm step‑down correction
- Headless plotting (Agg), logging, clear function boundaries and type hints

Usage
-----
python age_analysis.py \
  --meta data/processed/all_people_7_20_2025.csv \
  --pred data/processed/autorater_predictions_all4.csv \
  --out  figures/age_analysis \
  --age-bins 50,74,80,101 \
  --age-labels 50-73,74-79,80-100 \
  --band 74:80 \
  --date-tol-days 14 \
  --bootstrap 2000 \
  --calib-min-n 50 \
  --seed 2025 \
  --perm-tests --perm-B 5000
"""
from __future__ import annotations

import argparse
import logging
import re
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
    brier_score_loss,
    confusion_matrix,
    roc_auc_score,
    roc_curve,
)


# ---------------------------------------------------------------------
# Logging
# ---------------------------------------------------------------------

def setup_logging(level: str = "INFO") -> None:
    """Configure root logger."""
    log_level = getattr(logging, level.upper(), logging.INFO)
    logging.basicConfig(level=log_level, format="[%(levelname)s] %(message)s")


# ---------------------------------------------------------------------
# IO & Parsing
# ---------------------------------------------------------------------

REQUIRED_META = ["Subject", "Age", "Sex", "Acq Date"]
REQUIRED_PRED = ["subject_id", "Acq_Date", "autorater_prediction", "H"]


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description=(
            "Autorater vs Age analysis with ROC/metrics/calibration by age bins "
            "and optional pairwise AUC permutation tests."
        )
    )
    p.add_argument("--meta", type=Path, required=True, help="Metadata CSV (Subject, Age, Sex, Acq Date).")
    p.add_argument(
        "--pred",
        type=Path,
        required=True,
        help="Predictions CSV (subject_id, Acq_Date, autorater_prediction, H[, label]).",
    )
    p.add_argument("--out", type=Path, required=True, help="Output directory for figures/CSVs.")
    p.add_argument(
        "--age-bins",
        type=str,
        default="50,74,80,101",
        help="Inclusive-left, exclusive-right bin edges, comma-separated (e.g., 50,74,80,101).",
    )
    p.add_argument(
        "--age-labels",
        type=str,
        default="50-73,74-79,80-100",
        help="Comma-separated labels matching bins-1 (e.g., 50-73,74-79,80-100).",
    )
    p.add_argument("--band", type=str, default="74:80", help="Optional single-band ROC as 'low:high' (e.g., 74:80).")
    p.add_argument("--date-tol-days", type=int, default=14, help="Tolerance for nearest-date matching.")
    p.add_argument("--bootstrap", type=int, default=2000, help="Bootstrap repetitions for AUC CI.")
    p.add_argument("--calib-min-n", type=int, default=50, help="Min per-bin n to draw calibration curves.")
    p.add_argument("--seed", type=int, default=2025, help="RNG seed for reproducibility.")
    p.add_argument("--log", type=str, default="INFO", help="Logging level (DEBUG, INFO, WARNING, ERROR).")
    # Optional pairwise tests
    p.add_argument("--perm-tests", action="store_true", help="Run pairwise AUC permutation tests across age bins.")
    p.add_argument("--perm-B", type=int, default=5000, help="Permutation repetitions for pairwise AUC tests.")
    return p.parse_args()


def ensure_required_columns(df: pd.DataFrame, req: Sequence[str], tag: str) -> None:
    missing = [c for c in req if c not in df.columns]
    if missing:
        raise ValueError(f"[{tag}] Missing columns {missing}. Current columns: {list(df.columns)}")


def parse_date_series(s: pd.Series) -> pd.Series:
    """Robust date parsing returning a normalized datetime64[ns] series (midnight)."""
    out = pd.to_datetime(s, errors="coerce", infer_datetime_format=True)
    mask = out.isna()
    if mask.any():
        try_formats = ["%m/%d/%Y", "%m/%d/%y", "%Y-%m-%d", "%Y/%m/%d", "%m-%d-%Y", "%d/%m/%Y"]
        for fmt in try_formats:
            tmp = pd.to_datetime(s[mask], format=fmt, errors="coerce")
            out.loc[mask] = tmp
            mask = out.isna()
            if not mask.any():
                break
    return out.dt.normalize()


def coerce_binary(series: pd.Series) -> pd.Series:
    """
    Coerce to binary 0/1 if possible; accepts strings '0'/'1', numeric, or
    common labels via mapping {CN->0, AD->1, ...}.
    """
    x = pd.to_numeric(series, errors="coerce")
    if x.isna().mean() > 0.5:
        s = series.astype(str).str.upper().str.strip()
        mapping = {"CN": 0, "CONTROL": 0, "NEG": 0, "NO": 0, "AD": 1, "POS": 1, "YES": 1}
        x = s.map(mapping)
    return x.astype("Int64")


# ---------------------------------------------------------------------
# Matching
# ---------------------------------------------------------------------

def nearest_date_match(pred: pd.DataFrame, meta: pd.DataFrame, date_tol_days: int) -> pd.DataFrame:
    """Exact join on (subject_id, AcqDate_std); fallback to nearest date within ±date_tol_days."""
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
            sid: df[["AcqDate_std", "Age", "Sex"]].dropna(subset=["AcqDate_std"]).sort_values("AcqDate_std")
            for sid, df in meta.groupby("Subject", sort=False)
        }

        ages: List[float] = []
        sexes: List[str] = []
        for _, row in merged.loc[need_fill].iterrows():
            sid = row["subject_id"]
            d0 = row["AcqDate_std"]
            age_val, sex_val = np.nan, np.nan
            if pd.notna(d0) and sid in meta_grp and len(meta_grp[sid]):
                cand = meta_grp[sid]
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

def save_scatter_pred_vs_age(df: pd.DataFrame, out_path: Path) -> None:
    plt.figure(figsize=(8.5, 5.2))
    colors = np.where(df["H"] == 1, "crimson", "royalblue")
    plt.scatter(df["Age"].values, df["autorater_prediction"].values, c=colors, s=16, alpha=0.55, edgecolors="none")
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


def plot_and_save_roc(y_true: np.ndarray, y_score: np.ndarray, title: str, savepath: Path) -> float:
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

def bootstrap_auc_ci(y: np.ndarray, p: np.ndarray, B: int, rng: np.random.Generator) -> Tuple[float, float, float]:
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
# Pairwise AUC permutation tests (independent bins)
# ---------------------------------------------------------------------

def _auc_np(y: np.ndarray, p: np.ndarray) -> float:
    if len(np.unique(y)) < 2:
        return float("nan")
    return float(roc_auc_score(y, p))


def perm_test_auc_diff(df: pd.DataFrame, bin1: str, bin2: str, B: int, rng: np.random.Generator) -> Tuple[float, float]:
    """Two‑sided permutation test for AUC(bin1) - AUC(bin2).

    Null: group labels are exchangeable (independent bins). We permute bin labels
    while preserving (y, p) pairs and group sizes.
    """
    g1 = df[df["age_bin"] == bin1]
    g2 = df[df["age_bin"] == bin2]
    if (len(g1) < 20 or len(g2) < 20 or g1["H"].nunique() < 2 or g2["H"].nunique() < 2):
        return float("nan"), float("nan")

    y1, p1 = g1["H"].to_numpy(), g1["autorater_prediction"].to_numpy()
    y2, p2 = g2["H"].to_numpy(), g2["autorater_prediction"].to_numpy()

    auc1 = _auc_np(y1, p1)
    auc2 = _auc_np(y2, p2)
    if np.isnan(auc1) or np.isnan(auc2):
        return float("nan"), float("nan")

    obs_diff = auc1 - auc2

    # pool then permute indices; split by n1/n2 each draw
    y_all = np.concatenate([y1, y2])
    p_all = np.concatenate([p1, p2])
    n1 = len(y1)
    n_all = len(y_all)

    diffs: List[float] = []
    for _ in range(B):
        idx = rng.permutation(n_all)
        idx1 = idx[:n1]
        idx2 = idx[n1:]
        a1 = _auc_np(y_all[idx1], p_all[idx1])
        a2 = _auc_np(y_all[idx2], p_all[idx2])
        if np.isnan(a1) or np.isnan(a2):
            continue
        diffs.append(a1 - a2)

    if not diffs:
        return float(obs_diff), float("nan")

    diffs = np.asarray(diffs, dtype=float)
    pval = (np.sum(np.abs(diffs) >= np.abs(obs_diff)) + 1.0) / (len(diffs) + 1.0)
    return float(obs_diff), float(pval)


def holm_adjust(pvals: np.ndarray) -> np.ndarray:
    """Holm step‑down adjusted p‑values (returns array in original order)."""
    order = np.argsort(pvals)
    m = len(pvals)
    adj = np.zeros(m, dtype=float)
    max_so_far = 0.0
    for rank, idx in enumerate(order):
        factor = m - rank
        val = min(1.0, float(pvals[idx]) * factor)
        max_so_far = max(max_so_far, val)
        adj[idx] = max_so_far
    return adj


def compute_pairwise_auc_tests(use: pd.DataFrame, age_labels: Sequence[str], B: int, rng: np.random.Generator) -> pd.DataFrame:
    rows: List[dict] = []
    for i in range(len(age_labels)):
        for j in range(i + 1, len(age_labels)):
            b1, b2 = age_labels[i], age_labels[j]
            d, pval = perm_test_auc_diff(use, b1, b2, B=B, rng=rng)
            rows.append(
                {
                    "bin1": b1,
                    "bin2": b2,
                    "AUC_diff_bin1_minus_bin2": d,
                    "p_value_perm_two_sided": pval,
                    "method": "permutation (group-label shuffle), sizes fixed",
                }
            )
    pairwise_df = pd.DataFrame(rows)
    if len(pairwise_df):
        raw = pairwise_df["p_value_perm_two_sided"].to_numpy(dtype=float)
        mask = ~np.isnan(raw)
        adj = np.full_like(raw, np.nan, dtype=float)
        if mask.any():
            adj[mask] = holm_adjust(raw[mask])
        pairwise_df["p_value_holm"] = adj
    return pairwise_df


# ---------------------------------------------------------------------
# Main analysis
# ---------------------------------------------------------------------

def safe_label(s: str) -> str:
    """Make a filesystem‑friendly label for filenames (e.g., '50–73' -> '50_73')."""
    return re.sub(r"[^A-Za-z0-9]+", "_", str(s)).strip("_")


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
    do_perm_tests: bool,
    perm_B: int,
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
    meta["AcqDate_std"] = parse_date_series(meta["Acq Date"])  # datetime64[ns]
    pred["AcqDate_std"] = parse_date_series(pred["Acq_Date"])  # datetime64[ns]

    meta["Age"] = pd.to_numeric(meta["Age"], errors="coerce")
    pred["autorater_prediction"] = pd.to_numeric(pred["autorater_prediction"], errors="coerce")

    # H: prefer provided numeric; fallback via 'label' if needed
    H_series = coerce_binary(pred["H"])
    if H_series.isna().any() and "label" in pred.columns:
        logging.info("Detected NaNs in 'H'; attempting to derive from 'label' column.")
        derived = coerce_binary(pred["label"])
        H_series = H_series.fillna(derived)
    pred["H"] = H_series

    # Optional AD flag
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
    auc_band = float("nan")
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
    use["age_bin"] = pd.cut(use["Age"], bins=age_bins, labels=age_labels, right=False, include_lowest=True)

    rows: List[dict] = []
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
        plt.savefig(out_dir / f"fig_roc_{safe_label(b)}.png", dpi=200)
        plt.close()

        # Threshold @ 0.5
        yhat = (p >= 0.5).astype(int)
        acc = accuracy_score(y, yhat)
        tn, fp, fn, tp = confusion_matrix(y, yhat, labels=[0, 1]).ravel()
        tpr_ = tp / (tp + fn) if (tp + fn) else float("nan")
        tnr_ = tn / (tn + fp) if (tn + fp) else float("nan")
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
    a0, lo, hi = bootstrap_auc_ci(use["H"].values, use["autorater_prediction"].values, B=bootstrap_B, rng=rng)
    logging.info("Overall AUC = %.3f  (95%% CI %.3f–%.3f)", a0, lo, hi)

    ci_rows: List[dict] = []
    for b in age_labels:
        sub = use[use["age_bin"] == b]
        if len(sub) >= 20 and sub["H"].nunique() == 2:
            a_bin, l, h = bootstrap_auc_ci(sub["H"].values, sub["autorater_prediction"].values, B=bootstrap_B, rng=rng)
        else:
            a_bin, l, h = (float("nan"), float("nan"), float("nan"))
        ci_rows.append({"age_bin": b, "AUC": a_bin, "AUC_CI_low": l, "AUC_CI_high": h})

    perf_ci = perf.merge(pd.DataFrame(ci_rows), on="age_bin", how="left")
    perf_ci.to_csv(out_dir / "metrics_by_age_with_auc_ci.csv", index=False, encoding="utf-8-sig")

    # Youden thresholds per bin
    rows_thr: List[dict] = []
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
    cal_rows: List[dict] = []
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
        plt.savefig(cal_dir / f"calibration_{safe_label(b)}.png", dpi=200)
        plt.close()

        cal_rows.append({"age_bin": b, "Brier": bs})
    pd.DataFrame(cal_rows).to_csv(out_dir / "calibration_by_age.csv", index=False, encoding="utf-8-sig")

    # Optional permutation tests across age bins
    if do_perm_tests:
        pairwise_df = compute_pairwise_auc_tests(use, age_labels, B=perm_B, rng=rng)
        pairwise_df.to_csv(out_dir / "pairwise_auc_perm_test.csv", index=False, encoding="utf-8-sig")
        logging.info("Pairwise AUC permutation tests saved: %s", (out_dir / "pairwise_auc_perm_test.csv").resolve())

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
        do_perm_tests=args.perm_tests,
        perm_B=args.perm_B,
    )


if __name__ == "__main__":
    main()
