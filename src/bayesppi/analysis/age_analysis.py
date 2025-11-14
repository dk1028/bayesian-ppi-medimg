#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
age_analysis.py — Autorater vs Age analysis (bins: 50–73, 74–79, 80–100)

Repository/TMLR-friendly (no Colab-specific code):
- CLI arguments for all inputs/outputs (no hard-coded Google Drive paths)
- Headless plotting via matplotlib Agg backend (saves figures to files)
- Exact-date merge with ±T-day nearest-date fallback (default T=14)
- Metrics per bin (n, prevalence, AUC, ACC/TPR/TNR @0.5)
- AUC 95% bootstrap CI (overall + per-bin)
- Pairwise AUC differences via permutation test + Holm adjustment
- Calibration curves + Brier score (when sample size is sufficient)
- Add-ons (saved under --out2):
    1) Frequentist PPI (analytic) vs CRE (Beta) — real-data & simulation
    2) Out-of-fold thresholding (Youden) vs leaky full-sample threshold
    3) Exchangeability diagnostic (L vs U separability) + IW weights prep
    4) Threshold uncertainty propagation (bootstrap)

Example usage
-------------
python age_analysis.py \
  --meta-csv data/processed/all_people_7_20_2025.csv \
  --pred-csv data/processed/autorater_predictions_all4.csv \
  --out figures/age_analysis5 \
  --out2 figures/age_analysis6

Reproducibility notes
---------------------
- Uses a fixed RNG (default seed=2025) unless overridden.
- Plots are deterministic up to sampling-based procedures (bootstrap/permutation).
"""
from __future__ import annotations

import argparse
import logging
from pathlib import Path
from typing import Iterable, List, Optional, Sequence, Tuple

import matplotlib
matplotlib.use("Agg")  # headless for CI/docs/servers
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from sklearn.metrics import (
    auc,
    accuracy_score,
    brier_score_loss,
    confusion_matrix,
    roc_auc_score,
    roc_curve,
)
from sklearn.calibration import calibration_curve
from sklearn.model_selection import KFold
from sklearn.linear_model import LogisticRegression


# ---------------------------------------------------------------------
# CLI & logging
# ---------------------------------------------------------------------

def setup_logging(level: str = "INFO") -> None:
    logging.basicConfig(
        level=getattr(logging, level.upper(), logging.INFO),
        format="[%(levelname)s] %(message)s",
    )


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Autorater vs Age analysis with add-ons.")
    p.add_argument("--meta-csv", type=Path, required=True,
                   help="Metadata CSV with columns: Subject, Age, Sex, Acq Date (varied formats ok)")
    p.add_argument("--pred-csv", type=Path, required=True,
                   help="Predictions CSV with columns: subject_id, Acq_Date, autorater_prediction, H[, label]")

    p.add_argument("--out", type=Path, required=True,
                   help="Output directory for core figures/CSVs.")
    p.add_argument("--out2", type=Path, default=None,
                   help="Output directory for add-ons; default: <out parent>/age_analysis6")

    p.add_argument("--age-bins", type=float, nargs="+", default=[50, 74, 80, 101],
                   help="Age bin edges (right-open). Default: 50 74 80 101")
    p.add_argument("--age-labels", type=str, nargs="+", default=["50–73", "74–79", "80–100"],
                   help="Labels for age bins. Must have len = len(age-bins)-1")
    p.add_argument("--band", type=str, default="74,80",
                   help="Standalone ROC band as 'low,high' (right-open). Default: 74,80")

    p.add_argument("--date-tol-days", type=int, default=14,
                   help="Tolerance (days) for nearest-date fallback matching. Default: 14")
    p.add_argument("--boot-B", type=int, default=2000,
                   help="Bootstrap repetitions for AUC CI. Default: 2000")
    p.add_argument("--perm-B", type=int, default=5000,
                   help="Permutation repetitions for AUC diff tests. Default: 5000")
    p.add_argument("--calib-min-n", type=int, default=50,
                   help="Min per-bin sample size to draw calibration curves. Default: 50")

    p.add_argument("--rng-seed", type=int, default=2025, help="RNG seed. Default: 2025")
    p.add_argument("--oof-k", type=int, default=5, help="K-fold for OOF thresholding. Default: 5")
    p.add_argument("--log", type=str, default="INFO", help="Logging level")
    return p.parse_args()


# ---------------------------------------------------------------------
# Utilities
# ---------------------------------------------------------------------

def parse_date_series(s: pd.Series) -> pd.Series:
    out = pd.to_datetime(s, errors="coerce")
    mask = out.isna()
    if mask.any():
        try_formats = ["%m/%d/%Y", "%m/%d/%y", "%Y-%m-%d", "%Y/%m/%d", "%m-%d-%Y", "%d/%m/%Y"]
        for fmt in try_formats:
            out2 = pd.to_datetime(s[mask], format=fmt, errors="coerce")
            out.loc[mask] = out2
            mask = out.isna()
            if not mask.any():
                break
    return out.dt.date


def ensure_dir(p: Path) -> Path:
    p.mkdir(parents=True, exist_ok=True)
    return p


# ---------------------------------------------------------------------
# Core computations
# ---------------------------------------------------------------------

def plot_roc(y_true: np.ndarray, y_score: np.ndarray, title: str, savepath: Path) -> float:
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
    return float(roc_auc)


def auc_ci(y: Sequence[int], p: Sequence[float], B: int, rng: np.random.Generator) -> Tuple[float, float, float]:
    y = np.asarray(y)
    p = np.asarray(p)
    auc0 = roc_auc_score(y, p)
    boots: List[float] = []
    n = len(y)
    for _ in range(B):
        idx = rng.integers(0, n, size=n)
        if len(np.unique(y[idx])) < 2:
            continue
        boots.append(roc_auc_score(y[idx], p[idx]))
    if boots:
        lo, hi = np.percentile(boots, [2.5, 97.5])
    else:
        lo, hi = (np.nan, np.nan)
    return float(auc0), float(lo), float(hi)


def youden_threshold(y: Sequence[int], p: Sequence[float]) -> Tuple[float, float, float]:
    fpr, tpr, thr = roc_curve(y, p)
    j = tpr - fpr
    k = int(np.argmax(j))
    return float(thr[k]), float(tpr[k]), float(1 - fpr[k])


def holm_adjust(pvals: np.ndarray, alpha: float = 0.05) -> np.ndarray:
    order = np.argsort(pvals)
    m = len(pvals)
    adj = np.zeros(m, dtype=float)
    max_so_far = 0.0
    for rank, idx in enumerate(order):
        factor = m - rank
        val = min(1.0, pvals[idx] * factor)
        max_so_far = max(max_so_far, val)
        adj[idx] = max_so_far
    return adj


# ---------------------------------------------------------------------
# Add-ons: PPI vs CRE (analytic Beta), OOF threshold, Exchangeability, Bootstrap
# ---------------------------------------------------------------------
Z975 = 1.959963984540054


def ppi_analytic_estimator(sub_idx: np.ndarray, df_all: pd.DataFrame) -> Tuple[float, Tuple[float, float]]:
    """Frequentist PPI (difference estimator) with normal CI.
    g_hat = mean(A) + mean(H-A on labeled),
    Var = Var(A)/N + Var(H-A)/n.
    """
    A = df_all["A_class"].astype(float).to_numpy()
    N = len(A)
    A_bar = float(A.mean())
    varA = float(A.var(ddof=1)) if N > 1 else 0.0

    sub = df_all.loc[sub_idx]
    R = (sub["H"].astype(float) - sub["A_class"].astype(float)).to_numpy()
    n = len(R)
    r_bar = float(R.mean()) if n > 0 else 0.0
    varR = float(R.var(ddof=1)) if n > 1 else 0.0

    ghat = A_bar + r_bar
    se = (varA / N + varR / n) ** 0.5 if (N > 0 and n > 0) else 0.0
    return float(ghat), (float(ghat - Z975 * se), float(ghat + Z975 * se))


def cre_beta_estimator(sub_idx: np.ndarray, df_all: pd.DataFrame, rng: np.random.Generator,
                       alpha: float = 1.0, beta: float = 1.0, draws: int = 4000) -> Tuple[float, Tuple[float, float]]:
    """CRE via independent Beta posteriors (conjugate approximation) + sampling for CI."""
    A_all = df_all["A_class"].astype(int).to_numpy()
    N = len(A_all)
    NA1 = int(A_all.sum())

    sub = df_all.loc[sub_idx]
    n1 = int((sub["A_class"] == 1).sum())
    H1 = int(sub.loc[sub["A_class"] == 1, "H"].sum())
    n0 = int((sub["A_class"] == 0).sum())
    H0 = int(sub.loc[sub["A_class"] == 0, "H"].sum())

    aA, bA = alpha + NA1, beta + (N - NA1)
    aH1, bH1 = alpha + H1, beta + (n1 - H1)
    aH0, bH0 = alpha + H0, beta + (n0 - H0)

    thetaA = rng.beta(aA, bA, size=draws)
    thetaH1 = rng.beta(aH1, bH1, size=draws)
    thetaH0 = rng.beta(aH0, bH0, size=draws)
    g = thetaA * thetaH1 + (1 - thetaA) * thetaH0

    return float(np.mean(g)), (float(np.quantile(g, 0.025)), float(np.quantile(g, 0.975)))


def oof_threshold_metrics(df_in: pd.DataFrame, K: int, out_dir: Path) -> Tuple[pd.DataFrame, pd.DataFrame]:
    """K-fold: choose Youden threshold on train; evaluate on held-out."""
    kf = KFold(n_splits=K, shuffle=True, random_state=2025)
    rows = []
    for fold, (tr, te) in enumerate(kf.split(df_in), 1):
        trdf = df_in.iloc[tr]
        tedf = df_in.iloc[te]
        ytr, ptr = trdf["H"].to_numpy(), trdf["autorater_prediction"].to_numpy()
        yte, pte = tedf["H"].to_numpy(), tedf["autorater_prediction"].to_numpy()
        thr, _, _ = youden_threshold(ytr, ptr)
        yhat = (pte >= thr).astype(int)
        acc = accuracy_score(yte, yhat)
        tn, fp, fn, tp = confusion_matrix(yte, yhat, labels=[0, 1]).ravel()
        tpr = float(tp / (tp + fn)) if (tp + fn) else float("nan")
        tnr = float(tn / (tn + fp)) if (tn + fp) else float("nan")
        rows.append({"fold": fold, "thr_train": float(thr), "ACC": acc, "TPR": tpr, "TNR": tnr})

    oof = pd.DataFrame(rows)
    oof.to_csv(out_dir / "age6_oof_threshold_metrics.csv", index=False, encoding="utf-8-sig")

    # Leaky baseline (thr chosen on full data, evaluate on same)
    y, p = df_in["H"].to_numpy(), df_in["autorater_prediction"].to_numpy()
    thr_full, _, _ = youden_threshold(y, p)
    yhat_full = (p >= thr_full).astype(int)
    acc_full = accuracy_score(y, yhat_full)
    tn, fp, fn, tp = confusion_matrix(y, yhat_full, labels=[0, 1]).ravel()
    tpr_full = float(tp / (tp + fn)) if (tp + fn) else float("nan")
    tnr_full = float(tn / (tn + fp)) if (tn + fp) else float("nan")
    leak_row = pd.DataFrame([
        {"fold": "leaky_full", "thr_train": float(thr_full), "ACC": acc_full, "TPR": tpr_full, "TNR": tnr_full}
    ])
    leak_row.to_csv(out_dir / "age6_oof_threshold_metrics_leaky.csv", index=False, encoding="utf-8-sig")
    return oof, leak_row


def exchangeability_and_iw(df_in: pd.DataFrame, out_dir: Path, rng: np.random.Generator,
                            R: int = 50, label_frac: float = 0.3) -> Tuple[pd.DataFrame, Optional[pd.DataFrame]]:
    """Randomly mark a labeled subset (~label_frac); test separability L vs U (propensity AUC),
    and compute importance weights for labeled examples (last run)."""
    N = len(df_in)
    n_lab = max(20, int(label_frac * N))
    aucs: List[float] = []
    last_weights: Optional[pd.DataFrame] = None

    for _ in range(R):
        lab_idx = rng.choice(N, size=n_lab, replace=False)
        L = np.zeros(N, dtype=int)
        L[lab_idx] = 1

        # Simple feature set; extend as needed
        sex_enc = (
            df_in["Sex"].astype(str).str.upper().map({"M": 1, "MALE": 1, "F": 0, "FEMALE": 0}).fillna(0)
        ).astype(float)
        X = pd.DataFrame({
            "p": df_in["autorater_prediction"].to_numpy(),
            "age": df_in["Age"].to_numpy(),
            "sex": sex_enc.to_numpy(),
        }).to_numpy()
        yL = L

        clf = LogisticRegression(max_iter=1000)
        clf.fit(X, yL)
        pL = clf.predict_proba(X)[:, 1]
        aucL = roc_auc_score(yL, pL)
        aucs.append(float(aucL))

        # IW for labeled: w = (1-s)/s * p(x)/(1-p(x)), s = n_lab/N
        s = n_lab / N
        w = ((1 - s) / s) * (pL / (1 - pL + 1e-9))
        w_lab = w[lab_idx]
        last_weights = pd.DataFrame({
            "idx": df_in.index[lab_idx].to_numpy(),
            "subject_id": df_in.iloc[lab_idx]["subject_id"].to_numpy(),
            "weight": w_lab,
        })

    aucs = np.asarray(aucs)
    lo, hi = np.quantile(aucs, [0.025, 97.5])
    diag = pd.DataFrame([
        {"R": R, "n_lab": n_lab, "N": N, "AUC_mean": float(np.mean(aucs)), "AUC_lo": float(lo), "AUC_hi": float(hi)}
    ])
    diag.to_csv(out_dir / "age6_exchangeability_auc.csv", index=False, encoding="utf-8-sig")

    if last_weights is not None:
        last_weights.to_csv(out_dir / "age6_iw_weights_labeled.csv", index=False, encoding="utf-8-sig")

    # Propensity overlap plot (last run)
    plt.figure(figsize=(6, 4))
    plt.hist(pL[yL == 1], bins=20, alpha=0.6, label="Labeled", density=True)
    plt.hist(pL[yL == 0], bins=20, alpha=0.6, label="Unlabeled", density=True)
    plt.xlabel("Propensity p(L=1 | X)")
    plt.ylabel("Density")
    plt.title("Exchangeability diagnostic: propensity overlap")
    plt.legend()
    plt.tight_layout()
    plt.savefig(out_dir / "age6_propensity_hist.png", dpi=200)
    plt.close()

    return diag, last_weights


def threshold_bootstrap(df_in: pd.DataFrame, out_dir: Path, rng: np.random.Generator, B: int = 2000) -> Tuple[pd.DataFrame, pd.DataFrame]:
    """Bootstrap: re-fit Youden threshold each replicate; compute metric distributions.
    Note: evaluation is on the original sample; adapt if you prefer bootstrap-eval.
    """
    y = df_in["H"].to_numpy()
    p = df_in["autorater_prediction"].to_numpy()
    N = len(df_in)
    accs: List[float] = []
    tprs: List[float] = []
    tnrs: List[float] = []
    thrs: List[float] = []
    for _ in range(B):
        idx = rng.integers(0, N, size=N)
        yb, pb = y[idx], p[idx]
        thr_b, _, _ = youden_threshold(yb, pb)
        yhat_b = (p >= thr_b).astype(int)
        accs.append(accuracy_score(y, yhat_b))
        tn, fp, fn, tp = confusion_matrix(y, yhat_b, labels=[0, 1]).ravel()
        tprs.append(float(tp / (tp + fn)) if (tp + fn) else float("nan"))
        tnrs.append(float(tn / (tn + fp)) if (tn + fp) else float("nan"))
        thrs.append(float(thr_b))

    out = pd.DataFrame({"thr": thrs, "ACC": accs, "TPR": tprs, "TNR": tnrs})
    out.to_csv(out_dir / "age6_thr_bootstrap_samples.csv", index=False, encoding="utf-8-sig")

    summ = pd.DataFrame([
        {
            "thr_mean": float(np.nanmean(thrs)),
            "thr_lo": float(np.nanpercentile(thrs, 2.5)),
            "thr_hi": float(np.nanpercentile(thrs, 97.5)),
            "ACC_mean": float(np.nanmean(accs)),
            "ACC_lo": float(np.nanpercentile(accs, 2.5)),
            "ACC_hi": float(np.nanpercentile(accs, 97.5)),
            "TPR_mean": float(np.nanmean(tprs)),
            "TPR_lo": float(np.nanpercentile(tprs, 2.5)),
            "TPR_hi": float(np.nanpercentile(tprs, 97.5)),
            "TNR_mean": float(np.nanmean(tnrs)),
            "TNR_lo": float(np.nanpercentile(tnrs, 2.5)),
            "TNR_hi": float(np.nanpercentile(tnrs, 97.5)),
        }
    ])
    summ.to_csv(out_dir / "age6_thr_bootstrap_summary.csv", index=False, encoding="utf-8-sig")

    # Threshold distribution plot
    plt.figure(figsize=(6, 4))
    plt.hist(thrs, bins=30, alpha=0.8, density=True)
    plt.xlabel("Youden threshold (bootstrap)")
    plt.ylabel("Density")
    plt.title("Threshold uncertainty (bootstrap)")
    plt.tight_layout()
    plt.savefig(out_dir / "age6_thr_bootstrap_hist.png", dpi=200)
    plt.close()

    return out, summ


# ---------------------------------------------------------------------
# Main workflow
# ---------------------------------------------------------------------

def main() -> None:
    args = parse_args()
    setup_logging(args.log)

    rng = np.random.default_rng(args.rng_seed)

    out_dir = ensure_dir(args.out)
    out_dir2 = ensure_dir(args.out2 if args.out2 is not None else (out_dir.parent / "age_analysis6"))

    # Validate age bins and labels
    if len(args.age_labels) != len(args.age_bins) - 1:
        raise ValueError("age-labels must have length = len(age-bins) - 1")

    # Load data
    logging.info("Reading: %s", args.meta_csv)
    meta = pd.read_csv(args.meta_csv, dtype=str)
    logging.info("Reading: %s", args.pred_csv)
    pred = pd.read_csv(args.pred_csv, dtype=str)

    # Trim whitespace in headers
    meta.columns = [c.strip() for c in meta.columns]
    pred.columns = [c.strip() for c in pred.columns]

    # Required columns
    for c in ["Subject", "Age", "Sex", "Acq Date"]:
        if c not in meta.columns:
            raise ValueError(f"[META] Missing column '{c}'. Current: {list(meta.columns)}")
    for c in ["subject_id", "Acq_Date", "autorater_prediction", "H"]:
        if c not in pred.columns:
            raise ValueError(f"[PRED] Missing column '{c}'. Current: {list(pred.columns)}")

    # Parse
    meta["Subject"] = meta["Subject"].str.strip()
    pred["subject_id"] = pred["subject_id"].str.strip()

    meta["AcqDate_std"] = parse_date_series(meta["Acq Date"])  # type: ignore
    pred["AcqDate_std"] = parse_date_series(pred["Acq_Date"])  # type: ignore

    meta["Age"] = pd.to_numeric(meta["Age"], errors="coerce")
    pred["autorater_prediction"] = pd.to_numeric(pred["autorater_prediction"], errors="coerce")
    pred["H"] = pd.to_numeric(pred["H"], errors="coerce").astype("Int64")

    if "label" in pred.columns:
        pred["is_AD"] = (pred["label"].astype(str).str.upper() == "AD").astype(int)

    # Exact merge then nearest-date fallback
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
        logging.info("Exact match failed rows: %d → nearest-date (±%d days)", int(need_fill.sum()), args.date_tol_days)
        meta_grp = {
            sid: df[["AcqDate_std", "Age", "Sex"]]
            .dropna(subset=["AcqDate_std"])  # type: ignore
            .sort_values("AcqDate_std")
            for sid, df in meta.groupby("Subject", sort=False)
        }
        ages: List[float] = []
        sexes: List[str] = []
        for _, row in merged.loc[need_fill].iterrows():
            sid = row["subject_id"]
            d0 = row["AcqDate_std"]
            age_val, sex_val = np.nan, np.nan
            if pd.notna(d0) and sid in meta_grp:
                cand = meta_grp[sid]
                diffs = cand["AcqDate_std"].apply(lambda d: abs(pd.to_datetime(d) - pd.to_datetime(d0))).dt.days
                j = diffs.idxmin() if len(diffs) else None
                if j is not None and diffs.loc[j] <= args.date_tol_days:
                    age_val = cand.loc[j, "Age"]
                    sex_val = cand.loc[j, "Sex"]
            ages.append(age_val)
            sexes.append(sex_val)
        merged.loc[need_fill, "Age"] = ages
        merged.loc[need_fill, "Sex"] = sexes

    use = merged.dropna(subset=["Age", "autorater_prediction", "H"]).copy()
    use["Age"] = use["Age"].astype(float)
    use["H"] = use["H"].astype(int)
    use["A_class"] = (use["autorater_prediction"] >= 0.5).astype(int)

    logging.info("Final matches: %d / %d", len(use), len(pred))

    # 1) Scatter: P(AD) vs Age
    plt.figure(figsize=(8.5, 5.2))
    colors = np.where(use["H"] == 1, "crimson", "royalblue")
    plt.scatter(use["Age"], use["autorater_prediction"], c=colors, s=16, alpha=0.55, edgecolors="none")
    plt.xlabel("Age (years)")
    plt.ylabel("Autorater predicted P(AD)")
    plt.title("Predicted AD Probability vs Age (colored by H)")
    from matplotlib.lines import Line2D

    legend_elems = [
        Line2D([0], [0], marker="o", color="w", label="CN (H=0)", markerfacecolor="royalblue", markersize=7),
        Line2D([0], [0], marker="o", color="w", label="AD (H=1)", markerfacecolor="crimson", markersize=7),
    ]
    plt.legend(handles=legend_elems, loc="lower right")
    plt.grid(alpha=0.25, linestyle="--")
    plt.tight_layout()
    plt.savefig(out_dir / "fig_pred_vs_age.png", dpi=200)
    plt.close()

    # 2) Overall & band ROC
    auc_overall = plot_roc(use["H"].to_numpy(), use["autorater_prediction"].to_numpy(),
                           "ROC (All ages: 50–100)", out_dir / "fig_roc_overall.png")
    low, high = map(float, args.band.split(","))
    band_df = use[(use["Age"] >= low) & (use["Age"] < high)].copy()
    if len(band_df) >= 10 and band_df["H"].nunique() == 2:
        auc_band = plot_roc(band_df["H"].to_numpy(), band_df["autorater_prediction"].to_numpy(),
                            f"ROC (Age {int(low)}–{int(high)-1})",
                            out_dir / f"fig_roc_{int(low)}_{int(high)-1}.png")
    else:
        auc_band = np.nan

    # 3) Binning & per-bin metrics
    use["age_bin"] = pd.cut(use["Age"], bins=args.age_bins, labels=args.age_labels, right=False, include_lowest=True)

    rows: List[dict] = []
    for b in args.age_labels:
        sub = use[use["age_bin"] == b]
        if len(sub) < 20 or sub["H"].nunique() < 2:
            continue
        y = sub["H"].to_numpy()
        p = sub["autorater_prediction"].to_numpy()
        auc_b = plot_roc(y, p, f"ROC (Age {b})", out_dir / f"fig_roc_{b.replace('–', '_')}.png")
        yhat = (p >= 0.5).astype(int)
        acc = accuracy_score(y, yhat)
        tn, fp, fn, tp = confusion_matrix(y, yhat, labels=[0, 1]).ravel()
        tpr = float(tp / (tp + fn)) if (tp + fn) else float("nan")
        tnr = float(tn / (tn + fp)) if (tn + fp) else float("nan")
        prev = float(y.mean())
        rows.append({
            "age_bin": b,
            "n": len(sub),
            "prevalence_AD": prev,
            "AUC": auc_b,
            "ACC@0.5": acc,
            "TPR@0.5": tpr,
            "TNR@0.5": tnr,
        })

    perf = pd.DataFrame(rows).sort_values("age_bin")
    perf.to_csv(out_dir / "metrics_by_age.csv", index=False, encoding="utf-8-sig")

    # Overlay ROC by bins
    plt.figure(figsize=(7.6, 6))
    for b in args.age_labels:
        sub = use[use["age_bin"] == b]
        if len(sub) < 20 or sub["H"].nunique() < 2:
            continue
        fpr, tpr, _ = roc_curve(sub["H"].to_numpy(), sub["autorater_prediction"].to_numpy())
        roc_auc = auc(fpr, tpr)
        plt.plot(fpr, tpr, lw=1.8, label=f"{b} (AUC {roc_auc:.2f})")
    plt.plot([0, 1], [0, 1], "k--", lw=1)
    plt.xlabel("False Positive Rate")
    plt.ylabel("True Positive Rate")
    plt.title("ROC by Age Bins (50–73, 74–79, 80–100)")
    plt.legend(loc="lower right", fontsize=9)
    plt.grid(alpha=0.25)
    plt.tight_layout()
    plt.savefig(out_dir / "fig_roc_by_age_bins.png", dpi=200)
    plt.close()

    # 4) AUC 95% CI (overall & per-bin)
    a0, lo, hi = auc_ci(use["H"], use["autorater_prediction"], B=args.boot_B, rng=rng)
    logging.info("Overall AUC = %.3f (95%% CI %.3f–%.3f)", a0, lo, hi)

    ci_rows: List[dict] = []
    for b in args.age_labels:
        sub = use[use["age_bin"] == b]
        if len(sub) >= 20 and sub["H"].nunique() == 2:
            a, l, h = auc_ci(sub["H"], sub["autorater_prediction"], B=args.boot_B, rng=rng)
        else:
            a, l, h = (np.nan, np.nan, np.nan)
        ci_rows.append({"age_bin": b, "AUC_CI_low": l, "AUC_CI_high": h})
    perf_ci = perf.merge(pd.DataFrame(ci_rows), on="age_bin", how="left")
    perf_ci.to_csv(out_dir / "metrics_by_age_with_auc_ci.csv", index=False, encoding="utf-8-sig")

    # 5) Pairwise AUC differences via permutation + Holm
    def _auc_np(y: np.ndarray, p: np.ndarray) -> float:
        if len(np.unique(y)) < 2:
            return float("nan")
        return float(roc_auc_score(y, p))

    def perm_test_auc_diff(df: pd.DataFrame, bin1: str, bin2: str, B: int) -> Tuple[float, float]:
        g1 = df[df["age_bin"] == bin1]
        g2 = df[df["age_bin"] == bin2]
        if (len(g1) < 20 or len(g2) < 20 or g1["H"].nunique() < 2 or g2["H"].nunique() < 2):
            return float("nan"), float("nan")
        y1, p1 = g1["H"].to_numpy(), g1["autorater_prediction"].to_numpy()
        y2, p2 = g2["H"].to_numpy(), g2["autorater_prediction"].to_numpy()
        auc1, auc2 = _auc_np(y1, p1), _auc_np(y2, p2)
        if np.isnan(auc1) or np.isnan(auc2):
            return float("nan"), float("nan")
        obs_diff = auc1 - auc2

        y_all = np.concatenate([y1, y2])
        p_all = np.concatenate([p1, p2])
        n1 = len(y1)
        n_all = len(y_all)
        diffs: List[float] = []
        for _ in range(B):
            idx = rng.permutation(n_all)
            idx1, idx2 = idx[:n1], idx[n1:]
            a1b, a2b = _auc_np(y_all[idx1], p_all[idx1]), _auc_np(y_all[idx2], p_all[idx2])
            if np.isnan(a1b) or np.isnan(a2b):
                continue
            diffs.append(a1b - a2b)
        if not diffs:
            return float(obs_diff), float("nan")
        diffs = np.asarray(diffs)
        pval = (np.sum(np.abs(diffs) >= np.abs(obs_diff)) + 1.0) / (len(diffs) + 1.0)
        return float(obs_diff), float(pval)

    pair_rows: List[dict] = []
    for i in range(len(args.age_labels)):
        for j in range(i + 1, len(args.age_labels)):
            b1, b2 = args.age_labels[i], args.age_labels[j]
            d, pval = perm_test_auc_diff(use, b1, b2, B=args.perm_B)
            pair_rows.append({
                "bin1": b1,
                "bin2": b2,
                "AUC_diff_bin1_minus_bin2": d,
                "p_value_perm_two_sided": pval,
                "method": "permutation (group-label shuffle), sizes fixed",
            })
    pairwise_df = pd.DataFrame(pair_rows)
    if len(pairwise_df):
        raw = pairwise_df["p_value_perm_two_sided"].to_numpy(dtype=float)
        if np.any(np.isnan(raw)):
            mask = ~np.isnan(raw)
            adj = np.full_like(raw, np.nan, dtype=float)
            adj[mask] = holm_adjust(raw[mask])
        else:
            adj = holm_adjust(raw)
        pairwise_df["p_value_holm"] = adj
    pairwise_df.to_csv(out_dir / "pairwise_auc_perm_test.csv", index=False, encoding="utf-8-sig")

    # 6) Youden optimal threshold per bin
    rows_thr: List[dict] = []
    for b in args.age_labels:
        sub = use[use["age_bin"] == b]
        if len(sub) < 20 or sub["H"].nunique() < 2:
            continue
        y = sub["H"].to_numpy()
        p = sub["autorater_prediction"].to_numpy()
        thr, tpr_star, tnr_star = youden_threshold(y, p)
        yhat_star = (p >= thr).astype(int)
        acc_star = accuracy_score(y, yhat_star)
        rows_thr.append({
            "age_bin": b,
            "thr_youden": float(thr),
            "ACC@thr": acc_star,
            "TPR@thr": float(tpr_star),
            "TNR@thr": float(tnr_star),
        })
    thr_df = pd.DataFrame(rows_thr).sort_values("age_bin")
    thr_df.to_csv(out_dir / "metrics_by_age_youden.csv", index=False, encoding="utf-8-sig")

    # 7) Calibration curves (per-bin)
    cal_dir = ensure_dir(out_dir / "calibration_curves")
    cal_rows: List[dict] = []
    for b in args.age_labels:
        sub = use[use["age_bin"] == b]
        if len(sub) < args.calib_min_n or sub["H"].nunique() < 2:
            continue
        y = sub["H"].to_numpy()
        p = sub["autorater_prediction"].to_numpy()
        frac_pos, mean_pred = calibration_curve(y, p, n_bins=10, strategy="uniform")
        bs = brier_score_loss(y, p)

        plt.figure(figsize=(5.2, 5))
        plt.plot(mean_pred, frac_pos, "o-", label="Observed")
        plt.plot([0, 1], [0, 1], "k--", label="Ideal")
        plt.xlabel("Predicted probability")
        plt.ylabel("Observed fraction of AD")
        plt.title(f"Calibration (Age {b})  Brier={bs:.3f}")
        plt.legend()
        plt.grid(alpha=0.25)
        plt.tight_layout()
        plt.savefig(cal_dir / f"calibration_{b.replace('–', '_')}.png", dpi=200)
        plt.close()

        cal_rows.append({"age_bin": b, "Brier": float(bs)})

    pd.DataFrame(cal_rows).to_csv(out_dir / "calibration_by_age.csv", index=False, encoding="utf-8-sig")

    # ------------------------------------------------------------------
    # Add-ons saved under out_dir2
    # ------------------------------------------------------------------
    # (i) Real-data prevalence estimates: PPI vs CRE (uniform & Jeffreys)
    rows_prev: List[dict] = []
    scopes = [("overall", use)] + [(f"age_{b}", use[use["age_bin"] == b]) for b in args.age_labels]
    for scope_name, df_scope in scopes:
        if len(df_scope) == 0:
            continue
        idx = df_scope.index.to_numpy()
        m_p, (l_p, h_p) = ppi_analytic_estimator(idx, use)
        m_cu, (l_cu, h_cu) = cre_beta_estimator(idx, use, rng=rng, alpha=1.0, beta=1.0)
        m_cj, (l_cj, h_cj) = cre_beta_estimator(idx, use, rng=rng, alpha=0.5, beta=0.5)
        rows_prev.append({
            "scope": scope_name,
            "N": len(df_scope),
            "ppi_hat": m_p, "ppi_lo": l_p, "ppi_hi": h_p,
            "cre_uniform_hat": m_cu, "cre_uniform_lo": l_cu, "cre_uniform_hi": h_cu,
            "cre_jeffreys_hat": m_cj, "cre_jeffreys_lo": l_cj, "cre_jeffreys_hi": h_cj,
        })
    realdata_est = pd.DataFrame(rows_prev)
    realdata_est.to_csv(out_dir2 / "age6_prevalence_estimators_realdata.csv", index=False, encoding="utf-8-sig")

    # (ii) Simulation on label budgets (treat full-data prevalence as 'truth')
    def simulate_ppi_vs_cre(nsim: int = 200, label_sizes: Sequence[int] = (10, 20, 40, 80)) -> pd.DataFrame:
        g_true = float(use["H"].mean())
        res: List[dict] = []
        for nh in label_sizes:
            if nh > len(use):
                continue
            for prior_name, a, b in [("uniform", 1.0, 1.0), ("jeffreys", 0.5, 0.5)]:
                cov = {"ppi": 0, "cre": 0}
                width = {"ppi": [], "cre": []}
                for _ in range(nsim):
                    idx = rng.choice(len(use), size=nh, replace=False)
                    m_p, (l_p, h_p) = ppi_analytic_estimator(idx, use)
                    cov["ppi"] += int(l_p <= g_true <= h_p)
                    width["ppi"].append(h_p - l_p)
                    m_c, (l_c, h_c) = cre_beta_estimator(idx, use, rng=rng, alpha=a, beta=b)
                    cov["cre"] += int(l_c <= g_true <= h_c)
                    width["cre"].append(h_c - l_c)
                res.append({
                    "n_labels": nh,
                    "prior": prior_name,
                    "ppi_cov": cov["ppi"] / nsim,
                    "ppi_w": float(np.mean(width["ppi"])) if width["ppi"] else float("nan"),
                    "cre_cov": cov["cre"] / nsim,
                    "cre_w": float(np.mean(width["cre"])) if width["cre"] else float("nan"),
                })
        out = pd.DataFrame(res)
        out.to_csv(out_dir2 / "age6_ppi_vs_cre_sim.csv", index=False, encoding="utf-8-sig")
        return out

    _sim_est = simulate_ppi_vs_cre()

    # (iii) OOF threshold vs leaky
    _oof_all, _leak_all = oof_threshold_metrics(use, K=args.oof_k, out_dir=out_dir2)

    # (iv) Exchangeability + IW; also writes propensity overlap plot
    _ex_diag, _iw_last = exchangeability_and_iw(use, out_dir=out_dir2, rng=rng)

    # (v) Threshold uncertainty bootstrap
    _thr_boot_samps, _thr_boot_summ = threshold_bootstrap(use, out_dir=out_dir2, rng=rng, B=2000)

    # Summary logs
    logging.info("\n=== SUMMARY ===")
    logging.info("Overall AUC (50–100): %.3f", auc_overall)
    if not np.isnan(auc_band):
        logging.info("AUC (Age %d–%d): %.3f", int(low), int(high) - 1, float(auc_band))
    else:
        logging.info("AUC (Age %d–%d): NA (insufficient data)", int(low), int(high) - 1)

    logging.info("Saved core outputs → %s", out_dir)
    logging.info("Saved add-on outputs → %s", out_dir2)


if __name__ == "__main__":
    main()
