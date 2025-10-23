#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
autorater.py

Render a side-by-side figure comparing an exemplar AD case and an exemplar CN case
from autorater predictions.

Repository/TMLR-friendly upgrades:
- No hard-coded paths (all via CLI flags)
- Headless plotting (matplotlib Agg), no interactive windows
- Robust CSV validation; derives H from label if needed
- Safe NIfTI loading with nibabel and slice normalization
- Deterministic, saved artifact (PNG/SVG/PDF)

Expected columns in the input CSV
---------------------------------
- autorater_prediction : float in [0,1] (can rename via --pred-col)
- nifti_path           : path to a .nii(.gz) file (can rename via --nifti-col)
- (either) H           : {0,1} numeric  (can rename via --h-col)
- (or)     label       : categorical with AD / CN (can rename via --label-col and tokens)

Examples
--------
# Basic usage (defaults assume common column names)
python autorater.py \
  --csv data/processed/autorater_predictions_all4.csv \
  --out figures/autorater_extremes.png

# If your CSV uses different column names
python autorater.py \
  --csv data/processed/preds.csv \
  --pred-col p_ad \
  --nifti-col path \
  --label-col diagnosis \
  --ad-token AD --cn-token CN \
  --out figures/autorater_extremes.svg

# If you already have a numeric H column
python autorater.py \
  --csv data/processed/preds.csv \
  --h-col H \
  --out figures/autorater_extremes.png
"""
from __future__ import annotations

import argparse
import logging
from pathlib import Path
from typing import Optional, Tuple

import matplotlib

# Headless backend for CI/docs/servers
matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import nibabel as nib  # noqa: E402
import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402


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
    p = argparse.ArgumentParser(description="Make a figure with the most confident AD and CN examples.")
    p.add_argument("--csv", type=Path, required=True, help="CSV with predictions, labels, and NIfTI paths.")
    p.add_argument("--out", type=Path, required=True, help="Output image path (.png/.svg/.pdf).")

    # Column names
    p.add_argument("--pred-col", type=str, default="autorater_prediction", help="Prediction probability column name.")
    p.add_argument("--nifti-col", type=str, default="nifti_path", help="NIfTI path column name.")
    p.add_argument("--h-col", type=str, default=None, help="Optional numeric binary H column name (0/1).")
    p.add_argument("--label-col", type=str, default="label", help="Label column name (used if --h-col is not given).")

    # Label tokens (used only if deriving H from label column)
    p.add_argument("--ad-token", type=str, default="AD", help="Token in label column that maps to H=1.")
    p.add_argument("--cn-token", type=str, default="CN", help="Token in label column that maps to H=0.")

    # Slice controls
    p.add_argument("--slice-axis", type=int, choices=[0, 1, 2], default=2, help="Axis to slice along (0,1,2).")
    p.add_argument("--slice-idx", type=int, default=None, help="Explicit slice index; default uses the middle slice.")
    p.add_argument("--vmin", type=float, default=None, help="Optional fixed intensity min for display.")
    p.add_argument("--vmax", type=float, default=None, help="Optional fixed intensity max for display.")

    # Styling
    p.add_argument("--figsize", type=float, nargs=2, default=(12.0, 8.0), help="Figure size in inches (W H).")
    p.add_argument("--dpi", type=int, default=200, help="Output DPI for raster formats.")
    p.add_argument("--title", type=str, default="Model prediction quality comparison: AD vs CN (examples)",
                   help="Suptitle for the figure.")
    p.add_argument("--log", type=str, default="INFO", help="Logging level (DEBUG, INFO, WARNING, ERROR).")
    return p.parse_args()


# ---------------------------------------------------------------------
# Data loading & validation
# ---------------------------------------------------------------------
def load_dataframe(
    csv_path: Path,
    pred_col: str,
    nifti_col: str,
    h_col: Optional[str],
    label_col: str,
    ad_token: str,
    cn_token: str,
) -> pd.DataFrame:
    if not csv_path.exists():
        raise FileNotFoundError(f"CSV not found: {csv_path}")

    df = pd.read_csv(csv_path)
    cols = set(df.columns)

    if pred_col not in cols:
        raise ValueError(f"Missing prediction column '{pred_col}'. Found: {list(cols)}")
    if nifti_col not in cols:
        raise ValueError(f"Missing NIfTI path column '{nifti_col}'. Found: {list(cols)}")

    # Build H
    if h_col and h_col in cols:
        H = pd.to_numeric(df[h_col], errors="coerce")
        if not set(H.dropna().unique()).issubset({0, 1}):
            raise ValueError(f"Column '{h_col}' must be binary in {{0,1}}.")
        df["H"] = H.astype("Int64")
    else:
        if label_col not in cols:
            raise ValueError(f"Need '{label_col}' to derive H, or pass --h-col.")
        lab = df[label_col].astype(str).str.strip().str.upper()
        df["H"] = pd.NA
        df.loc[lab == ad_token.upper(), "H"] = 1
        df.loc[lab == cn_token.upper(), "H"] = 0
        if df["H"].isna().any():
            n_na = int(df["H"].isna().sum())
            logging.warning("Derived H has %d missing rows (labels not matching %s/%s). Dropping them.", n_na, ad_token, cn_token)
        df = df.dropna(subset=["H"]).copy()
        df["H"] = df["H"].astype(int)

    # Coerce predictions and drop non-numeric
    p = pd.to_numeric(df[pred_col], errors="coerce")
    n_bad = int(p.isna().sum())
    if n_bad:
        logging.warning("Found %d non-numeric predictions in '%s'; dropping those rows.", n_bad, pred_col)
    df = df.loc[p.notna()].copy()
    df["autorater_prediction"] = p.loc[p.notna()].astype(float)

    # Ensure NIfTI paths exist
    missing = (~df[nifti_col].astype(str).map(lambda s: Path(s).exists())).sum()
    if missing:
        logging.warning("%d rows have missing NIfTI files; they will be excluded.", int(missing))
    df = df[df[nifti_col].astype(str).map(lambda s: Path(s).exists())].copy()
    df["nifti_path"] = df[nifti_col].astype(str)

    if df.empty:
        raise ValueError("No valid rows remaining after validation (check columns, values, and file paths).")

    return df


# ---------------------------------------------------------------------
# Selection & slicing
# ---------------------------------------------------------------------
def select_extremes(df: pd.DataFrame) -> Tuple[pd.Series, pd.Series]:
    """Return (best_AD_row, best_CN_row) based on max/min autorater_prediction within H=1/H=0."""
    ad_df = df[df["H"] == 1]
    cn_df = df[df["H"] == 0]
    if ad_df.empty or cn_df.empty:
        raise ValueError("Need at least one AD (H=1) and one CN (H=0) example in the CSV.")
    ad_row = ad_df.sort_values("autorater_prediction", ascending=False).iloc[0]
    cn_row = cn_df.sort_values("autorater_prediction", ascending=True).iloc[0]
    return ad_row, cn_row


def load_slice(nifti_path: str, axis: int, idx: Optional[int]) -> np.ndarray:
    """Load a 3D NIfTI, extract a slice along axis (default: middle), normalize to [0,1]."""
    img = nib.load(nifti_path)
    data = np.asanyarray(img.get_fdata())
    if data.ndim < 3:
        raise ValueError(f"Expected 3D NIfTI, got shape {data.shape} for {nifti_path}")

    # Choose slice index
    if idx is None:
        idx = data.shape[axis] // 2
    idx = int(np.clip(idx, 0, data.shape[axis] - 1))

    slicer = [slice(None)] * data.ndim
    slicer[axis] = idx
    sl = np.asarray(data[tuple(slicer)], dtype=float)

    # Normalize safely
    mn, mx = np.nanmin(sl), np.nanmax(sl)
    if not np.isfinite(mn) or not np.isfinite(mx) or mx <= mn:
        return np.zeros_like(sl, dtype=float)
    sl = (sl - mn) / (mx - mn)
    return sl


# ---------------------------------------------------------------------
# Plotting
# ---------------------------------------------------------------------
def make_figure(
    ad_slice: np.ndarray,
    ad_pred: float,
    ad_h: int,
    cn_slice: np.ndarray,
    cn_pred: float,
    cn_h: int,
    figsize: Tuple[float, float],
    dpi: int,
    title: str,
    out_path: Path,
    vmin: Optional[float],
    vmax: Optional[float],
) -> None:
    fig, axes = plt.subplots(2, 3, figsize=figsize, constrained_layout=True)

    # AD row
    axes[0, 0].imshow(ad_slice, cmap="gray", vmin=vmin, vmax=vmax)
    axes[0, 0].set_title("(A) AD MRI Slice")
    axes[0, 0].axis("off")

    axes[0, 1].bar(["P(AD)"], [ad_pred])
    axes[0, 1].set_ylim(0, 1)
    axes[0, 1].set_title("(B) Autorater Prediction")

    axes[0, 2].text(
        0.5, 0.5, f"Ground-Truth H = {'AD' if ad_h == 1 else 'CN'}",
        fontsize=14, ha="center", va="center"
    )
    axes[0, 2].axis("off")
    axes[0, 2].set_title("(C) Ground-Truth Label")

    # CN row
    axes[1, 0].imshow(cn_slice, cmap="gray", vmin=vmin, vmax=vmax)
    axes[1, 0].set_title("(A) CN MRI Slice")
    axes[1, 0].axis("off")

    axes[1, 1].bar(["P(AD)"], [cn_pred])
    axes[1, 1].set_ylim(0, 1)
    axes[1, 1].set_title("(B) Autorater Prediction")

    axes[1, 2].text(
        0.5, 0.5, f"Ground-Truth H = {'AD' if cn_h == 1 else 'CN'}",
        fontsize=14, ha="center", va="center"
    )
    axes[1, 2].axis("off")
    axes[1, 2].set_title("(C) Ground-Truth Label")

    plt.suptitle(title, fontsize=16)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path, dpi=dpi)
    plt.close(fig)


# ---------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------
def main() -> None:
    args = parse_args()
    setup_logging(args.log)

    df = load_dataframe(
        csv_path=args.csv,
        pred_col=args.pred_col,
        nifti_col=args.nifti_col,
        h_col=args.h_col,
        label_col=args.label_col,
        ad_token=args.ad_token,
        cn_token=args.cn_token,
    )

    ad_row, cn_row = select_extremes(df)

    # Load slices
    ad_slice = load_slice(ad_row["nifti_path"], axis=args.slice_axis, idx=args.slice_idx)
    cn_slice = load_slice(cn_row["nifti_path"], axis=args.slice_axis, idx=args.slice_idx)

    ad_pred, ad_h = float(ad_row["autorater_prediction"]), int(ad_row["H"])
    cn_pred, cn_h = float(cn_row["autorater_prediction"]), int(cn_row["H"])

    make_figure(
        ad_slice=ad_slice,
        ad_pred=ad_pred,
        ad_h=ad_h,
        cn_slice=cn_slice,
        cn_pred=cn_pred,
        cn_h=cn_h,
        figsize=(float(args.figsize[0]), float(args.figsize[1])),
        dpi=int(args.dpi),
        title=args.title,
        out_path=args.out,
        vmin=args.vmin,
        vmax=args.vmax,
    )

    logging.info("Saved figure: %s", args.out.resolve())


if __name__ == "__main__":
    main()
