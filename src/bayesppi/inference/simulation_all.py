#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
simulation_all.py

Build a matched table of CN/AD labels and NIfTI paths by merging:
  (1) an "ALL" cohort metadata CSV (subject, group label, acquisition date)
  (2) a filesystem of NIfTI files produced by dcm2niix.

Repository/TMLR-friendly upgrades:
- No hard-coded paths: all I/O via CLI
- Robust column cleaning and validation
- Multiple date-format fallbacks
- Flexible filename parsing (subject/date tokens)
- Recursive file search with glob pattern
- Clear logging; deterministic output

Examples
--------
# Basic usage (defaults mirror the original intent)
python simulation_all.py \
  --meta data/processed/all_people_7_20_2025.csv \
  --nifti-root ADNI_NIfTI/ \
  --out data/processed/matched_cn_ad_labels_all.csv

# If your NIfTI filenames differ, specify token indices and formats:
# <SUBJ_PART0>_<PART1>_<PART2>_<DATE_TOKEN>_... .nii.gz
python simulation_all.py \
  --meta data/processed/all_people_7_20_2025.csv \
  --nifti-root ADNI_NIfTI/ \
  --subject-token-idx 0 1 2 \
  --date-token-idx 3 \
  --date-token-fmt "%Y-%m-%d" \
  --out data/processed/matched_cn_ad_labels_all.csv

# If the meta CSV has different column names:
python simulation_all.py \
  --meta data/processed/meta.csv \
  --meta-subject-col Subject \
  --meta-group-col Group \
  --meta-date-col Acq_Date \
  --meta-date-fmts "%m/%d/%Y" "%Y-%m-%d" \
  --nifti-root ADNI_NIfTI/ \
  --out data/processed/matched.csv
"""
from __future__ import annotations

import argparse
import logging
from pathlib import Path
from typing import List, Optional

import pandas as pd


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
    p = argparse.ArgumentParser(description="Match ALL-cohort meta CSV to NIfTI files.")
    p.add_argument("--meta", type=Path, required=True, help="Path to ALL cohort metadata CSV.")
    p.add_argument("--nifti-root", type=Path, required=True, help="Root directory with .nii.gz files (per subject).")
    p.add_argument("--out", type=Path, required=True, help="Output CSV path for matches.")

    # Meta CSV column names (after cleaning)
    p.add_argument("--meta-subject-col", type=str, default="Subject", help="Subject ID column (default: Subject).")
    p.add_argument("--meta-group-col", type=str, default="Group", help="Group/label column (default: Group).")
    p.add_argument("--meta-date-col", type=str, default="Acq_Date", help="Acquisition date column (default: Acq_Date).")

    # Date parsing
    p.add_argument("--meta-date-fmts", type=str, nargs="+",
                   default=["%m/%d/%Y"], help="Allowed date formats for meta date (try in order).")
    p.add_argument("--date-token-fmt", type=str, default="%Y-%m-%d",
                   help="Date format used in NIfTI filename token (default: %%Y-%%m-%%d).")

    # Filename tokenization: split by '_' and join subject tokens; date token index gives scan date
    p.add_argument("--subject-token-idx", type=int, nargs="+", default=[0, 1, 2],
                   help="Indices of filename tokens to join into subject_id (default: 0 1 2).")
    p.add_argument("--date-token-idx", type=int, default=3,
                   help="Index of the filename token containing the scan date (default: 3).")

    # Recursive glob pattern
    p.add_argument("--glob", type=str, default="**/*.nii.gz",
                   help="Glob pattern for NIfTI files relative to --nifti-root (default: **/*.nii.gz).")

    # Logging
    p.add_argument("--log", type=str, default="INFO", help="Logging level (DEBUG, INFO, WARNING, ERROR).")
    return p.parse_args()


# ---------------------------------------------------------------------
# Utilities
# ---------------------------------------------------------------------
def clean_columns(df: pd.DataFrame) -> pd.DataFrame:
    """Trim whitespace and replace internal spaces with underscores."""
    df = df.copy()
    df.columns = (
        df.columns
        .str.strip()
        .str.replace(r"\s+", "_", regex=True)
    )
    return df


def parse_dates_with_fallback(s: pd.Series, fmts: List[str]) -> pd.Series:
    """Try multiple date formats; return pandas datetime (date)."""
    out = pd.to_datetime(s, errors="coerce", infer_datetime_format=False)
    mask = out.isna()
    for fmt in fmts:
        if not mask.any():
            break
        try_out = pd.to_datetime(s[mask], format=fmt, errors="coerce")
        out.loc[mask] = try_out
        mask = out.isna()
    return out.dt.date


def extract_subject_and_date_from_filename(
    path: Path,
    subj_idx: List[int],
    date_idx: int,
    date_fmt: str,
) -> Optional[tuple[str, pd.Timestamp]]:
    """
    Parse subject_id and scan_date from a NIfTI filename stem by splitting on '_'.
    Returns (subject_id, date) or None if parsing fails.

    Handles double suffix (.nii.gz) by stripping the trailing '.nii' from the stem if present.
    """
    stem = path.stem  # '...nii.gz' -> '...nii', then strip '.nii' if present
    if stem.endswith(".nii"):
        stem = stem[:-4]
    parts = stem.split("_")
    try:
        subject_id = "_".join(parts[i] for i in subj_idx)
        date_token = parts[date_idx]
    except Exception:
        return None
    try:
        scan_date = pd.to_datetime(date_token, format=date_fmt).date()
    except Exception:
        return None
    return subject_id, scan_date


# ---------------------------------------------------------------------
# Core
# ---------------------------------------------------------------------
def main() -> None:
    args = parse_args()
    setup_logging(args.log)

    # Load meta CSV
    if not args.meta.exists():
        raise FileNotFoundError(f"Meta CSV not found: {args.meta}")
    meta = pd.read_csv(args.meta)
    meta = clean_columns(meta)

    # Validate required columns (after cleaning)
    s_col, g_col, d_col = args.meta_subject_col, args.meta_group_col, args.meta_date_col
    for col in (s_col, g_col, d_col):
        if col not in meta.columns:
            raise ValueError(f"Missing column '{col}' in meta CSV after cleaning. Found: {list(meta.columns)}")

    # Parse dates in meta
    meta[d_col] = parse_dates_with_fallback(meta[d_col], args.meta_date_fmts)
    n_bad_dates = int(meta[d_col].isna().sum())
    if n_bad_dates:
        logging.warning("Meta CSV: %d rows have unparseable dates in '%s' and will be dropped.", n_bad_dates, d_col)
    meta = meta.dropna(subset=[d_col]).copy()

    # Keep only needed columns, normalize names
    meta = meta.rename(columns={s_col: "subject_id", g_col: "label", d_col: "Acq_Date"})
    meta = meta[["subject_id", "label", "Acq_Date"]].drop_duplicates()

    logging.info("Meta rows after cleaning/dedup: %d", len(meta))

    # Scan NIfTI directory
    if not args.nifti_root.exists():
        raise FileNotFoundError(f"NIfTI root not found: {args.nifti_root}")

    records = []
    n_scanned = 0
    for nii in args.nifti_root.glob(args.glob):
        if not nii.is_file():
            continue
        n_scanned += 1
        parsed = extract_subject_and_date_from_filename(
            nii,
            subj_idx=args.subject_token_idx,
            date_idx=args.date_token_idx,
            date_fmt=args.date_token_fmt,
        )
        if parsed is None:
            continue
        subject_id, scan_date = parsed
        records.append({"subject_id": subject_id, "scan_date": scan_date, "nifti_path": str(nii)})

    nifti_df = pd.DataFrame(records)
    logging.info("Scanned %d files, parsed %d subject/date tokens.", n_scanned, len(nifti_df))

    if nifti_df.empty:
        logging.warning("No NIfTI files matched the parsing rules. Nothing to merge.")
        # Still write an empty file with headers for reproducibility
        out_empty = pd.DataFrame(columns=["subject_id", "scan_date", "nifti_path", "label", "Acq_Date"])
        args.out.parent.mkdir(parents=True, exist_ok=True)
        out_empty.to_csv(args.out, index=False)
        logging.info("Wrote empty matches CSV: %s", args.out.resolve())
        return

    # Merge by exact (subject_id, date)
    merged = pd.merge(
        nifti_df,
        meta,
        left_on=["subject_id", "scan_date"],
        right_on=["subject_id", "Acq_Date"],
        how="inner",
    ).drop_duplicates()

    logging.info("Number of matched samples: %d", len(merged))
    if len(merged):
        logging.debug("\n%s", merged.head().to_string(index=False))

    # Save
    args.out.parent.mkdir(parents=True, exist_ok=True)
    merged.to_csv(args.out, index=False)
    logging.info("✅ Saved to %s", args.out.resolve())


if __name__ == "__main__":
    main()
