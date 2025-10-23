#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
change_dicom.py

Robust, repository-friendly DICOM → NIfTI converter using dcm2niix.

Why this version:
- No hard-coded paths (everything via CLI flags)
- Cross-platform (Windows/macOS/Linux) and shell-safe
- Clear logging and dry-run support
- Flexible series discovery via --glob (default looks for */MPRAGE/*)
- Deterministic, informative output filenames
- Optional CSV manifest of conversions

Examples
--------
# Basic (assumes dcm2niix in PATH)
python change_dicom.py \
  --dicom-root /data/ADNI \
  --out /data/ADNI_NIfTI

# With explicit dcm2niix path and different series pattern
python change_dicom.py \
  --dicom-root "C:/Users/me/Documents/ADNI" \
  --out "C:/Users/me/Documents/ADNI_NIfTI" \
  --dcm2niix "C:/tools/dcm2niix.exe" \
  --glob "*/MPRAGE/*"

# Dry-run to preview commands (no conversion)
python change_dicom.py --dicom-root ... --out ... --dry-run
"""
from __future__ import annotations

import argparse
import csv
import logging
import os
import shlex
import shutil
import subprocess
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable, List, Optional, Tuple


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
    p = argparse.ArgumentParser(description="Convert DICOM series to NIfTI using dcm2niix.")
    p.add_argument("--dicom-root", type=Path, required=True, help="Root directory containing subject DICOM folders.")
    p.add_argument("--out", type=Path, required=True, help="Output root directory for NIfTI.")
    p.add_argument("--dcm2niix", type=str, default="dcm2niix", help="Path to dcm2niix executable (or name if in PATH).")
    p.add_argument("--glob", type=str, default="*/MPRAGE/*", help="Glob pattern (relative to dicom-root) to find series folders.")
    p.add_argument("--gzip", choices=["y", "n"], default="y", help="Gzip output NIfTI (dcm2niix -z).")
    p.add_argument("--bids", choices=["y", "n"], default="n", help="Write BIDS sidecars (dcm2niix -b).")
    p.add_argument("--crop", choices=["y", "n"], default="n", help="Crop images (dcm2niix -x).")
    p.add_argument("--merge", choices=["n", "a", "v", "o", "t"], default="n",
                   help="Merge 2D slices (dcm2niix -m). See dcm2niix docs.")
    p.add_argument("--manifest", type=Path, default=None, help="Optional path to write a CSV manifest of conversions.")
    p.add_argument("--dry-run", action="store_true", help="Print the commands without executing.")
    p.add_argument("--log", type=str, default="INFO", help="Logging level: DEBUG, INFO, WARNING, ERROR.")
    return p.parse_args()


# ---------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------
def resolve_dcm2niix(exe: str) -> str:
    """
    Resolve dcm2niix path or raise a helpful error.
    """
    # If user passed an absolute/relative path that exists, use it
    exe_path = Path(exe)
    if exe_path.exists():
        return str(exe_path)

    # Otherwise try to find in PATH
    which = shutil.which(exe)
    if which:
        return which

    raise FileNotFoundError(
        f"Could not find dcm2niix executable: '{exe}'. "
        f"Provide a valid path via --dcm2niix or ensure it's in your PATH."
    )


@dataclass
class SeriesInfo:
    series_dir: Path
    subject_id: str
    scan_token: str  # usually date or folder name token


def infer_subject_and_date(series_dir: Path, root: Path) -> Tuple[str, str]:
    """
    Try to infer a subject ID and a scan token (often a date) from the path.
    The original script assumed: <root>/<SUBJECT>/MPRAGE/<DATE_TOKEN>/...
    Here we attempt a robust fallback.

    Returns: (subject_id, scan_token)
    """
    # Parts relative to root
    try:
        rel = series_dir.relative_to(root)
        parts = list(rel.parts)
    except Exception:
        parts = list(series_dir.parts)

    # Heuristics:
    # - subject likely near the front (e.g., "002_S_0729")
    # - date token often last folder name
    subject_id = "unknown_subject"
    scan_token = series_dir.name

    if len(parts) >= 1:
        scan_token = parts[-1]
    if len(parts) >= 2:
        # If the pattern */MPRAGE/* is used, subject is typically parts[0]
        subject_id = parts[0]
    # Fallback: look for something that looks like site_subject pattern
    for p in parts:
        if any(c.isdigit() for c in p) and "_" in p:
            subject_id = p
            break

    return subject_id, scan_token


def iter_series(dicom_root: Path, pattern: str) -> Iterable[Path]:
    """
    Yield candidate series directories under dicom_root matching pattern.
    Using glob (not rglob) relative to the root to keep parity with user's original.
    """
    yield from (dicom_root.glob(pattern))


def build_output_name(subject_id: str, scan_token: str) -> str:
    """
    Compose a safe output base filename for dcm2niix (-f).
    """
    safe_subj = "".join(ch if ch.isalnum() or ch in "-_." else "_" for ch in subject_id)
    safe_scan = "".join(ch if ch.isalnum() or ch in "-_." else "_" for ch in scan_token)
    return f"{safe_subj}_{safe_scan}"


def run_dcm2niix(
    dcm2niix: str,
    series_dir: Path,
    out_dir: Path,
    out_base: str,
    gzip: str = "y",
    bids: str = "n",
    crop: str = "n",
    merge: str = "n",
    dry_run: bool = False,
) -> subprocess.CompletedProcess | None:
    """
    Build and (optionally) run the dcm2niix command.
    Returns CompletedProcess (or None if dry-run).
    """
    cmd = [
        dcm2niix,
        "-z", gzip,             # gzip NIfTI
        "-b", bids,             # BIDS sidecar
        "-x", crop,             # crop
        "-m", merge,            # merge slices
        "-f", out_base,         # output filename base
        "-o", str(out_dir),     # output directory
        str(series_dir),        # input DICOM series dir
    ]

    # Log a shell-safe echo of the command
    pretty = " ".join(shlex.quote(c) for c in cmd)
    logging.info("dcm2niix: %s", pretty)

    if dry_run:
        return None

    out_dir.mkdir(parents=True, exist_ok=True)
    return subprocess.run(cmd, check=False, capture_output=True, text=True)


# ---------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------
def main() -> None:
    args = parse_args()
    setup_logging(args.log)

    dicom_root: Path = args.dicom_root
    out_root: Path = args.out
    dcm2niix = resolve_dcm2niix(args.dcm2niix)

    if not dicom_root.exists():
        raise FileNotFoundError(f"DICOM root does not exist: {dicom_root}")

    out_root.mkdir(parents=True, exist_ok=True)

    manifest_rows: List[dict] = []

    found_any = False
    for series_dir in iter_series(dicom_root, args.glob):
        if not series_dir.is_dir():
            continue
        found_any = True

        subject_id, scan_token = infer_subject_and_date(series_dir, dicom_root)
        out_dir = out_root / subject_id
        out_base = build_output_name(subject_id, scan_token)

        # Run conversion
        result = run_dcm2niix(
            dcm2niix=dcm2niix,
            series_dir=series_dir,
            out_dir=out_dir,
            out_base=out_base,
            gzip=args.gzip,
            bids=args.bids,
            crop=args.crop,
            merge=args.merge,
            dry_run=args.dry_run,
        )

        # Record manifest row
        row = {
            "series_dir": str(series_dir),
            "subject_id": subject_id,
            "scan_token": scan_token,
            "out_dir": str(out_dir),
            "out_base": out_base,
            "status": "DRY-RUN" if args.dry_run else ("OK" if (result and result.returncode == 0) else "ERROR"),
        }
        if result and not args.dry_run:
            row["stdout"] = (result.stdout or "").strip()
            row["stderr"] = (result.stderr or "").strip()
            if result.returncode != 0:
                logging.warning("dcm2niix returned non-zero (%d) for %s", result.returncode, series_dir)
        manifest_rows.append(row)

    if not found_any:
        logging.warning("No series matched pattern '%s' under %s", args.glob, dicom_root)

    # Write manifest CSV if requested
    if args.manifest:
        args.manifest.parent.mkdir(parents=True, exist_ok=True)
        fieldnames = ["series_dir", "subject_id", "scan_token", "out_dir", "out_base", "status", "stdout", "stderr"]
        # Ensure all keys exist for CSV header even in dry-run entries
        for r in manifest_rows:
            r.setdefault("stdout", "")
            r.setdefault("stderr", "")
        with args.manifest.open("w", newline="", encoding="utf-8") as f:
            w = csv.DictWriter(f, fieldnames=fieldnames)
            w.writeheader()
            w.writerows(manifest_rows)
        logging.info("Wrote manifest: %s", args.manifest.resolve())

    logging.info("Done. Output root: %s", out_root.resolve())


if __name__ == "__main__":
    main()
