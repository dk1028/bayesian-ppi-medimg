#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
fig1_pipeline.py

Make a small "pipeline" figure showing a raw axial MRI slice (from a NIfTI file)
and its preprocessed version (normalized + resized to 64×64 by default).

Repository/TMLR-friendly:
- No hard-coded paths (all via CLI)
- Headless matplotlib backend (no GUI; saves to file)
- Works with .nii and .nii.gz
- Robust file discovery under a root directory or use an explicit --nifti path
- Skimage resize if available; otherwise falls back to SciPy (zoom) or a simple
  nearest-neighbor implementation (no extra deps required)

Examples
--------
# 1) Auto-pick the first NIfTI under a root directory, save PNG
python fig1_pipeline.py \
  --root ADNI_NIfTI/ \
  --out figures/fig1_pipeline.png

# 2) Use a specific NIfTI path and custom output size & slice
python fig1_pipeline.py \
  --nifti ADNI_NIfTI/002_S_0729/002_S_0729_2017-05-17.nii.gz \
  --size 64 64 \
  --slice-axis 2 \
  --slice-idx 80 \
  --out figures/fig1_pipeline.svg \
  --dpi 200
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
    p = argparse.ArgumentParser(description="Render raw vs. preprocessed MRI slice from a NIfTI volume.")
    g = p.add_mutually_exclusive_group(required=True)
    g.add_argument("--root", type=Path, help="Root directory to search for a NIfTI (first match is used).")
    g.add_argument("--nifti", type=Path, help="Explicit path to a .nii or .nii.gz file.")

    p.add_argument("--glob", type=str, default="**/*.nii.gz",
                   help="Glob (relative to --root) to find NIfTI (default: **/*.nii.gz).")
    p.add_argument("--fallback-glob", type=str, default="**/*.nii",
                   help="Fallback glob if none found with --glob (default: **/*.nii).")

    p.add_argument("--slice-axis", type=int, choices=[0, 1, 2], default=2,
                   help="Slice axis (0, 1, or 2). Default: 2 (axial).")
    p.add_argument("--slice-idx", type=int, default=None,
                   help="Slice index; default uses the middle slice.")
    p.add_argument("--size", type=int, nargs="+", default=[64, 64],
                   help="Output size H W (default: 64 64).")
    p.add_argument("--vmin", type=float, default=None, help="Optional fixed intensity min for display.")
    p.add_argument("--vmax", type=float, default=None, help="Optional fixed intensity max for display.")

    p.add_argument("--title-raw", type=str, default="(A) Raw NIfTI slice", help="Title for raw slice panel.")
    p.add_argument("--title-proc", type=str, default="(B) Resized & normalized",
                   help="Title for processed slice panel.")

    p.add_argument("--out", type=Path, required=True, help="Output image path (.png, .svg, .pdf, etc.).")
    p.add_argument("--dpi", type=int, default=300, help="DPI for raster formats.")
    p.add_argument("--figsize", type=float, nargs=2, default=(6.0, 3.0), help="Figure size in inches (W H).")
    p.add_argument("--log", type=str, default="INFO", help="Logging level (DEBUG, INFO, WARNING, ERROR).")
    return p.parse_args()


# ---------------------------------------------------------------------
# Resizing backends
# ---------------------------------------------------------------------
def _resize_skimage(img: np.ndarray, out_hw: Tuple[int, int]) -> np.ndarray:
    from skimage.transform import resize as sk_resize  # type: ignore
    return sk_resize(
        img, out_hw, order=1, mode="constant", cval=0.0, anti_aliasing=True, preserve_range=True
    ).astype(float)


def _resize_scipy(img: np.ndarray, out_hw: Tuple[int, int]) -> np.ndarray:
    from scipy.ndimage import zoom  # type: ignore
    in_h, in_w = img.shape[:2]
    out_h, out_w = out_hw
    zoom_h, zoom_w = out_h / max(in_h, 1), out_w / max(in_w, 1)
    return zoom(img, (zoom_h, zoom_w), order=1)


def _resize_nn(img: np.ndarray, out_hw: Tuple[int, int]) -> np.ndarray:
    """Nearest-neighbor fallback using numpy (no external deps)."""
    in_h, in_w = img.shape[:2]
    out_h, out_w = out_hw
    if in_h == 0 or in_w == 0:
        return np.zeros(out_hw, dtype=float)
    y = (np.linspace(0, in_h - 1, out_h)).astype(int)
    x = (np.linspace(0, in_w - 1, out_w)).astype(int)
    return img[np.ix_(y, x)]


def resize_image(img: np.ndarray, out_hw: Tuple[int, int]) -> np.ndarray:
    """Try skimage → scipy → nearest neighbor."""
    try:
        return _resize_skimage(img, out_hw)
    except Exception:
        try:
            return _resize_scipy(img, out_hw)
        except Exception:
            logging.warning("Falling back to nearest-neighbor resize (install scikit-image or SciPy for better quality).")
            return _resize_nn(img, out_hw)


# ---------------------------------------------------------------------
# Core helpers
# ---------------------------------------------------------------------
def find_first_nifti(root: Path, glob: str, fallback_glob: str) -> Path:
    hits = sorted(root.glob(glob))
    if not hits:
        hits = sorted(root.glob(fallback_glob))
    if not hits:
        raise FileNotFoundError(f"No NIfTI files found under {root} with patterns '{glob}' or '{fallback_glob}'.")
    return hits[0]


def load_slice(nifti_path: Path, axis: int, idx: Optional[int]) -> np.ndarray:
    img = nib.load(str(nifti_path))
    data = np.asanyarray(img.get_fdata())
    if data.ndim < 3:
        raise ValueError(f"Expected 3D NIfTI, got shape {data.shape} for {nifti_path}")

    if idx is None:
        idx = data.shape[axis] // 2
    idx = int(np.clip(idx, 0, data.shape[axis] - 1))

    slicer = [slice(None)] * data.ndim
    slicer[axis] = idx
    sl = np.asarray(data[tuple(slicer)], dtype=float)

    # Normalize to [0,1] safely
    mn, mx = np.nanmin(sl), np.nanmax(sl)
    if not np.isfinite(mn) or not np.isfinite(mx) or mx <= mn:
        return np.zeros_like(sl, dtype=float)
    return (sl - mn) / (mx - mn)


# ---------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------
def main() -> None:
    args = parse_args()
    setup_logging(args.log)

    # Resolve NIfTI path
    if args.nifti:
        nifti_path = args.nifti
    else:
        if not args.root.exists():
            raise FileNotFoundError(f"--root does not exist: {args.root}")
        nifti_path = find_first_nifti(args.root, args.glob, args.fallback_glob)
    logging.info("Using NIfTI: %s", nifti_path)

    # Load + preprocess
    raw = load_slice(nifti_path, axis=args.slice_axis, idx=args.slice_idx)

    size = args.size
    if len(size) == 1:
        out_hw = (int(size[0]), int(size[0]))
    elif len(size) >= 2:
        out_hw = (int(size[0]), int(size[1]))
    else:
        out_hw = (64, 64)

    proc = resize_image(raw, out_hw)

    # Plot and save
    fig, axes = plt.subplots(1, 2, figsize=(float(args.figsize[0]), float(args.figsize[1])))
    axes[0].imshow(raw, cmap="gray", vmin=args.vmin, vmax=args.vmax)
    axes[0].set_title(args.title_raw)
    axes[0].axis("off")

    axes[1].imshow(proc, cmap="gray", vmin=args.vmin, vmax=args.vmax)
    axes[1].set_title(f"{args.title_proc} ({out_hw[1]}×{out_hw[0]})")
    axes[1].axis("off")

    fig.tight_layout()
    args.out.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(args.out, dpi=int(args.dpi))
    plt.close(fig)

    logging.info("Saved pipeline figure: %s", args.out.resolve())


if __name__ == "__main__":
    main()
