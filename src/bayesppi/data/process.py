#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
process.py

Minimal, repo-friendly script to render a simple 3D-CNN pipeline diagram
(Input → Conv3D → Pool → … → Sigmoid) as a horizontal flow. Designed for:
- Headless environments (CI/docs) with matplotlib Agg backend
- No interactive windows (saves to file)
- No hard-coded paths (all via CLI)
- Clean typography and consistent spacing
- Deterministic output (single RNG not required)

Examples
--------
# 1) Use the default example and save as PNG
python process.py --out figures/fig1_pipeline.png

# 2) Custom layers (labels auto-spaced) and SVG output
python process.py \
  --layers "Input\n1×64×64×64;Conv3D\n32×62×62×62;MaxPool\n32×31×31×31;Conv3D\n64×29×29×29;MaxPool\n64×14×14×14;Flatten;Dense\n128;Sigmoid\n1" \
  --out figures/fig1_pipeline.svg --figsize 12 3 --dpi 200

# 3) Wider spacing and thicker arrows
python process.py --xstep 2.2 --arrow-width 0.06 --out figures/pipeline.png
"""
from __future__ import annotations

import argparse
import logging
from pathlib import Path
from typing import List, Tuple

import matplotlib

# Headless backend for GitHub Actions / docs builds
matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
from matplotlib.patches import Rectangle, FancyArrow  # noqa: E402


# ---------------------------------------------------------------------
# CLI / Logging
# ---------------------------------------------------------------------
def setup_logging(level: str = "INFO") -> None:
    log_level = getattr(logging, level.upper(), logging.INFO)
    logging.basicConfig(level=log_level, format="[%(levelname)s] %(message)s")


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Render a simple model pipeline diagram.")
    p.add_argument(
        "--layers",
        type=str,
        default="Input\n1×64×64×64;Conv3D\n32×62×62×62;MaxPool\n32×31×31×31;Conv3D\n64×29×29×29;MaxPool\n64×14×14×14;Flatten;Dense\n128;Sigmoid\n1",
        help="Semicolon-separated list of layer labels. Use '\\n' for line breaks within a box.",
    )
    p.add_argument("--out", type=Path, required=True, help="Output image path (.png, .svg, .pdf, etc.).")
    p.add_argument("--figsize", type=float, nargs=2, default=(12.0, 3.0), help="Figure size (inches): W H.")
    p.add_argument("--dpi", type=int, default=200, help="DPI for raster formats (PNG, JPG).")
    p.add_argument("--box-width", type=float, default=1.6, help="Width of each layer box (axis units).")
    p.add_argument("--box-height", type=float, default=0.75, help="Height of each layer box (axis units).")
    p.add_argument("--y", type=float, default=0.5, help="Vertical baseline for boxes (axis units).")
    p.add_argument("--xstep", type=float, default=2.0, help="Horizontal step between box anchors (axis units).")
    p.add_argument("--xpad", type=float, default=0.5, help="Extra x padding at both ends for aesthetics.")
    p.add_argument("--arrow-len", type=float, default=0.6, help="Arrow length between boxes (axis units).")
    p.add_argument("--arrow-width", type=float, default=0.05, help="Arrow shaft width.")
    p.add_argument("--font-size", type=float, default=10.0, help="Font size inside boxes.")
    p.add_argument("--no-arrows", action="store_true", help="Disable arrows between boxes.")
    p.add_argument("--log", type=str, default="INFO", help="Logging level (DEBUG, INFO, WARNING, ERROR).")
    return p.parse_args()


# ---------------------------------------------------------------------
# Core rendering
# ---------------------------------------------------------------------
def parse_layer_labels(spec: str) -> List[str]:
    """
    Parse the semicolon-separated layer spec. Supports '\n' for multi-line labels.
    """
    parts = [s.strip() for s in spec.split(";") if s.strip()]
    return [p.replace("\\n", "\n") for p in parts]


def compute_layout(n_layers: int, xstep: float, xpad: float) -> Tuple[float, float]:
    """
    Compute axis limits to fit n_layers boxes with spacing and padding.
    """
    if n_layers <= 0:
        return -0.5, 0.5
    xmin = -xpad
    xmax = (n_layers - 1) * xstep + xpad
    return xmin, xmax


def draw_pipeline(
    ax: plt.Axes,
    layers: List[str],
    y: float,
    box_w: float,
    box_h: float,
    xstep: float,
    arrow_len: float,
    arrow_w: float,
    font_size: float,
    draw_arrows: bool = True,
) -> None:
    """
    Draw boxes (with labels) and optional arrows along a horizontal line.
    """
    # Consistent gray for fill/edges
    face = (0.9, 0.9, 0.9)
    edge = (0.0, 0.0, 0.0)

    for idx, label in enumerate(layers):
        x = idx * xstep
        ax.add_patch(Rectangle((x, y), box_w, box_h, fill=True, edgecolor=edge, facecolor=face))
        ax.text(x + box_w / 2.0, y + box_h / 2.0, label, ha="center", va="center", fontsize=font_size)
        # Arrow to next
        if draw_arrows and idx < len(layers) - 1:
            ax.add_patch(
                FancyArrow(
                    x + box_w,
                    y + box_h / 2.0,
                    arrow_len,
                    0.0,
                    width=arrow_w,
                    length_includes_head=True,
                    head_width=arrow_w * 3.0,
                    head_length=arrow_len * 0.35,
                    edgecolor=edge,
                    facecolor=edge,
                )
            )


def render_pipeline(
    layers_spec: str,
    out_path: Path,
    figsize: Tuple[float, float],
    dpi: int,
    box_w: float,
    box_h: float,
    y: float,
    xstep: float,
    xpad: float,
    arrow_len: float,
    arrow_w: float,
    font_size: float,
    draw_arrows: bool,
) -> None:
    layers = parse_layer_labels(layers_spec)
    xmin, xmax = compute_layout(len(layers), xstep=xstep, xpad=xpad)

    fig, ax = plt.subplots(figsize=figsize)
    draw_pipeline(
        ax=ax,
        layers=layers,
        y=y,
        box_w=box_w,
        box_h=box_h,
        xstep=xstep,
        arrow_len=arrow_len,
        arrow_w=arrow_w,
        font_size=font_size,
        draw_arrows=draw_arrows,
    )

    # Aesthetics
    ax.set_xlim(xmin, xmax + 0.5)  # small extra pad on right
    ax.set_ylim(0, y + box_h + 0.75)
    ax.axis("off")
    fig.tight_layout()

    # Save (dpi applies to raster)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path, dpi=dpi)
    plt.close(fig)


# ---------------------------------------------------------------------
# Entry
# ---------------------------------------------------------------------
def main() -> None:
    args = parse_args()
    setup_logging(args.log)

    render_pipeline(
        layers_spec=args.layers,
        out_path=args.out,
        figsize=(float(args.figsize[0]), float(args.figsize[1])),
        dpi=int(args.dpi),
        box_w=float(args.box_width),
        box_h=float(args.box_height),
        y=float(args.y),
        xstep=float(args.xstep),
        xpad=float(args.xpad),
        arrow_len=float(args.arrow_len),
        arrow_w=float(args.arrow_width),
        font_size=float(args.font_size),
        draw_arrows=(not args.no_arrows),
    )

    logging.info("Saved diagram: %s", args.out.resolve())


if __name__ == "__main__":
    main()
