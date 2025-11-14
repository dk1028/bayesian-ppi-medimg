#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
implementation_6570.py — 65–70 subset prevalence simulation (CRE / Naive / PPI / Difference)

- TMLR/repo-friendly CLI script (no Colab mounts)
- Reads a predictions CSV and (optionally) filters to an age range (default: 65–70 inclusive)
- Builds A_class by thresholding a probability column; derives H from either a binary column or a string label
- Runs nsim label-budget experiments across priors and estimators
- Saves coverage & interval-width summary to CSV

Estimators:
  • CRE (Chain-Rule Estimator, Bayesian) via PyMC (Binomial count likelihoods, Beta priors)
  • Naïve Bayesian (labels only; Beta–Bernoulli)
  • PPI analytic (A_prob + (H−A_prob) rectifier; CI via Var(A)/N + Var(H−A)/n, small-n t critical)
  • Difference estimator (design-based) with bootstrap CI (uses A_class)

Usage example:
  python src/bayesppi/inference/implementation_6570.py \
    --pred-csv data/processed/autorater_predictions_all2.csv \
    --pred-col autorater_prediction \
    --label-col label --label-pos AD \
    --age-col Age --age-min 65 --age-max 70 \
    --threshold 0.5 \
    --nsim 50 --label-sizes 10 20 40 80 \
    --out-csv runs/simulation_6570_results.csv

Notes:
  • Deterministic behavior across platforms: cores=1 in PyMC, single RNG seed reused across priors,
    and pre-drawn labeled indices per label budget.
  • Increase --draws/--tune for final results; keep smaller for quick checks.
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path
from typing import Iterable, Tuple, Dict, List

import numpy as np
import pandas as pd
from scipy.stats import t as student_t  # small-n critical value

# Optional heavy import with friendly error if missing
try:
    import pymc as pm
except Exception as e:  # pragma: no cover
    pm = None
    _PM_ERR = e
else:
    _PM_ERR = None


# --------------------------------------------------------------------------------------
# Utilities
# --------------------------------------------------------------------------------------

def _as_int_series(x: pd.Series, name: str = "binary") -> pd.Series:
    y = pd.to_numeric(x, errors="coerce")
    if y.isna().any():
        bad = x[y.isna()].unique()[:5]
        raise ValueError(f"Non-numeric values in '{name}'; examples: {bad!r}")
    y = y.astype(int)
    if not set(y.unique()).issubset({0, 1}):
        raise ValueError(f"Column '{name}' must be binary in {{0,1}}.")
    return y


def derive_H(df: pd.DataFrame, h_col: str | None, label_col: str | None, label_pos: str) -> pd.Series:
    """Return binary H from either an explicit H column or a string label column."""
    if h_col and h_col in df.columns:
        return _as_int_series(df[h_col], name=h_col)
    if label_col and label_col in df.columns:
        lab = df[label_col].astype(str).str.upper().str.strip()
        return (lab == str(label_pos).upper().strip()).astype(int)
    if 'H' in df.columns:
        return _as_int_series(df['H'], name='H')
    raise ValueError("Could not derive H: provide --h-col or --label-col/--label-pos, or include an 'H' column.")


def ensure_numeric(series: pd.Series, name: str) -> pd.Series:
    out = pd.to_numeric(series, errors='coerce')
    if out.isna().any():
        raise ValueError(f"Column '{name}' has {int(out.isna().sum())} non-numeric values after coercion.")
    return out.astype(float)


def maybe_filter_age(df: pd.DataFrame, age_col: str | None, age_min: float | None, age_max: float | None) -> pd.DataFrame:
    if not age_col:
        return df
    if age_col not in df.columns:
        raise ValueError(f"--age-col '{age_col}' not found in CSV.")
    age = pd.to_numeric(df[age_col], errors='coerce')
    mask = pd.Series(True, index=df.index)
    if age_min is not None:
        mask &= (age >= float(age_min))
    if age_max is not None:
        mask &= (age <= float(age_max))
    out = df.loc[mask].copy()
    if len(out) == 0:
        raise ValueError("Age filter returned 0 rows. Check --age-col/--age-min/--age-max.")
    return out


# --------------------------------------------------------------------------------------
# Estimators
# --------------------------------------------------------------------------------------

def ppi_analytic_estimator_prob(df_all: pd.DataFrame, labeled_idx: np.ndarray) -> Tuple[float, Tuple[float, float]]:
    """Frequentist PPI with CONTINUOUS probabilities (A_prob = 'p').
    CI via ĝ ± c * sqrt( Var(A_prob)/N + Var(H−A_prob)/n ),
    where c = t_{0.975, n-1} if n<30, else z_{0.975}.
    """
    # Pool component
    A_all = df_all['p'].values  # probability
    N_all = len(A_all)
    A_bar = A_all.mean()
    varA  = A_all.var(ddof=1) if N_all > 1 else 0.0

    # Labeled rectifier
    sub   = df_all.iloc[labeled_idx]
    R     = (sub['H'].astype(float) - sub['p'].astype(float)).values
    n_lab = len(R)
    r_bar = R.mean() if n_lab > 0 else 0.0
    varR  = R.var(ddof=1) if n_lab > 1 else 0.0

    ghat = float(A_bar + r_bar)
    se   = np.sqrt(varA / max(N_all, 1) + varR / max(n_lab, 1)) if (N_all > 0 and n_lab > 0) else 0.0

    if n_lab >= 30:
        crit = 1.959963984540054  # z_{0.975}
    else:
        dfree = max(n_lab - 1, 1)
        crit = float(student_t.ppf(0.975, df=dfree))

    return ghat, (ghat - crit * se, ghat + crit * se)


def difference_estimator(df_all: pd.DataFrame, labeled_idx: np.ndarray, B: int = 1000, rng: np.random.Generator | None = None) -> Tuple[float, Tuple[float, float]]:
    """Design-based difference estimator using thresholded A_class with bootstrap CI."""
    if rng is None:
        rng = np.random.default_rng()
    A_bar = df_all['A_class'].astype(float).mean()
    resid = df_all.iloc[labeled_idx]['H'].astype(float) - df_all.iloc[labeled_idx]['A_class'].astype(float)
    g_hat = float(A_bar + resid.mean())

    n = len(resid)
    boots = []
    for _ in range(B):
        idx = rng.integers(0, n, size=n)
        boots.append(float(A_bar + resid.iloc[idx].mean()))
    lo, hi = np.percentile(boots, [2.5, 97.5])
    return g_hat, (float(lo), float(hi))


def naive_estimator(df_all: pd.DataFrame, labeled_idx: np.ndarray, alpha: float, beta: float, draws: int = 2000, rng: np.random.Generator | None = None) -> Tuple[float, Tuple[float, float]]:
    """Naïve Beta–Bernoulli on H only."""
    if rng is None:
        rng = np.random.default_rng()
    sub = df_all.iloc[labeled_idx]
    n   = len(sub)
    H_sum = int(sub['H'].sum())
    a, b  = alpha + H_sum, beta + n - H_sum
    samp = rng.beta(a, b, size=draws)
    return float(samp.mean()), (float(np.quantile(samp, 0.025)), float(np.quantile(samp, 0.975)))


def chain_rule_estimator(df_all: pd.DataFrame, labeled_idx: np.ndarray, alpha: float, beta: float, draws: int = 500, tune: int = 500, target_accept: float = 0.9) -> Tuple[float, Tuple[float, float]]:
    """Bayesian CRE with Beta priors and Binomial count likelihoods via PyMC (uses A_class)."""
    if pm is None:  # pragma: no cover
        raise RuntimeError(f"PyMC is not available: {_PM_ERR}")

    # Pool counts for A_class
    N   = int(len(df_all))
    NA1 = int(df_all['A_class'].sum())

    # Labeled subset counts for H|A
    sub = df_all.iloc[labeled_idx]
    n1 = int((sub['A_class'] == 1).sum())
    H1 = int(sub.loc[sub['A_class'] == 1, 'H'].sum())
    n0 = int((sub['A_class'] == 0).sum())
    H0 = int(sub.loc[sub['A_class'] == 0, 'H'].sum())

    with pm.Model() as model:
        theta_A  = pm.Beta('theta_A',  alpha, beta)
        theta_H1 = pm.Beta('theta_H1', alpha, beta)
        theta_H0 = pm.Beta('theta_H0', alpha, beta)

        pm.Binomial('obs_A',  N,  theta_A,  observed=NA1)
        pm.Binomial('obs_H1', n1, theta_H1, observed=H1)
        pm.Binomial('obs_H0', n0, theta_H0, observed=H0)

        g = pm.Deterministic('g', theta_A * theta_H1 + (1.0 - theta_A) * theta_H0)

        idata = pm.sample(
            draws=draws,
            tune=tune,
            chains=2,
            cores=1,            # deterministic across machines
            target_accept=target_accept,
            progressbar=False,
            return_inferencedata=True,
        )

    g_samp = idata.posterior['g'].values.reshape(-1)
    return float(g_samp.mean()), (float(np.quantile(g_samp, 0.025)), float(np.quantile(g_samp, 0.975)))


# --------------------------------------------------------------------------------------
# Main
# --------------------------------------------------------------------------------------

def parse_args(argv: Iterable[str]) -> argparse.Namespace:
    p = argparse.ArgumentParser(description="65–70 subset prevalence simulation with CRE/Naive/PPI/Diff.")
    p.add_argument('--pred-csv', type=Path, required=True, help='CSV with predictions (& optionally age).')
    p.add_argument('--pred-col', type=str, default='autorater_prediction', help='Probability column for AD.')
    p.add_argument('--h-col', type=str, default=None, help='Existing binary column for labels (0/1).')
    p.add_argument('--label-col', type=str, default='label', help='String label column (used if --h-col missing).')
    p.add_argument('--label-pos', type=str, default='AD', help='Positive class string in --label-col.')
    p.add_argument('--age-col', type=str, default=None, help='Age column name (numerical). If provided, will filter.')
    p.add_argument('--age-min', type=float, default=65.0, help='Inclusive age lower bound.')
    p.add_argument('--age-max', type=float, default=70.0, help='Inclusive age upper bound.')
    p.add_argument('--threshold', type=float, default=0.5, help='Threshold for A_class from probability.')
    p.add_argument('--nsim', type=int, default=50, help='Number of simulation replicates.')
    p.add_argument('--label-sizes', type=int, nargs='+', default=[10, 20, 40, 80], help='Label budgets.')
    p.add_argument('--diff-bootstrap', type=int, default=1000, help='Bootstrap B for Difference CIs.')
    p.add_argument('--draws', type=int, default=500, help='PyMC draws for CRE.')
    p.add_argument('--tune', type=int, default=500, help='PyMC tune for CRE.')
    p.add_argument('--target-accept', type=float, default=0.90, help='NUTS target_accept for CRE.')
    p.add_argument('--seed', type=int, default=2025, help='Random seed.')
    p.add_argument('--out-csv', type=Path, default=Path('runs/simulation_6570_results.csv'))
    p.add_argument('--priors', type=str, nargs='+', default=['uniform', 'jeffreys'], choices=['uniform', 'jeffreys'])
    return p.parse_args(list(argv))


def main(argv: Iterable[str] | None = None) -> int:
    if argv is None:
        argv = sys.argv[1:]
    args = parse_args(argv)

    rng = np.random.default_rng(args.seed)

    # 1) Load data
    df = pd.read_csv(args.pred_csv)

    # 2) Optional age filter (default 65–70 inclusive if --age-col present)
    df = maybe_filter_age(df, args.age_col, args.age_min, args.age_max)

    # 3) Build H and A_prob/A_class
    H = derive_H(df, h_col=args.h_col, label_col=args.label_col, label_pos=args.label_pos)
    p = ensure_numeric(df[args.pred_col], name=args.pred_col)

    df_work = pd.DataFrame({
        'H': H.astype(int),
        'p': p.astype(float),  # continuous probabilities for PPI analytic
    })
    df_work['A_class'] = (df_work['p'] >= float(args.threshold)).astype(int)

    # 4) Truth from available labels (within the filtered subset)
    g_true = float(df_work['H'].mean())
    N      = int(len(df_work))

    # Map priors
    prior_map: Dict[str, Tuple[float, float]] = {
        'uniform': (1.0, 1.0),
        'jeffreys': (0.5, 0.5),
    }

    # Pre-draw labeled indices per budget ONCE to reuse across priors
    draws_by_nh: Dict[int, List[np.ndarray]] = {
        nh: [rng.choice(N, size=nh, replace=False) for _ in range(args.nsim)]
        for nh in args.label_sizes if nh <= N
    }

    # 5) Simulation loop
    results: List[Dict[str, float]] = []
    for pname in args.priors:
        alpha, beta = prior_map[pname]
        for nh in args.label_sizes:
            if nh > N:
                print(f"[WARN] Skipping n_labels={nh} (>N={N})")
                continue

            cov = {'chain': 0, 'naive': 0, 'ppi': 0, 'diff': 0}
            wid = {'chain': [], 'naive': [], 'ppi': [], 'diff': []}

            for idx in draws_by_nh[nh]:
                # CRE (A_class)
                m_c, (l_c, h_c) = chain_rule_estimator(
                    df_work, idx, alpha=alpha, beta=beta,
                    draws=args.draws, tune=args.tune, target_accept=args.target_accept
                )
                cov['chain'] += int(l_c <= g_true <= h_c)
                wid['chain'].append(h_c - l_c)

                # Naïve (H only)
                m_n, (l_n, h_n) = naive_estimator(df_work, idx, alpha=alpha, beta=beta, rng=rng)
                cov['naive'] += int(l_n <= g_true <= h_n)
                wid['naive'].append(h_n - l_n)

                # PPI analytic (continuous prob + small-n t)
                m_p, (l_p, h_p) = ppi_analytic_estimator_prob(df_work, idx)
                cov['ppi'] += int(l_p <= g_true <= h_p)
                wid['ppi'].append(h_p - l_p)

                # Difference (A_class) with bootstrap
                m_d, (l_d, h_d) = difference_estimator(df_work, idx, B=args.diff_bootstrap, rng=rng)
                cov['diff'] += int(l_d <= g_true <= h_d)
                wid['diff'].append(h_d - l_d)

            results.append({
                'prior': pname,
                'n_labels': int(nh),
                'chain_cov': cov['chain'] / args.nsim,
                'chain_w': float(np.mean(wid['chain'])) if wid['chain'] else float('nan'),
                'naive_cov': cov['naive'] / args.nsim,
                'naive_w': float(np.mean(wid['naive'])) if wid['naive'] else float('nan'),
                'ppi_cov': cov['ppi'] / args.nsim,
                'ppi_w': float(np.mean(wid['ppi'])) if wid['ppi'] else float('nan'),
                'diff_cov': cov['diff'] / args.nsim,
                'diff_w': float(np.mean(wid['diff'])) if wid['diff'] else float('nan'),
            })

    out_df = pd.DataFrame(results)
    args.out_csv.parent.mkdir(parents=True, exist_ok=True)
    out_df.to_csv(args.out_csv, index=False)

    # Console preview
    print("\n=== 65–70 Subset Simulation Summary ===")
    if args.age_col:
        print(f"Age filter: {args.age_min} ≤ {args.age_col} ≤ {args.age_max}")
    print(f"N={N}  g_true={g_true:.4f}  threshold={args.threshold}")
    print(out_df.to_string(index=False))
    print(f"\n✅ Saved: {args.out_csv}")
    return 0


if __name__ == '__main__':  # pragma: no cover
    raise SystemExit(main())
