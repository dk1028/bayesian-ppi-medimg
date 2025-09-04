# age_effect_analysis_bins_50_73_74_79_80_100.py
# -*- coding: utf-8 -*-
"""
Autorater vs Age analysis (bins: 50–73, 74–79, 80–100)
- Merge predictions with ADNI metadata (Subject + Acq Date exact, then ±14 days fallback)
- Scatter: predicted P(AD) vs Age (colored by H)
- ROC: overall + per age bin, plus combined overlay
- Metrics per bin: n, prevalence, AUC, ACC/TPR/TNR @0.5
- AUC 95% bootstrap CI per bin + overall
- Youden’s J optimal threshold per bin (ACC/TPR/TNR @thr)
- Calibration curves + Brier score per bin (if sufficient data)
Outputs: figures (PNG) + CSV summaries
"""

import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from pathlib import Path
from sklearn.metrics import roc_curve, auc, roc_auc_score, accuracy_score, confusion_matrix, brier_score_loss
from sklearn.calibration import calibration_curve

# ============================
# 0) Paths & parameters
# ============================
CSV_META = Path(r"C:\Users\AV75950\Downloads\all_people_7_20_2025.csv")       # ADNI Data Collections: Subject, Age, Sex, Acq Date ...
CSV_PRED = Path(r"C:\Users\AV75950\Documents\autorater_predictions_all4.csv") # Autorater results: subject_id, Acq_Date, autorater_prediction, H, label ...
OUT_DIR  = Path(r"C:\Users\AV75950\Documents\autorater_age_analysis3")
OUT_DIR.mkdir(parents=True, exist_ok=True)

# Age bins (right edge exclusive) — 50–73, 74–79, 80–100
AGE_BINS    = [50, 74, 80, 101]
AGE_LABELS  = ["50–73", "74–79", "80–100"]

# If you want a standalone ROC for a specific age band (e.g., 74–79)
AGE_BAND_FOR_ROC = (74, 80)   # [low, high)

# Allowed day tolerance for nearest-date matching
DATE_TOL_DAYS = 14

# Number of bootstrap repetitions
BOOT_B = 2000

# Minimum sample size to draw calibration curves
CALIB_MIN_N = 50

# ============================
# 1) Load & clean data
# ============================
meta = pd.read_csv(CSV_META, dtype=str)
pred = pd.read_csv(CSV_PRED, dtype=str)

# Trim whitespace
meta.columns = [c.strip() for c in meta.columns]
pred.columns = [c.strip() for c in pred.columns]

# Required columns check
need_meta_cols = ['Subject', 'Age', 'Sex', 'Acq Date']
for c in need_meta_cols:
    if c not in meta.columns:
        raise ValueError(f"[META] Missing column '{c}'. Current columns: {list(meta.columns)}")

need_pred_cols = ['subject_id', 'Acq_Date', 'autorater_prediction', 'H']
for c in need_pred_cols:
    if c not in pred.columns:
        raise ValueError(f"[PRED] Missing column '{c}'. Current columns: {list(pred.columns)}")

# Type/date parsing
meta['Subject'] = meta['Subject'].str.strip()
pred['subject_id'] = pred['subject_id'].str.strip()

def parse_date_series(s):
    out = pd.to_datetime(s, errors='coerce')
    mask = out.isna()
    if mask.any():
        try_formats = ['%m/%d/%Y', '%m/%d/%y', '%Y-%m-%d', '%Y/%m/%d', '%m-%d-%Y', '%d/%m/%Y']
        for fmt in try_formats:
            out2 = pd.to_datetime(s[mask], format=fmt, errors='coerce')
            out.loc[mask] = out2
            mask = out.isna()
            if not mask.any():
                break
    return out.dt.date

meta['AcqDate_std'] = parse_date_series(meta['Acq Date'])
pred['AcqDate_std'] = parse_date_series(pred['Acq_Date'])

# Numeric casting
meta['Age'] = pd.to_numeric(meta['Age'], errors='coerce')
pred['autorater_prediction'] = pd.to_numeric(pred['autorater_prediction'], errors='coerce')
pred['H'] = pd.to_numeric(pred['H'], errors='coerce').astype('Int64')  # 0/1
# If label exists, make a reference is_AD
if 'label' in pred.columns:
    pred['is_AD'] = (pred['label'].astype(str).str.upper() == 'AD').astype(int)

# ============================
# 2) Matching (exact → ±14 days nearest)
# ============================
meta_key = meta[['Subject', 'AcqDate_std', 'Age', 'Sex']].drop_duplicates()
merged = pred.merge(
    meta_key, left_on=['subject_id', 'AcqDate_std'], right_on=['Subject', 'AcqDate_std'],
    how='left', suffixes=('', '_meta')
)

# Nearest-date matching (if needed)
need_fill = merged['Age'].isna()
if need_fill.any():
    print(f"[INFO] Exact match failed rows: {need_fill.sum()} → trying nearest-date matching (±{DATE_TOL_DAYS} days)")
    meta_grp = {sid: df[['AcqDate_std', 'Age', 'Sex']].dropna(subset=['AcqDate_std']).sort_values('AcqDate_std')
                for sid, df in meta.groupby('Subject', sort=False)}

    ages, sexes = [], []
    for idx, row in merged.loc[need_fill].iterrows():
        sid = row['subject_id']
        d0  = row['AcqDate_std']
        age_val, sex_val = np.nan, np.nan
        if pd.notna(d0) and sid in meta_grp:
            cand = meta_grp[sid]
            diffs = cand['AcqDate_std'].apply(lambda d: abs(pd.to_datetime(d) - pd.to_datetime(d0))).dt.days
            j = diffs.idxmin() if len(diffs) else None
            if j is not None and diffs.loc[j] <= DATE_TOL_DAYS:
                age_val = cand.loc[j, 'Age']
                sex_val = cand.loc[j, 'Sex']
        ages.append(age_val); sexes.append(sex_val)
    merged.loc[need_fill, 'Age'] = ages
    merged.loc[need_fill, 'Sex'] = sexes

# Usable data
use = merged.dropna(subset=['Age', 'autorater_prediction', 'H']).copy()
use['Age'] = use['Age'].astype(float)
use['H']   = use['H'].astype(int)

print(f"[INFO] Final matches: {len(use)} / {len(pred)}")

# ============================
# 3) Scatter: P(AD) vs Age
# ============================
plt.figure(figsize=(8.5, 5.2))
colors = np.where(use['H']==1, 'crimson', 'royalblue')
plt.scatter(use['Age'], use['autorater_prediction'], c=colors, s=16, alpha=0.55, edgecolors='none')
plt.xlabel("Age (years)")
plt.ylabel("Autorater predicted P(AD)")
plt.title("Predicted AD Probability vs Age (colored by H)")
from matplotlib.lines import Line2D
legend_elems = [
    Line2D([0],[0], marker='o', color='w', label='CN (H=0)', markerfacecolor='royalblue', markersize=7),
    Line2D([0],[0], marker='o', color='w', label='AD (H=1)', markerfacecolor='crimson',  markersize=7),
]
plt.legend(handles=legend_elems, loc='lower right')
plt.grid(alpha=0.25, linestyle='--')
plt.tight_layout()
plt.savefig(OUT_DIR / "fig_pred_vs_age.png", dpi=200)
plt.close()

# ============================
# 4) ROC utility function
# ============================
def plot_roc(y_true, y_score, title, savepath):
    fpr, tpr, _ = roc_curve(y_true, y_score)
    roc_auc = auc(fpr, tpr)
    plt.figure(figsize=(5.6, 5.6))
    plt.plot(fpr, tpr, lw=2, label=f"AUC = {roc_auc:.3f}")
    plt.plot([0,1],[0,1],'k--', lw=1)
    plt.xlim(0,1); plt.ylim(0,1.03)
    plt.xlabel("False Positive Rate (1 - Specificity)")
    plt.ylabel("True Positive Rate (Sensitivity)")
    plt.title(title)
    plt.legend(loc='lower right')
    plt.grid(alpha=0.25, linestyle='--')
    plt.tight_layout()
    plt.savefig(savepath, dpi=200)
    plt.close()
    return roc_auc

# ============================
# 5) Overall ROC
# ============================
auc_overall = plot_roc(use['H'].values, use['autorater_prediction'].values,
                       "ROC (All ages: 50–100)", OUT_DIR / "fig_roc_overall.png")

# Specific age band ROC (optional: 74–79)
low, high = AGE_BAND_FOR_ROC
band_df = use[(use['Age'] >= low) & (use['Age'] < high)].copy()
if len(band_df) >= 10 and band_df['H'].nunique() == 2:
    auc_band = plot_roc(band_df['H'].values, band_df['autorater_prediction'].values,
                        f"ROC (Age {low}–{high-1})", OUT_DIR / f"fig_roc_{low}_{high-1}.png")
else:
    auc_band = np.nan

# ============================
# 6) Age bin assignment (50–73, 74–79, 80–100)
# ============================
use['age_bin'] = pd.cut(use['Age'], bins=AGE_BINS, labels=AGE_LABELS,
                        right=False, include_lowest=True)

# ============================
# 7) Per-bin ROC/metrics @0.5
# ============================
rows = []
for b in AGE_LABELS:
    sub = use[use['age_bin'] == b]
    if len(sub) < 20 or sub['H'].nunique() < 2:
        # Skip if insufficient samples/classes
        continue
    y = sub['H'].values
    p = sub['autorater_prediction'].values
    # Save AUC/ROC
    auc_b = plot_roc(y, p, f"ROC (Age {b})", OUT_DIR / f"fig_roc_{b.replace('–','_')}.png")
    # @0.5 threshold
    yhat = (p >= 0.5).astype(int)
    acc = accuracy_score(y, yhat)
    tn, fp, fn, tp = confusion_matrix(y, yhat, labels=[0,1]).ravel()
    tpr = tp / (tp + fn) if (tp + fn) else np.nan
    tnr = tn / (tn + fp) if (tn + fp) else np.nan
    prev = y.mean()
    rows.append({
        'age_bin': b,
        'n': len(sub),
        'prevalence_AD': prev,
        'AUC': auc_b,
        'ACC@0.5': acc,
        'TPR@0.5': tpr,
        'TNR@0.5': tnr
    })

perf = pd.DataFrame(rows).sort_values('age_bin')
perf.to_csv(OUT_DIR / "metrics_by_age.csv", index=False, encoding='utf-8-sig')

# Overlay ROC by age bins
plt.figure(figsize=(7.6, 6))
for b in AGE_LABELS:
    sub = use[use['age_bin']==b]
    if len(sub) < 20 or sub['H'].nunique() < 2:
        continue
    fpr, tpr, _ = roc_curve(sub['H'].values, sub['autorater_prediction'].values)
    roc_auc = auc(fpr, tpr)
    plt.plot(fpr, tpr, lw=1.8, label=f"{b} (AUC {roc_auc:.2f})")
plt.plot([0,1],[0,1],'k--', lw=1)
plt.xlabel("False Positive Rate")
plt.ylabel("True Positive Rate")
plt.title("ROC by Age Bins (50–73, 74–79, 80–100)")
plt.legend(loc='lower right', fontsize=9)
plt.grid(alpha=0.25)
plt.tight_layout()
plt.savefig(OUT_DIR / "fig_roc_by_age_bins.png", dpi=200)
plt.close()

# ============================
# 8) AUC 95% bootstrap CI (overall & per-bin)
# ============================
rng = np.random.default_rng(2025)
def auc_ci(y, p, B=2000):
    y = np.asarray(y); p = np.asarray(p)
    auc0 = roc_auc_score(y, p)
    boots = []
    n = len(y)
    for _ in range(B):
        idx = rng.integers(0, n, size=n)
        if len(np.unique(y[idx])) < 2:
            continue
        boots.append(roc_auc_score(y[idx], p[idx]))
    if len(boots):
        lo, hi = np.percentile(boots, [2.5, 97.5])
    else:
        lo, hi = (np.nan, np.nan)
    return auc0, lo, hi

auc0, lo, hi = auc_ci(use['H'], use['autorater_prediction'], B=BOOT_B)
print(f"\nOverall AUC = {auc0:.3f}  (95% CI {lo:.3f}–{hi:.3f})")

ci_rows = []
for b in AGE_LABELS:
    sub = use[use['age_bin']==b]
    if len(sub) >= 20 and sub['H'].nunique()==2:
        a0, l, h = auc_ci(sub['H'], sub['autorater_prediction'], B=BOOT_B)
    else:
        a0, l, h = (np.nan, np.nan, np.nan)
    ci_rows.append({'age_bin': b, 'AUC_CI_low': l, 'AUC_CI_high': h})
perf_ci = perf.merge(pd.DataFrame(ci_rows), on='age_bin', how='left')
perf_ci.to_csv(OUT_DIR / "metrics_by_age_with_auc_ci.csv", index=False, encoding='utf-8-sig')

# ============================
# 9) Youden optimal threshold & metrics
# ============================
def youden_threshold(y, p):
    fpr, tpr, thr = roc_curve(y, p)
    j = tpr - fpr
    k = np.argmax(j)
    return thr[k], tpr[k], (1 - fpr[k])

rows_thr = []
for b in AGE_LABELS:
    sub = use[use['age_bin']==b]
    if len(sub) < 20 or sub['H'].nunique() < 2:
        continue
    y = sub['H'].values
    p = sub['autorater_prediction'].values
    thr, tpr_star, tnr_star = youden_threshold(y, p)
    yhat_star = (p >= thr).astype(int)
    acc_star  = accuracy_score(y, yhat_star)
    rows_thr.append({
        'age_bin': b,
        'thr_youden': float(thr),
        'ACC@thr': acc_star,
        'TPR@thr': float(tpr_star),
        'TNR@thr': float(tnr_star),
    })

thr_df = pd.DataFrame(rows_thr).sort_values('age_bin')
thr_df.to_csv(OUT_DIR / "metrics_by_age_youden.csv", index=False, encoding='utf-8-sig')

# ============================
# 10) Calibration curves & Brier (per-bin, only when sufficient)
# ============================
cal_dir = OUT_DIR / "calibration_curves"
cal_dir.mkdir(exist_ok=True)

cal_rows = []
for b in AGE_LABELS:
    sub = use[use['age_bin']==b]
    if len(sub) < CALIB_MIN_N or sub['H'].nunique() < 2:
        continue
    y = sub['H'].values
    p = sub['autorater_prediction'].values
    frac_pos, mean_pred = calibration_curve(y, p, n_bins=10, strategy='uniform')
    bs = brier_score_loss(y, p)

    plt.figure(figsize=(5.2,5))
    plt.plot(mean_pred, frac_pos, 'o-', label='Observed')
    plt.plot([0,1],[0,1], 'k--', label='Ideal')
    plt.xlabel("Predicted probability")
    plt.ylabel("Observed fraction of AD")
    plt.title(f"Calibration (Age {b})  Brier={bs:.3f}")
    plt.legend()
    plt.grid(alpha=0.25)
    plt.tight_layout()
    plt.savefig(cal_dir / f"calibration_{b.replace('–','_')}.png", dpi=200)
    plt.close()

    cal_rows.append({'age_bin': b, 'Brier': bs})

pd.DataFrame(cal_rows).to_csv(OUT_DIR / "calibration_by_age.csv", index=False, encoding='utf-8-sig')

# ============================
# 11) Summary print
# ============================
print("\n=== SUMMARY ===")
print(f"Overall AUC (50–100): {auc_overall:.3f}")
if not np.isnan(auc_band):
    print(f"AUC (Age {low}–{high-1}): {auc_band:.3f}")
else:
    print(f"AUC (Age {low}–{high-1}): NA (insufficient data)")

print("\nPer-age-bin metrics @0.5 threshold:")
if len(perf):
    print(perf.to_string(index=False))
else:
    print("  (no bins had enough size or both classes)")

print("\nPer-age-bin AUC 95% CI:")
if len(perf_ci):
    print(perf_ci[['age_bin','n','AUC','AUC_CI_low','AUC_CI_high']].to_string(index=False))
else:
    print("  (no bins had enough size or both classes)")

print("\nPer-age-bin optimal thresholds (Youden) & metrics:")
if len(thr_df):
    print(thr_df.to_string(index=False))
else:
    print("  (no bins had enough size or both classes)")

print("\n✅ Saved:")
print(f"- {OUT_DIR / 'fig_pred_vs_age.png'}")
print(f"- {OUT_DIR / 'fig_roc_overall.png'}")
print(f"- {OUT_DIR / 'fig_roc_by_age_bins.png'}")
print(f"- {OUT_DIR / 'metrics_by_age.csv'}")
print(f"- {OUT_DIR / 'metrics_by_age_with_auc_ci.csv'}")
print(f"- {OUT_DIR / 'metrics_by_age_youden.csv'}")
print(f"- {OUT_DIR / 'calibration_by_age.csv'} (and PNGs in {cal_dir})")
